#!/usr/bin/env python3
"""
run_experiment.py
=================
End-to-end training and evaluation for the HEA Transformer.

Usage
-----
  python scripts/run_experiment.py --quick    # smoke test (~2 min, CPU)
  python scripts/run_experiment.py            # full run  (~20 min, CPU)

Stages
------
  1. Dataset generation (MC-sampled FCC supercells)
  2. Train/val/test split + tokenisation
  3. Sklearn baseline models
  4. MLM pre-training
  5. SRO fine-tuning (physics-informed)
  6. Quick phase head training
  7. Evaluation + physics consistency check
  8. Plot generation
  9. Save results JSON
"""

import sys, os, time, json, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.supercell_generator import build_dataset, N_SP, CANTOR, VOCAB
from src.data.tokenizer           import SiteOccupancyTokenizer, SequenceDataset
from src.models.transformer       import HEAEncoder, MLMHead, SROHead, PhaseHead
from src.training.losses          import cross_entropy, sro_loss, Adam
from src.training.trainer         import pretrain_epoch, finetune_epoch, \
                                         evaluate, quick_phase_train
from src.evaluation.metrics       import compute_metrics, physics_check
from src.evaluation.baselines     import run_all_baselines, \
                                         frozen_transformer_baseline
from src.evaluation.plots         import (
    plot_architecture, plot_tokenization, plot_sro_parity,
    plot_attention, plot_ablation, plot_scenario_mae,
    plot_training_curves, plot_sro_matrix, plot_confusion,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def get_config(quick=False):
    if quick:
        return dict(
            nx=3, ny=3, nz=3,           # 108 atoms — minimum for correct 2NN SRO
            n_per_scenario=30,
            n_mc_steps=2000,
            scenarios=['random', 'ordering', 'cluster'],
            n_epochs_pretrain=3,
            n_epochs_finetune=10,
            d_model=32, n_heads=2, n_layers=2,
            d_ff=64, d_emb=8,
            lr=1e-3, wd=1e-4,
            batch_size=8, mask_prob=0.15, seed=42,
        )
    return dict(
        nx=3, ny=3, nz=3,
        n_per_scenario=100,
        n_mc_steps=12000,
        scenarios=['random', 'ordering', 'cluster', 'mixed', 'CrMn_ordering'],
        n_epochs_pretrain=6,
        n_epochs_finetune=25,
        d_model=64, n_heads=4, n_layers=3,
        d_ff=128, d_emb=16,
        lr=5e-4, wd=1e-4,
        batch_size=16, mask_prob=0.15, seed=42,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    args   = parser.parse_args()
    cfg    = get_config(args.quick)

    rng    = np.random.default_rng(cfg['seed'])
    np.random.seed(cfg['seed'])

    print("=" * 68)
    print("HEAFormer: Site-Occupancy Transformer for SRO Prediction")
    print("=" * 68)
    cfg_print = {k: v for k, v in cfg.items() if k != 'scenarios'}
    print(json.dumps(cfg_print, indent=2))

    # ------------------------------------------------------------------
    # 1. Dataset
    # ------------------------------------------------------------------
    print("\n[1/9] Generating dataset ...")
    t0 = time.time()
    ds = build_dataset(
        nx=cfg['nx'], ny=cfg['ny'], nz=cfg['nz'],
        n_per_scenario=cfg['n_per_scenario'],
        n_mc_steps=cfg['n_mc_steps'],
        scenarios=cfg['scenarios'],
        n_shells=2, seed=cfg['seed'], verbose=True,
    )
    print(f"  Generated in {time.time()-t0:.1f}s")

    N_atoms = ds['meta']['N']
    sro_dim = ds['meta']['n_sro']
    scenarios = ds['meta']['scenarios']

    # ------------------------------------------------------------------
    # 2. Tokenise + split
    # ------------------------------------------------------------------
    print("\n[2/9] Tokenising and splitting ...")
    tok = SiteOccupancyTokenizer(n_shells=2)
    seq = SequenceDataset(ds, tok, mask_prob=cfg['mask_prob'], seed=cfg['seed'])

    n_total  = len(seq)
    perm     = rng.permutation(n_total)
    n_tr     = int(0.70 * n_total)
    n_va     = int(0.15 * n_total)
    tr_idx   = perm[:n_tr].tolist()
    va_idx   = perm[n_tr:n_tr+n_va].tolist()
    te_idx   = perm[n_tr+n_va:].tolist()
    print(f"  train={len(tr_idx)}  val={len(va_idx)}  test={len(te_idx)}")
    print(f"  feat_dim={tok.feat_dim}   sro_dim={sro_dim}")

    # quick figures from data
    print("\n  Generating data figures ...")
    plot_architecture()
    s0 = seq[tr_idx[0]]
    plot_tokenization(ds['occupancy'][tr_idx[0]],
                      s0['token_ids'], s0['mask'])
    for si, sc in enumerate(scenarios[:3]):
        idx_sc = np.where(ds['scenario_ids'] == si)[0]
        if len(idx_sc):
            plot_sro_matrix(ds['labels'][idx_sc[0]]['alpha'],
                            shell=0, title=sc)

    # ------------------------------------------------------------------
    # 3. Baselines
    # ------------------------------------------------------------------
    print("\n[3/9] Running sklearn baselines ...")
    bl = run_all_baselines(seq, tr_idx, te_idx, verbose=True)

    # ------------------------------------------------------------------
    # 4. Init model
    # ------------------------------------------------------------------
    print("\n[4/9] Initialising HEAFormer ...")
    encoder = HEAEncoder(
        vocab=VOCAB, feat_dim=tok.feat_dim,
        d_model=cfg['d_model'], n_heads=cfg['n_heads'],
        n_layers=cfg['n_layers'], d_ff=cfg['d_ff'],
        d_emb=cfg['d_emb'], seed=cfg['seed'],
    )
    mlm_head   = MLMHead(cfg['d_model'], VOCAB)
    sro_head   = SROHead(cfg['d_model'], sro_dim)
    phase_head = PhaseHead(cfg['d_model'], n_classes=3)

    n_p = sum(p.size for p in encoder.all_params())
    print(f"  Encoder params: {n_p:,}")

    # ------------------------------------------------------------------
    # 5. MLM pre-training
    # ------------------------------------------------------------------
    print(f"\n[5/9] MLM pre-training ({cfg['n_epochs_pretrain']} epochs) ...")
    pre_opt = Adam(lr=cfg['lr']*2, wd=0)
    pre_hist = []
    for ep in range(cfg['n_epochs_pretrain']):
        r = pretrain_epoch(encoder, mlm_head, seq, tr_idx, pre_opt, rng)
        pre_hist.append(r['mlm_loss'])
        print(f"  Ep {ep+1:2d}  MLM loss={r['mlm_loss']:.4f}  "
              f"acc={r['mlm_acc']:.4f}")

    # ------------------------------------------------------------------
    # 6. SRO fine-tuning
    # ------------------------------------------------------------------
    print(f"\n[6/9] SRO fine-tuning ({cfg['n_epochs_finetune']} epochs) ...")
    ft_opt = Adam(lr=cfg['lr'], wd=cfg['wd'])
    tr_loss_hist, va_mae_hist, tr_mae_hist = [], [], []

    for ep in range(cfg['n_epochs_finetune']):
        r_tr = finetune_epoch(
            encoder, sro_head, seq, tr_idx, ft_opt, rng,
            n_shells=2, n_sp=N_SP,
            batch_size=cfg['batch_size'],
        )
        r_va = evaluate(encoder, sro_head, None, seq, va_idx)
        tr_loss_hist.append(r_tr['loss'])
        tr_mae_hist.append(r_tr['sro_mae'])
        va_mae_hist.append(r_va['sro_mae'])
        print(f"  Ep {ep+1:2d}  loss={r_tr['loss']:.4f}  "
              f"tr_MAE={r_tr['sro_mae']:.4f}  "
              f"va_MAE={r_va['sro_mae']:.4f}")

    history = dict(
        pretrain_loss=pre_hist,
        train_loss=tr_loss_hist,
        val_loss=va_mae_hist,
        train_mae=tr_mae_hist,
        val_mae=va_mae_hist,
    )
    plot_training_curves(history)

    # ------------------------------------------------------------------
    # 7. Phase head + frozen baseline
    # ------------------------------------------------------------------
    print("\n[7/9] Phase head training + frozen encoder baseline ...")
    quick_phase_train(encoder, phase_head, seq, tr_idx,
                      lr=cfg['lr'], epochs=4, rng=rng)

    frozen_bl = frozen_transformer_baseline(encoder, seq,
                                             tr_idx, te_idx, verbose=True)

    # ------------------------------------------------------------------
    # 8. Evaluation
    # ------------------------------------------------------------------
    print("\n[8/9] Evaluating on test set ...")
    res = evaluate(encoder, sro_head, phase_head, seq, te_idx)

    sro_p = np.array([sro_head.forward(
                encoder.forward(seq[i]['token_ids'],
                                seq[i]['features'])[0])
             for i in te_idx])
    sro_t = np.array([seq[i]['alpha_flat'] for i in te_idx])
    ph_p  = res.get('phase_pred', np.zeros(len(te_idx), int))
    ph_t  = res.get('phase_true', np.array([seq[i]['phase_label']
                                             for i in te_idx]))

    tx_metrics = compute_metrics(sro_p, sro_t, ph_p, ph_t, split='TEST')

    # physics consistency
    xmean = np.ones(N_SP) / N_SP
    phys  = physics_check(sro_p, xmean, 2, N_SP)
    print(f"\n  Physics constraint violations: {phys}")

    # per-scenario MAE
    sc_ids = ds['scenario_ids'][te_idx]
    sc_mae_trans = {}
    for si, sc in enumerate(scenarios):
        m_ = sc_ids == si
        if m_.sum():
            sc_mae_trans[sc] = float(np.mean(np.abs(sro_p[m_] - sro_t[m_])))
    print(f"  Per-scenario MAE: {sc_mae_trans}")

    # per-scenario MAE for best baseline (RF)
    _, y_sro_te, _ = seq.flat_features(te_idx)
    sc_mae_rf = {}
    # (use the RF model from bl['sro']['RandomForest'] if it was trained)
    # We recompute for simplicity via a quick RF
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    _, y_sro_tr, _ = seq.flat_features(tr_idx)
    Xtr_f, _, _    = seq.flat_features(tr_idx)
    Xte_f, _, _    = seq.flat_features(te_idx)
    rf = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    rf.fit(Xtr_f, y_sro_tr)
    rf_pred = rf.predict(Xte_f)
    for si, sc in enumerate(scenarios):
        m_ = sc_ids == si
        if m_.sum():
            sc_mae_rf[sc] = float(
                np.mean(np.abs(rf_pred[m_] - y_sro_te[m_])))

    # ------------------------------------------------------------------
    # 9. Figures
    # ------------------------------------------------------------------
    print("\n[9/9] Generating figures ...")

    plot_sro_parity(sro_p, sro_t,
                    model_name='HEAFormer',
                    mae=tx_metrics['sro_mae'],
                    r2=tx_metrics['sro_r2'])

    # attention
    attn_maps = res.get('attn_maps', [])
    if attn_maps:
        plot_attention(attn_maps[0],
                       ds['occupancy'][te_idx[0]],
                       layer=cfg['n_layers']-1, head=0,
                       n_show=min(20, N_atoms))

    # ablation: merge all SRO results
    all_sro_results = {}
    for nm, m in bl['sro'].items():
        if 'mae' in m:
            all_sro_results[nm] = m
    all_sro_results['FrozenEncoder+Ridge'] = frozen_bl.get(
        'sro_frozen+Ridge', {})
    all_sro_results['HEAFormer (ours)'] = dict(
        mae=tx_metrics['sro_mae'],
        r2 =tx_metrics['sro_r2'],
        rmse=tx_metrics['sro_rmse'],
    )
    plot_ablation(all_sro_results, metric='mae', task='SRO')
    plot_ablation(all_sro_results, metric='r2',  task='SRO')

    all_ph_results = {}
    for nm, m in bl['phase'].items():
        if 'acc' in m:
            all_ph_results[nm] = m
    all_ph_results['HEAFormer (ours)'] = dict(
        acc=tx_metrics['phase_acc'],
        f1 =tx_metrics['phase_f1'],
    )
    plot_ablation(all_ph_results, metric='acc', task='Phase')

    # per-scenario MAE chart
    if sc_mae_trans and sc_mae_rf:
        scenario_maes = {
            'HEAFormer (ours)': sc_mae_trans,
            'RandomForest':     sc_mae_rf,
        }
        plot_scenario_mae(scenario_maes, list(scenario_maes.keys()))

    # confusion matrix
    if tx_metrics.get('phase_cm'):
        plot_confusion(tx_metrics['phase_cm'], title='HEAFormer test set')

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("RESULTS SUMMARY")
    print("=" * 68)
    print(f"\nSRO Regression (test N={len(te_idx)}):")
    print(f"  {'Model':<32} {'MAE':>8}  {'R²':>8}  {'RMSE':>8}")
    print(f"  {'-'*60}")
    all_sro_sorted = sorted(all_sro_results.items(),
                             key=lambda x: x[1].get('mae', 9))
    for nm, m in all_sro_sorted:
        if 'mae' in m:
            print(f"  {nm:<32} {m['mae']:>8.4f}  "
                  f"{m.get('r2', float('nan')):>8.4f}  "
                  f"{m.get('rmse', float('nan')):>8.4f}")

    print(f"\nPhase Classification (test):")
    print(f"  {'Model':<32} {'Acc':>8}  {'F1':>8}")
    print(f"  {'-'*45}")
    for nm, m in sorted(all_ph_results.items(),
                         key=lambda x: -x[1].get('acc', 0)):
        if 'acc' in m:
            print(f"  {nm:<32} {m['acc']:>8.4f}  {m.get('f1',0):>8.4f}")

    print(f"\nPhysics constraint violations (predicted SRO):")
    for k, v in phys.items():
        print(f"  {k}: {v:.3e}")

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    results = dict(
        config=cfg,
        baselines_sro={k: {kk: float(vv) for kk, vv in v.items()
                           if isinstance(vv, float)}
                       for k, v in bl['sro'].items()},
        baselines_phase={k: {kk: float(vv) for kk, vv in v.items()
                             if isinstance(vv, float)}
                         for k, v in bl['phase'].items()},
        heatformer=tx_metrics,
        physics=phys,
        scenario_mae_transformer=sc_mae_trans,
        scenario_mae_rf=sc_mae_rf,
        history={k: [float(x) for x in v]
                 for k, v in history.items()},
    )
    out_json = os.path.join(ROOT, 'reports', 'results.json')
    with open(out_json, 'w') as f:
        # cm is not JSON-serialisable as np arrays
        safe = json.loads(json.dumps(results, default=lambda x:
               float(x) if hasattr(x,'item') else str(x)))
        json.dump(safe, f, indent=2)
    print(f"\nResults saved to: {out_json}")
    print("Figures saved to: figures/")
    print("\n[DONE]")


if __name__ == '__main__':
    main()
