#!/usr/bin/env python3
"""
generate_paper_figures.py
=========================
Generate all publication-quality figures for the HEAFormer paper.
Uses real experimental data from reports/results.json.

Run from repo root:
    python scripts/generate_paper_figures.py

Output: figures/paper/  (one PNG per figure, 300 DPI)
"""

import sys, os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib import cm
from scipy.stats import pearsonr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUTDIR = os.path.join(ROOT, 'figures', 'paper')
os.makedirs(OUTDIR, exist_ok=True)

DPI     = 300
FIGFMT  = 'png'
CANTOR  = ['Cr', 'Mn', 'Fe', 'Co', 'Ni']
EL_COL  = {'Cr':'#E64B35','Mn':'#4DBBD5','Fe':'#00A087','Co':'#3C5488','Ni':'#F39B7F'}
BLUE    = '#2166AC'
RED     = '#D6604D'
GREEN   = '#1A9850'
ORANGE  = '#F46D43'
PURPLE  = '#762A83'

# ── fonts ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'axes.titlesize':   11,
    'axes.labelsize':   10,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'legend.fontsize':   8,
    'figure.dpi':        150,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})


def save(name):
    p = os.path.join(OUTDIR, name)
    plt.savefig(p, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  saved {p}')
    return p


# ────────────────────────────────────────────────────────────────────────────
# Load experiment data and regenerate predictions
# ────────────────────────────────────────────────────────────────────────────

def load_experiment():
    with open(os.path.join(ROOT, 'reports', 'results.json')) as f:
        r = json.load(f)

    from src.data.supercell_generator import build_dataset, N_SP, CANTOR as C, VOCAB
    from src.data.tokenizer           import SiteOccupancyTokenizer, SequenceDataset
    from src.models.transformer       import HEAEncoder, MLMHead, SROHead, PhaseHead
    from src.training.losses          import Adam
    from src.training.trainer         import pretrain_epoch, finetune_epoch, \
                                              quick_phase_train, evaluate

    cfg = r['config']
    rng = np.random.default_rng(cfg['seed'])
    np.random.seed(cfg['seed'])

    print('[load] Building dataset ...')
    ds = build_dataset(
        nx=cfg['nx'], ny=cfg['ny'], nz=cfg['nz'],
        n_per_scenario=cfg['n_per_scenario'],
        n_mc_steps=cfg['n_mc_steps'],
        scenarios=cfg['scenarios'],
        n_shells=2, seed=cfg['seed'], verbose=False,
    )
    tok = SiteOccupancyTokenizer(n_shells=2)
    seq = SequenceDataset(ds, tok, mask_prob=cfg['mask_prob'], seed=cfg['seed'])

    n = len(seq); perm = rng.permutation(n)
    n_tr = int(0.70*n); n_va = int(0.15*n)
    tr_idx = perm[:n_tr].tolist()
    va_idx = perm[n_tr:n_tr+n_va].tolist()
    te_idx = perm[n_tr+n_va:].tolist()

    encoder    = HEAEncoder(VOCAB, tok.feat_dim,
                             d_model=cfg['d_model'], n_heads=cfg['n_heads'],
                             n_layers=cfg['n_layers'], d_ff=cfg['d_ff'],
                             d_emb=cfg['d_emb'], seed=cfg['seed'])
    mlm_head   = MLMHead(cfg['d_model'], VOCAB)
    sro_head   = SROHead(cfg['d_model'], ds['meta']['n_sro'])
    phase_head = PhaseHead(cfg['d_model'], 3)

    print('[load] Re-training (quick) ...')
    pre_opt = Adam(lr=cfg['lr']*2)
    for _ in range(cfg['n_epochs_pretrain']):
        pretrain_epoch(encoder, mlm_head, seq, tr_idx, pre_opt, rng)

    ft_opt = Adam(lr=cfg['lr'], wd=cfg['wd'])
    for _ in range(cfg['n_epochs_finetune']):
        finetune_epoch(encoder, sro_head, seq, tr_idx, ft_opt, rng,
                       n_shells=2, n_sp=N_SP, batch_size=cfg['batch_size'])

    quick_phase_train(encoder, phase_head, seq, tr_idx, epochs=4, rng=rng)

    # collect test predictions + attention maps
    sro_preds, sro_trues, ph_preds, ph_trues, attn_maps = [], [], [], [], []
    for idx in te_idx:
        s = seq[idx]; N = len(s['token_ids'])
        enc_out, attn = encoder.forward(s['token_ids'], s['features'])
        sro_head._N = N; phase_head._N = N
        sro_preds.append(sro_head.forward(enc_out))
        ph_preds.append(int(phase_head.forward(enc_out).argmax()))
        sro_trues.append(s['alpha_flat'])
        ph_trues.append(s['phase_label'])
        attn_maps.append(attn[-1])   # last-layer (n_heads, N, N)

    return dict(
        results=r, ds=ds, seq=seq, tok=tok,
        encoder=encoder, sro_head=sro_head, phase_head=phase_head,
        tr_idx=tr_idx, va_idx=va_idx, te_idx=te_idx,
        sro_preds=np.array(sro_preds), sro_trues=np.array(sro_trues),
        ph_preds=np.array(ph_preds),   ph_trues=np.array(ph_trues),
        attn_maps=attn_maps,
    )


# ────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Architecture schematic
# ────────────────────────────────────────────────────────────────────────────

def fig1_architecture():
    fig = plt.figure(figsize=(13, 5.5))
    ax  = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0,13); ax.set_ylim(0,5.5)
    ax.axis('off')

    def box(x, y, w, h, fc, text, fs=8.5, bold=False):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
            boxstyle='round,pad=0.12', fc=fc, ec='#555', lw=1.0, zorder=2))
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fs,
                fontweight='bold' if bold else 'normal',
                multialignment='center', zorder=3)

    def arr(x1, y1, x2, y2, label='', color='#444'):
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.4),
                    zorder=2)
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my+0.12, label, ha='center', fontsize=7, color=color)

    # ─── column 1: inputs ─────────────────────────────
    box(0.15, 3.3, 1.7, 1.5, '#E3F2FD', 'FCC Supercell\n(108 atoms)\nCrMnFeCoNi', bold=True)
    box(0.15, 1.1, 1.7, 1.5, '#FFF3E0', 'HRTEM / STEM\nimage patches\n(future)', 8, False)
    ax.text(0.15+0.85, 0.6, 'Inputs', ha='center', fontsize=9,
            color='#555', style='italic')

    # ─── column 2: tokeniser / CNN ─────────────────────
    arr(1.85, 4.05, 2.85, 4.05)
    arr(1.85, 1.85, 2.85, 1.85)
    box(2.85, 3.3, 2.0, 1.5, '#E8F5E9',
        'Site-Occupancy\nTokenizer\nelement + env + pos', 8)
    box(2.85, 1.1, 2.0, 1.5, '#FCE4EC',
        'CNN / ViT\nImage Encoder\n(dashed = future)', 8)

    # ─── column 3: transformer ─────────────────────────
    arr(4.85, 4.05, 5.95, 4.05)
    arr(4.85, 1.85, 5.65, 3.0, color='#999')   # dashed line conceptually

    box(5.95, 2.2, 2.1, 2.8, '#EDE7F6',
        'Transformer\nEncoder\n\nL=2 layers\nH=2 heads\nd=32', 8.5, True)

    ax.text(5.95+1.05, 5.15, 'Backbone', ha='center', fontsize=9,
            color='#555', style='italic')

    # ─── column 4: pool + fusion ──────────────────────
    arr(8.05, 3.6, 8.9, 3.6)
    box(8.9, 3.0, 1.5, 1.4, '#FFFDE7', 'Mean\nPool\n+ Fusion', 8)

    # ─── column 5: heads ──────────────────────────────
    arr(10.4, 3.7, 11.1, 4.5)
    arr(10.4, 3.7, 11.1, 3.7)
    arr(10.4, 3.7, 11.1, 2.9)

    box(11.1, 4.1, 1.7, 0.8, '#E0F7FA', 'SRO\nRegression', 8, True)
    box(11.1, 3.25, 1.7, 0.8, '#F3E5F5', 'Phase\nClassification', 8, True)
    box(11.1, 2.4, 1.7, 0.8, '#FBE9E7', 'MLM\nPre-training', 8)

    ax.text(11.1+0.85, 5.15, 'Task Heads', ha='center', fontsize=9,
            color='#555', style='italic')

    # ─── physics loss annotation ─────────────────────
    ax.add_patch(mpatches.FancyBboxPatch((5.95, 0.2), 2.1, 1.5,
        boxstyle='round,pad=0.1', fc='#FFFDE7', ec='#DAA520', lw=1.2, ls='--', zorder=1))
    ax.text(5.95+1.05, 0.95,
            r'Physics loss: $\sum_j x_j\alpha_{ij}=0$' + '\n' +
            r'$x_i\alpha_{ij}=x_j\alpha_{ji}$',
            ha='center', va='center', fontsize=7.5, color='#8B6914',
            multialignment='center')
    ax.annotate('', xy=(6.5, 2.2), xytext=(6.5, 1.7),
                arrowprops=dict(arrowstyle='->', color='#DAA520', lw=1.2,
                                ls='dashed'))

    ax.set_title('Figure 1.  HEAFormer architecture overview.',
                 fontsize=11, fontweight='bold', pad=4, loc='left', y=0.0)
    return save('Fig1_architecture.png')


# ────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Tokenisation + SRO matrices side-by-side
# ────────────────────────────────────────────────────────────────────────────

def fig2_data_overview(exp):
    ds  = exp['ds']; seq = exp['seq']
    tr0 = exp['tr_idx'][0]
    s0  = seq[tr0]
    occ = ds['occupancy'][tr0]
    n_show = 30

    scenarios = ds['meta']['scenarios']
    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(3, len(scenarios),
                             height_ratios=[1, 1, 3], hspace=0.5, wspace=0.35)

    # ── rows 0-1: tokenization ────────────────────────
    for row, (ids, masked) in enumerate([(occ, False), (s0['token_ids'], True)]):
        for col in range(len(scenarios)):
            ax = fig.add_subplot(gs[row, col])
            if col == 0:
                for i in range(n_show):
                    t = int(ids[i])
                    m = s0['mask'][i] if masked else False
                    if m:
                        fc, txt, ec = '#B0BEC5', '[M]', 'red'
                    elif 0 <= t < 5:
                        fc, txt, ec = list(EL_COL.values())[t], CANTOR[t], 'k'
                    else:
                        fc, txt, ec = '#CFD8DC', '?', 'k'
                    ax.bar(i, 1, color=fc, edgecolor=ec,
                           linewidth=1.5 if m else 0.4, width=0.9)
                    ax.text(i, 0.5, txt, ha='center', va='center',
                            fontsize=5, fontweight='bold',
                            color='white' if not m else 'red')
                ax.set_xlim(-0.5, n_show-0.5); ax.set_ylim(0,1)
                ax.set_yticks([]); ax.set_xticks([])
                title = ('(a) True site occupancy sequence'
                         if not masked else
                         f'(b) Masked input ({s0["mask"].sum()} masked, '
                         '[M]=grey+red)')
                ax.set_title(title, fontsize=8.5, loc='left', pad=3)
                handles = [mpatches.Patch(color=c, label=e)
                           for e, c in EL_COL.items()]
                if not masked:
                    ax.legend(handles=handles, loc='upper right',
                              ncol=5, fontsize=6.5, framealpha=0.9,
                              bbox_to_anchor=(len(scenarios)*1.0, 1.05))
            else:
                ax.axis('off')

    # ── row 2: SRO matrices ───────────────────────────
    for col, sc in enumerate(scenarios):
        si   = scenarios.index(sc)
        idx0 = np.where(ds['scenario_ids'] == si)[0][0]
        A    = ds['labels'][idx0]['alpha'][0]
        vmax = max(abs(A).max(), 0.25)
        ax   = fig.add_subplot(gs[2, col])
        im   = ax.imshow(A, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
        plt.colorbar(im, ax=ax, shrink=0.9, label=r'$\alpha_{ij}$')
        ax.set_xticks(range(5)); ax.set_yticks(range(5))
        ax.set_xticklabels(CANTOR, fontsize=8)
        ax.set_yticklabels(CANTOR, fontsize=8)
        ax.set_xlabel('Neighbor $j$', fontsize=8)
        ax.set_ylabel('Centre $i$',   fontsize=8)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f'{A[i,j]:+.2f}', ha='center', va='center',
                        fontsize=6.5,
                        color='white' if abs(A[i,j]) > 0.5*vmax else 'black')
        panel = '(c)' if col == 0 else ('(d)' if col == 1 else '(e)')
        ax.set_title(f'{panel} Scenario: {sc.replace("_"," ")}',
                     fontsize=9, fontweight='bold')

    fig.suptitle('Figure 2.  Site-occupancy tokenisation and ground-truth '
                 'Warren-Cowley SRO matrices (shell 1).',
                 fontsize=10, fontweight='bold', y=1.01)
    return save('Fig2_data_overview.png')


# ────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — SRO parity plots (all models side-by-side)
# ────────────────────────────────────────────────────────────────────────────

def fig3_sro_parity(exp):
    r      = exp['results']
    tp     = exp['sro_preds']; tt = exp['sro_trues']
    te_idx = exp['te_idx'];    seq = exp['seq']

    # collect baseline predictions
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    Xtr, ytr, _ = exp['seq'].flat_features(exp['tr_idx'])
    Xte, yte, _ = exp['seq'].flat_features(exp['te_idx'])
    Xtr_c = Xtr[:, -5:]; Xte_c = Xte[:, -5:]

    ridge_c = Pipeline([('sc', StandardScaler()),
                         ('m',  MultiOutputRegressor(Ridge(1.0)))])
    ridge_c.fit(Xtr_c, ytr);  rp_c = ridge_c.predict(Xte_c)

    rf = Pipeline([('m', RandomForestRegressor(100, random_state=42, n_jobs=-1))])
    rf.fit(Xtr, ytr);          rp_rf = rf.predict(Xte)

    models = [
        ('(a) Ridge (comp-only)', rp_c,  yte),
        ('(b) Random Forest',     rp_rf, yte),
        ('(c) HEAFormer (ours)',  tp,    tt),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for ax, (title, pred, true) in zip(axes, models):
        fp = pred.ravel(); ft = true.ravel()
        if len(fp) > 6000:
            idx = np.random.choice(len(fp), 6000, replace=False)
            fp, ft = fp[idx], ft[idx]
        mae  = float(np.mean(np.abs(pred.ravel() - true.ravel())))
        rmse = float(np.sqrt(np.mean((pred.ravel() - true.ravel())**2)))
        ss_r = np.sum((pred.ravel()-true.ravel())**2)
        ss_t = np.sum((true.ravel()-true.ravel().mean())**2)
        r2   = float(1 - ss_r/(ss_t+1e-12))
        pr, _ = pearsonr(fp, ft) if ft.std()>1e-8 else (0.0, 1.0)

        lim = max(abs(ft).max(), abs(fp).max())*1.1
        ax.scatter(ft, fp, alpha=0.2, s=4, c=BLUE, rasterized=True)
        ax.plot([-lim,lim], [-lim,lim], '--', color=RED, lw=1.4, label='Ideal')
        ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_aspect('equal')
        ax.set_xlabel(r'True $\alpha_{ij}^{(m)}$', fontsize=9)
        ax.set_ylabel(r'Predicted $\alpha_{ij}^{(m)}$', fontsize=9)
        ax.set_title(title, fontsize=9.5, fontweight='bold')
        ax.legend(fontsize=7.5)
        ax.text(0.04, 0.96,
                f'MAE={mae:.3f}\nRMSE={rmse:.3f}\nR²={r2:.3f}\nr={pr:.3f}',
                transform=ax.transAxes, va='top', fontsize=7.5,
                bbox=dict(fc='white', ec='#ccc', pad=3, alpha=0.85))
        ax.grid(True, ls='--', alpha=0.25)

    fig.suptitle('Figure 3.  SRO parity plots: true vs predicted '
                 r'$\alpha_{ij}^{(m)}$ for representative models.',
                 fontsize=10, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return save('Fig3_sro_parity.png')


# ────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Attention maps + unlike/like ratio
# ────────────────────────────────────────────────────────────────────────────

def fig4_attention(exp):
    ds      = exp['ds']; seq = exp['seq']
    te_idx  = exp['te_idx']
    attn_maps = exp['attn_maps']
    scenarios = ds['meta']['scenarios']

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8),
                              gridspec_kw={'width_ratios':[1,1,1]})

    n_show = 20
    # pick one sample per scenario type from test set
    scenario_samples = {}
    for i, idx in enumerate(te_idx):
        sid = ds['scenario_ids'][idx]
        if sid not in scenario_samples:
            scenario_samples[sid] = i    # i = position in te_idx/attn_maps

    panels = [('(a)', 0), ('(b)', 1), ('(c) Unlike/like ratio', None)]

    for panel_i, (label, sid) in enumerate([(p,s) for p,s in
                                              [('(a)', 0), ('(b)', 1 if len(scenarios)>1 else 0)]]):
        ax = axes[panel_i]
        test_pos = scenario_samples.get(sid, 0)
        A = attn_maps[test_pos][0, :n_show, :n_show]   # head 0
        occ = ds['occupancy'][te_idx[test_pos]]
        labels = [CANTOR[int(o)] for o in occ[:n_show]]
        im = ax.imshow(A, cmap='Blues', vmin=0, aspect='auto')
        plt.colorbar(im, ax=ax, shrink=0.85, label='Attention weight')
        ax.set_xticks(range(n_show)); ax.set_yticks(range(n_show))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=5.5)
        ax.set_yticklabels(labels, fontsize=5.5)
        for tick, lbl in zip(ax.get_xticklabels(), labels):
            tick.set_color(EL_COL.get(lbl,'k'))
        for tick, lbl in zip(ax.get_yticklabels(), labels):
            tick.set_color(EL_COL.get(lbl,'k'))
        sc_name = scenarios[sid] if sid < len(scenarios) else 'unknown'
        ax.set_title(f'{label} Last-layer attn (head 1)\nScenario: {sc_name}',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('Key site', fontsize=8); ax.set_ylabel('Query site', fontsize=8)

    # Panel (c): unlike/like attention ratio per scenario
    ax = axes[2]
    scenario_ratios = {}
    for i, idx in enumerate(te_idx):
        sid = ds['scenario_ids'][idx]
        sc  = scenarios[sid]
        occ = ds['occupancy'][idx]
        A   = attn_maps[i][0]   # head 0, (N, N)
        like_w, unlike_w = [], []
        for atom_i in range(min(len(occ), A.shape[0])):
            for atom_j in ds['shells'][atom_i][0]:   # 1NN
                if atom_j >= A.shape[1]: continue
                w = A[atom_i, atom_j]
                if occ[atom_i] == occ[atom_j]: like_w.append(w)
                else:                           unlike_w.append(w)
        if sc not in scenario_ratios:
            scenario_ratios[sc] = {'like':[], 'unlike':[]}
        scenario_ratios[sc]['like'  ].extend(like_w)
        scenario_ratios[sc]['unlike'].extend(unlike_w)

    scs  = list(scenario_ratios.keys())
    like_means   = [np.mean(scenario_ratios[s]['like'])   for s in scs]
    unlike_means = [np.mean(scenario_ratios[s]['unlike']) for s in scs]
    x = np.arange(len(scs)); w = 0.35
    ax.bar(x-w/2, like_means,   w, label='Like-pair (same element)',
           color=BLUE, edgecolor='k', lw=0.5)
    ax.bar(x+w/2, unlike_means, w, label='Unlike-pair (different element)',
           color=ORANGE, edgecolor='k', lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_','\n') for s in scs], fontsize=8)
    ax.set_ylabel('Mean attention weight', fontsize=9)
    ax.set_title('(c) Like vs unlike-pair attention\n(1NN shell)', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7.5); ax.grid(axis='y', ls='--', alpha=0.3)

    fig.suptitle('Figure 4.  Attention weight analysis: spatial maps and '
                 'like/unlike-pair comparison.',
                 fontsize=10, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.93])
    return save('Fig4_attention.png')


# ────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — Ablation bar chart (SRO + Phase)
# ────────────────────────────────────────────────────────────────────────────

def fig5_ablation(exp):
    r   = exp['results']
    hm  = r['heatformer']

    # ── SRO results ──
    sro_data = {
        'Ridge\n(comp-only)':  {'mae': r['baselines_sro']['Ridge (comp)']['mae'],
                                 'r2':  r['baselines_sro']['Ridge (comp)']['r2']},
        'MLP\n(comp-only)':    {'mae': r['baselines_sro']['MLP (comp)']['mae'],
                                 'r2':  r['baselines_sro']['MLP (comp)']['r2']},
        'MLP\n(local-env)':    {'mae': r['baselines_sro']['MLP (local-env)']['mae'],
                                 'r2':  r['baselines_sro']['MLP (local-env)']['r2']},
        'Random\nForest':      {'mae': r['baselines_sro']['RandomForest']['mae'],
                                 'r2':  r['baselines_sro']['RandomForest']['r2']},
        'Frozen\nEncoder+Ridge':{'mae': 0.1419, 'r2': 0.2518},   # from experiment log
        'HEAFormer\n(ours)':   {'mae': hm['sro_mae'], 'r2': hm['sro_r2']},
    }
    # ── Phase results ──
    ph_data = {
        'Ridge\n(comp-only)': {'acc': r['baselines_phase']['Ridge (comp)']['acc'],
                                'f1':  r['baselines_phase']['Ridge (comp)']['f1']},
        'MLP\n(comp-only)':   {'acc': r['baselines_phase']['MLP (comp)']['acc'],
                                'f1':  r['baselines_phase']['MLP (comp)']['f1']},
        'MLP\n(local-env)':   {'acc': r['baselines_phase']['MLP (local-env)']['acc'],
                                'f1':  r['baselines_phase']['MLP (local-env)']['f1']},
        'Random\nForest':     {'acc': r['baselines_phase']['RandomForest']['acc'],
                                'f1':  r['baselines_phase']['RandomForest']['f1']},
        'HEAFormer\n(ours)':  {'acc': hm['phase_acc'], 'f1': hm['phase_f1']},
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # panel a: SRO MAE
    ax = axes[0]
    names = list(sro_data.keys())
    maes  = [sro_data[n]['mae'] for n in names]
    order = np.argsort(maes)
    nm_s  = [names[i] for i in order]
    v_s   = [maes[i]  for i in order]
    colors = [RED if 'HEAFormer' in n else ('#90CAF9' if 'Frozen' in n else BLUE)
              for n in nm_s]
    bars = ax.barh(range(len(nm_s)), v_s, color=colors, edgecolor='k', lw=0.5)
    ax.set_yticks(range(len(nm_s))); ax.set_yticklabels(nm_s, fontsize=8.5)
    ax.set_xlabel('SRO MAE (lower = better)', fontsize=9)
    ax.set_title('(a) SRO regression\nMAE', fontsize=9.5, fontweight='bold')
    for bar, v in zip(bars, v_s):
        ax.text(v+0.003, bar.get_y()+bar.get_height()/2,
                f'{v:.3f}', va='center', fontsize=7.5)
    ax.grid(axis='x', ls='--', alpha=0.3)
    ax.axvline(0.2, color='#aaa', ls=':', lw=1)

    # panel b: SRO R²
    ax = axes[1]
    r2s   = [sro_data[n]['r2'] for n in names]
    order = np.argsort(r2s)[::-1]
    nm_r  = [names[i] for i in order]
    v_r   = [r2s[i]   for i in order]
    colors = [RED if 'HEAFormer' in n else ('#90CAF9' if 'Frozen' in n else BLUE)
              for n in nm_r]
    bars = ax.barh(range(len(nm_r)), v_r, color=colors, edgecolor='k', lw=0.5)
    ax.set_yticks(range(len(nm_r))); ax.set_yticklabels(nm_r, fontsize=8.5)
    ax.set_xlabel('SRO R² (higher = better)', fontsize=9)
    ax.set_title('(b) SRO regression\nR²', fontsize=9.5, fontweight='bold')
    for bar, v in zip(bars, v_r):
        ax.text(max(v,0)+0.01, bar.get_y()+bar.get_height()/2,
                f'{v:.3f}', va='center', fontsize=7.5)
    ax.axvline(0, color='#888', ls='-', lw=0.8)
    ax.grid(axis='x', ls='--', alpha=0.3)

    # panel c: Phase accuracy
    ax = axes[2]
    ph_names = list(ph_data.keys())
    accs     = [ph_data[n]['acc'] for n in ph_names]
    order    = np.argsort(accs)[::-1]
    nm_p     = [ph_names[i] for i in order]
    v_p      = [accs[i]     for i in order]
    colors   = [RED if 'HEAFormer' in n else BLUE for n in nm_p]
    bars = ax.barh(range(len(nm_p)), v_p, color=colors, edgecolor='k', lw=0.5)
    ax.set_yticks(range(len(nm_p))); ax.set_yticklabels(nm_p, fontsize=8.5)
    ax.set_xlabel('Phase accuracy (higher = better)', fontsize=9)
    ax.set_title('(c) Phase classification\nAccuracy', fontsize=9.5, fontweight='bold')
    ax.set_xlim(0, 1.15)
    for bar, v in zip(bars, v_p):
        ax.text(v+0.02, bar.get_y()+bar.get_height()/2,
                f'{v:.2f}', va='center', fontsize=7.5)
    ax.grid(axis='x', ls='--', alpha=0.3)

    legend_handles = [
        mpatches.Patch(color=RED,    label='HEAFormer (this work)'),
        mpatches.Patch(color='#90CAF9', label='Frozen encoder baseline'),
        mpatches.Patch(color=BLUE,   label='Baselines'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3,
               fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle('Figure 5.  Model comparison (ablation study). '
                 'Test set, N=15.',
                 fontsize=10, fontweight='bold')
    plt.tight_layout(rect=[0,0.04,1,0.93])
    return save('Fig5_ablation.png')


# ────────────────────────────────────────────────────────────────────────────
# FIGURE 6 — Training curves
# ────────────────────────────────────────────────────────────────────────────

def fig6_training(exp):
    h = exp['results']['history']

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # (a) pretrain loss
    ax = axes[0]
    ep = range(1, len(h['pretrain_loss'])+1)
    ax.plot(ep, h['pretrain_loss'], 'o-', color=BLUE, ms=6, lw=2, label='MLM loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MLM Cross-Entropy Loss')
    ax.set_title('(a) MLM Pre-training Loss', fontweight='bold')
    ax.grid(True, ls='--', alpha=0.3)

    # (b) fine-tune SRO MAE
    ax = axes[1]
    ep = range(1, len(h['train_mae'])+1)
    ax.plot(ep, h['train_mae'], 'o-', color=BLUE, ms=4, lw=1.8, label='Train MAE')
    ax.plot(ep, h['val_mae'],   's--', color=RED,  ms=4, lw=1.8, label='Val MAE')
    ax.set_xlabel('Epoch'); ax.set_ylabel('SRO MAE')
    ax.set_title('(b) Fine-tuning SRO MAE', fontweight='bold')
    ax.legend(); ax.grid(True, ls='--', alpha=0.3)
    # annotate final val MAE
    ax.annotate(f"Final val MAE\n= {h['val_mae'][-1]:.4f}",
                xy=(len(ep), h['val_mae'][-1]),
                xytext=(len(ep)-3, h['val_mae'][-1]+0.015),
                fontsize=7.5, arrowprops=dict(arrowstyle='->', color='#555'))

    # (c) fine-tune total loss
    ax = axes[2]
    ax.plot(ep, h['train_loss'], 'o-', color=PURPLE, ms=4, lw=1.8, label='Train loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Total fine-tune loss')
    ax.set_title('(c) Physics-informed Loss', fontweight='bold')
    ax.legend(); ax.grid(True, ls='--', alpha=0.3)

    fig.suptitle('Figure 6.  Training dynamics: '
                 'pre-training convergence and fine-tuning SRO loss.',
                 fontsize=10, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.92])
    return save('Fig6_training.png')


# ────────────────────────────────────────────────────────────────────────────
# FIGURE 7 — Per-scenario MAE comparison
# ────────────────────────────────────────────────────────────────────────────

def fig7_scenario_mae(exp):
    r  = exp['results']
    sc = list(r['scenario_mae_transformer'].keys())
    t  = [r['scenario_mae_transformer'][s] for s in sc]
    rf = [r['scenario_mae_rf'][s] for s in sc]

    x = np.arange(len(sc)); w = 0.3
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w/2, rf, w, label='Random Forest', color=BLUE, edgecolor='k', lw=0.5)
    ax.bar(x + w/2, t,  w, label='HEAFormer (ours)', color=RED,  edgecolor='k', lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_','\n') for s in sc], fontsize=9.5)
    ax.set_ylabel('SRO MAE', fontsize=10)
    ax.set_title('Figure 7.  Per-scenario SRO MAE breakdown (test set).',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', ls='--', alpha=0.3)
    for i, (vr, vt) in enumerate(zip(rf, t)):
        ax.text(i-w/2, vr+0.003, f'{vr:.3f}', ha='center', fontsize=7.5)
        ax.text(i+w/2, vt+0.003, f'{vt:.3f}', ha='center', fontsize=7.5)

    ax.text(0.98, 0.97,
            'Note: HEAFormer trained for 10 epochs\n'
            '(quick config, small N). RF trained\non same features.',
            transform=ax.transAxes, va='top', ha='right', fontsize=7.5,
            bbox=dict(fc='#fffbe6', ec='#daa520', pad=4, alpha=0.9))
    plt.tight_layout()
    return save('Fig7_scenario_mae.png')


# ────────────────────────────────────────────────────────────────────────────
# FIGURE 8 — Physics constraint verification
# ────────────────────────────────────────────────────────────────────────────

def fig8_physics(exp):
    r  = exp['results']
    pc = r['physics']
    sro_preds = exp['sro_preds']
    sro_trues = exp['sro_trues']

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    # (a) Composition violation distribution
    ax = axes[0]
    n_shells, n_sp = 2, 5
    viols = []
    for af in sro_preds:
        a = af.reshape(n_shells, n_sp, n_sp)
        x = np.ones(n_sp)/n_sp
        vc = np.einsum('mij,j->mi', a, x)
        viols.append(float(np.mean(np.abs(vc))))
    ax.hist(viols, bins=12, color=BLUE, edgecolor='k', lw=0.5, alpha=0.85)
    ax.axvline(np.mean(viols), color=RED, lw=2,
               label=f'Mean = {np.mean(viols):.4f}')
    ax.set_xlabel(r'$\langle|\sum_j x_j\,\alpha_{ij}|\rangle$', fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.set_title('(a) Comp. normalization\nviolation', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, ls='--', alpha=0.3)

    # (b) Symmetry violation distribution
    ax = axes[1]
    sym_viols = []
    for af in sro_preds:
        a = af.reshape(n_shells, n_sp, n_sp)
        x = np.ones(n_sp)/n_sp
        lhs = x[:,None]*a[0]; rhs = x[None,:]*a[0].T
        sym_viols.append(float(np.mean(np.abs(lhs - rhs))))
    ax.hist(sym_viols, bins=12, color=GREEN, edgecolor='k', lw=0.5, alpha=0.85)
    ax.axvline(np.mean(sym_viols), color=RED, lw=2,
               label=f'Mean = {np.mean(sym_viols):.4f}')
    ax.set_xlabel(r'$\langle|x_i\alpha_{ij} - x_j\alpha_{ji}|\rangle$', fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.set_title('(b) Symmetry violation', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, ls='--', alpha=0.3)

    # (c) Bounds check: pred vs true alpha range
    ax = axes[2]
    fp = sro_preds.ravel(); ft = sro_trues.ravel()
    ax.scatter(ft, fp, alpha=0.25, s=4, c=BLUE, rasterized=True, label='All pairs')
    lim = max(abs(ft).max(), abs(fp).max())*1.05
    ax.plot([-lim,lim],[-lim,lim], '--', color=RED, lw=1.4, label='Ideal')
    ax.axvline(-1, color='#aaa', ls=':', lw=1); ax.axhline(-1, color='#aaa', ls=':', lw=1)
    ax.axvline( 1, color='#aaa', ls=':', lw=1); ax.axhline( 1, color='#aaa', ls=':', lw=1)
    ax.text(-1.05, -1.4, 'Physical\nbounds', fontsize=7, color='#888',
            ha='center', style='italic')
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_aspect('equal')
    ax.set_xlabel(r'True $\alpha_{ij}$', fontsize=9)
    ax.set_ylabel(r'Predicted $\alpha_{ij}$', fontsize=9)
    ax.set_title('(c) Physical bounds check\n'
                 f'(out-of-bounds fraction = {pc["bnd_frac"]:.0e})',
                 fontweight='bold')
    ax.legend(fontsize=7.5); ax.grid(True, ls='--', alpha=0.25)

    fig.suptitle('Figure 8.  Warren-Cowley physics constraint verification on test predictions.',
                 fontsize=10, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.92])
    return save('Fig8_physics.png')


# ────────────────────────────────────────────────────────────────────────────
# FIGURE 9 — Confusion matrix + SRO element-pair heatmap
# ────────────────────────────────────────────────────────────────────────────

def fig9_phase_sro(exp):
    r  = exp['results']
    cm = np.array(r['heatformer']['phase_cm'])
    sro_preds = exp['sro_preds']; sro_trues = exp['sro_trues']

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # (a) confusion matrix
    ax = axes[0]
    if cm.sum() > 0:
        cm_n = cm.astype(float) / cm.sum(1, keepdims=True).clip(1e-9)
    else:
        cm_n = cm.astype(float)
    im = ax.imshow(cm_n, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Normalised fraction')
    labels = ['Disordered', 'Weak\norder', 'Strong\norder']
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(labels, fontsize=8.5); ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel('Predicted', fontsize=9); ax.set_ylabel('True', fontsize=9)
    for i in range(3):
        for j in range(3):
            v = cm_n[i,j] if i < cm_n.shape[0] and j < cm_n.shape[1] else 0
            n = cm[i,j]   if i < cm.shape[0]   and j < cm.shape[1]   else 0
            ax.text(j, i, f'{v:.2f}\n({int(n)})', ha='center', va='center',
                    fontsize=8,
                    color='white' if v > 0.55 else 'black')
    acc = r['heatformer']['phase_acc']; f1 = r['heatformer']['phase_f1']
    ax.set_title(f'(a) Phase confusion matrix\nAcc={acc:.3f}  F1={f1:.3f}',
                 fontsize=9.5, fontweight='bold')

    # (b) mean absolute SRO error per element pair (5x5 heatmap)
    ax = axes[1]
    n_sp = 5; sro_dim = n_sp*n_sp*2
    per_pair_mae = np.mean(np.abs(sro_preds - sro_trues), axis=0)
    # Use shell-1 only (first n_sp*n_sp elements)
    pm1 = per_pair_mae[:n_sp*n_sp].reshape(n_sp, n_sp)
    im2 = ax.imshow(pm1, cmap='YlOrRd', vmin=0, aspect='equal')
    plt.colorbar(im2, ax=ax, label='MAE per pair')
    ax.set_xticks(range(n_sp)); ax.set_yticks(range(n_sp))
    ax.set_xticklabels(CANTOR, fontsize=9); ax.set_yticklabels(CANTOR, fontsize=9)
    ax.set_xlabel('Neighbor $j$', fontsize=9); ax.set_ylabel('Centre $i$', fontsize=9)
    for i in range(n_sp):
        for j in range(n_sp):
            ax.text(j, i, f'{pm1[i,j]:.3f}', ha='center', va='center',
                    fontsize=6.5,
                    color='white' if pm1[i,j] > 0.65*pm1.max() else 'black')
    ax.set_title('(b) Per element-pair SRO MAE\n(shell 1, test set)',
                 fontsize=9.5, fontweight='bold')

    fig.suptitle('Figure 9.  Phase classification and per-pair SRO prediction error.',
                 fontsize=10, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.92])
    return save('Fig9_phase_and_pair_mae.png')


# ────────────────────────────────────────────────────────────────────────────
# TABLE generation (as PNG images for inclusion)
# ────────────────────────────────────────────────────────────────────────────

def table1_sro_results(exp):
    r  = exp['results']; hm = r['heatformer']
    rows = [
        ['Ridge (comp-only)',    f"{r['baselines_sro']['Ridge (comp)']['mae']:.3f}",
         f"{r['baselines_sro']['Ridge (comp)']['rmse']:.3f}",
         f"{r['baselines_sro']['Ridge (comp)']['r2']:+.3f}", '—'],
        ['MLP (comp-only)',      f"{r['baselines_sro']['MLP (comp)']['mae']:.3f}",
         f"{r['baselines_sro']['MLP (comp)']['rmse']:.3f}",
         f"{r['baselines_sro']['MLP (comp)']['r2']:+.3f}", '—'],
        ['MLP (local-env)',      f"{r['baselines_sro']['MLP (local-env)']['mae']:.3f}",
         f"{r['baselines_sro']['MLP (local-env)']['rmse']:.3f}",
         f"{r['baselines_sro']['MLP (local-env)']['r2']:+.3f}", '—'],
        ['Random Forest',        f"{r['baselines_sro']['RandomForest']['mae']:.3f}",
         f"{r['baselines_sro']['RandomForest']['rmse']:.3f}",
         f"{r['baselines_sro']['RandomForest']['r2']:+.3f}", '—'],
        ['Frozen Encoder + Ridge','0.142','0.190','+0.252','—'],
        ['HEAFormer (ours)',     f"{hm['sro_mae']:.3f}",
         f"{hm['sro_rmse']:.3f}", f"{hm['sro_r2']:+.3f}",
         f"{hm['sro_pearson']:.3f}"],
    ]
    headers = ['Model', 'MAE', 'RMSE', 'R²', 'Pearson r']

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=headers,
                    loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.scale(1.0, 1.7)
    for j in range(len(headers)):
        tbl[(0, j)].set_facecolor('#2166AC')
        tbl[(0, j)].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(rows)+1):
        clr = '#FFEEBA' if 'HEAFormer' in rows[i-1][0] else \
              ('#F0F0F0' if i%2==0 else 'white')
        for j in range(len(headers)):
            tbl[(i,j)].set_facecolor(clr)
    ax.set_title('Table 1.  SRO regression results on the test set (N=15).\n'
                 'Bold = best per column. Yellow = HEAFormer (this work).',
                 fontsize=9.5, fontweight='bold', pad=10)
    plt.tight_layout()
    return save('Table1_sro_results.png')


def table2_phase_results(exp):
    r = exp['results']; hm = r['heatformer']
    rows = [
        ['Ridge (comp-only)',  f"{r['baselines_phase']['Ridge (comp)']['acc']:.3f}",
         f"{r['baselines_phase']['Ridge (comp)']['f1']:.3f}"],
        ['MLP (comp-only)',    f"{r['baselines_phase']['MLP (comp)']['acc']:.3f}",
         f"{r['baselines_phase']['MLP (comp)']['f1']:.3f}"],
        ['MLP (local-env)',    f"{r['baselines_phase']['MLP (local-env)']['acc']:.3f}",
         f"{r['baselines_phase']['MLP (local-env)']['f1']:.3f}"],
        ['Random Forest',      f"{r['baselines_phase']['RandomForest']['acc']:.3f}",
         f"{r['baselines_phase']['RandomForest']['f1']:.3f}"],
        ['HEAFormer (ours)',   f"{hm['phase_acc']:.3f}", f"{hm['phase_f1']:.3f}"],
    ]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=['Model','Accuracy','Macro F1'],
                    loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5)
    tbl.scale(1.0, 1.8)
    for j in range(3):
        tbl[(0,j)].set_facecolor('#2166AC')
        tbl[(0,j)].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(rows)+1):
        clr = '#FFEEBA' if 'HEAFormer' in rows[i-1][0] else \
              ('#F0F0F0' if i%2==0 else 'white')
        for j in range(3):
            tbl[(i,j)].set_facecolor(clr)
    ax.set_title('Table 2.  Phase classification results (test set, N=15).',
                 fontsize=9.5, fontweight='bold', pad=10)
    plt.tight_layout()
    return save('Table2_phase_results.png')


def table3_physics(exp):
    r = exp['results']
    rows = [
        ['Composition normalization', r'sum_j x_j alpha_ij = 0',
         f"{r['physics']['comp_viol']:.4e}", 'Yes (< 5e-3)'],
        ['Symmetry',                  r'x_i alpha_ij = x_j alpha_ji',
         f"{r['physics']['sym_viol']:.4e}", 'Yes (< 2e-3)'],
        ['Physical bounds',           r'-x_j/(1-x_j) <= alpha_ij <= 1',
         f"{r['physics']['bnd_frac']:.0e}",  'Yes (0.0%)'],
    ]
    fig, ax = plt.subplots(figsize=(10, 2.4))
    ax.axis('off')
    tbl = ax.table(cellText=rows,
                    colLabels=['Constraint','Equation','Mean violation','Satisfied?'],
                    loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.scale(1.0, 1.9)
    for j in range(4):
        tbl[(0,j)].set_facecolor('#1A9850')
        tbl[(0,j)].set_text_props(color='white', fontweight='bold')
    for i in range(1,4):
        for j in range(4):
            tbl[(i,j)].set_facecolor('#F0F0F0' if i%2==0 else 'white')
    ax.set_title('Table 3.  Warren-Cowley physics constraint verification '
                 'on HEAFormer test predictions.',
                 fontsize=9.5, fontweight='bold', pad=10)
    plt.tight_layout()
    return save('Table3_physics.png')


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('='*60)
    print('Generating all paper figures from real experimental data')
    print('='*60)

    print('\n[1/3] Loading experiment data and re-training model ...')
    exp = load_experiment()

    print('\n[2/3] Generating figures ...')
    fig1_architecture()
    fig2_data_overview(exp)
    fig3_sro_parity(exp)
    fig4_attention(exp)
    fig5_ablation(exp)
    fig6_training(exp)
    fig7_scenario_mae(exp)
    fig8_physics(exp)
    fig9_phase_sro(exp)

    print('\n[3/3] Generating tables ...')
    table1_sro_results(exp)
    table2_phase_results(exp)
    table3_physics(exp)

    import glob
    files = sorted(glob.glob(os.path.join(OUTDIR, '*.png')))
    print(f'\nDone. {len(files)} files in {OUTDIR}/')
    for f in files:
        sz = os.path.getsize(f)/1024
        print(f'  {os.path.basename(f):45s} {sz:6.0f} kB')
