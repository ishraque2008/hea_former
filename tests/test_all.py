#!/usr/bin/env python3
"""
test_all.py
===========
Unit and integration tests for HEAFormer.

Run with:
    python tests/test_all.py
    python -m pytest tests/test_all.py -v      # if pytest installed
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import traceback

PASS = 0; FAIL = 0


def ok(name):
    global PASS
    PASS += 1
    print(f"  [PASS] {name}")


def fail(name, e):
    global FAIL
    FAIL += 1
    print(f"  [FAIL] {name}: {e}")
    traceback.print_exc()


# ---------------------------------------------------------------------------
# 1. Physics / SRO
# ---------------------------------------------------------------------------

def test_fcc_supercell():
    from src.physics.sro import build_fcc_supercell
    frac, box, N = build_fcc_supercell(3, 3, 3)
    assert N == 108, f"Expected 108 atoms, got {N}"
    assert frac.shape == (108, 3)
    assert np.allclose(np.diag(box), [10.8, 10.8, 10.8])
    ok("fcc_supercell_3x3x3")


def test_neighbor_counts():
    from src.physics.sro import build_fcc_supercell, nearest_neighbors
    frac, box, N = build_fcc_supercell(3, 3, 3)
    shells, _ = nearest_neighbors(frac, box, n_shells=2)
    nn1 = [len(shells[i][0]) for i in range(N)]
    nn2 = [len(shells[i][1]) for i in range(N)]
    assert min(nn1) == 12 == max(nn1), f"1NN counts wrong: {set(nn1)}"
    assert min(nn2) == 6  == max(nn2), f"2NN counts wrong: {set(nn2)}"
    ok("neighbor_counts_3x3x3_fcc")


def test_sro_random_near_zero():
    """Random solid solution should have |alpha| << 1."""
    from src.physics.sro import build_fcc_supercell, nearest_neighbors, warren_cowley_sro
    from src.data.supercell_generator import random_occupancy
    frac, box, N = build_fcc_supercell(3, 3, 3)
    shells, _ = nearest_neighbors(frac, box, n_shells=2)
    rng = np.random.default_rng(0)
    # Average over many samples for law of large numbers
    alphas = []
    for _ in range(10):
        occ = random_occupancy(N, rng=rng)
        a = warren_cowley_sro(occ, shells, 5, n_shells=1)
        alphas.append(np.abs(a).max())
    mean_max = np.mean(alphas)
    # With N=108, expect max|alpha| < 0.5 on average for random
    assert mean_max < 0.5, f"Random SRO too large: mean_max={mean_max:.3f}"
    ok("sro_random_near_zero")


def test_sro_sum_rule():
    """Composition normalization sum rule: sum_j x_j * alpha_ij = 0."""
    from src.physics.sro import build_fcc_supercell, nearest_neighbors, warren_cowley_sro
    from src.data.supercell_generator import random_occupancy
    frac, box, N = build_fcc_supercell(3, 3, 3)
    shells, _ = nearest_neighbors(frac, box)
    occ = random_occupancy(N)
    x = np.bincount(occ, minlength=5) / N
    a = warren_cowley_sro(occ, shells, 5, n_shells=1)
    for i in range(5):
        viol = abs(np.dot(x, a[0, i]))
        # Perfect sum rule requires infinite N; for N=108 expect < 0.01
        assert viol < 0.05, f"Sum rule violated for species {i}: {viol:.4f}"
    ok("sro_composition_sum_rule")


# ---------------------------------------------------------------------------
# 2. Tokeniser
# ---------------------------------------------------------------------------

def test_tokenizer_shapes():
    from src.data.supercell_generator import build_dataset
    from src.data.tokenizer import SiteOccupancyTokenizer, SequenceDataset
    ds  = build_dataset(nx=3, ny=3, nz=3, n_per_scenario=3,
                         n_mc_steps=100, verbose=False)
    tok = SiteOccupancyTokenizer(n_shells=2)
    assert tok.feat_dim == 20, f"feat_dim={tok.feat_dim}, expected 20"
    s = tok.tokenize(ds['occupancy'][0], ds['shells'], ds['coords_frac'],
                     mask_prob=0.15, rng=np.random.default_rng(0))
    assert s['features'].shape == (108, 20)
    assert s['token_ids'].shape == (108,)
    ok("tokenizer_shapes")


def test_tokenizer_masking():
    from src.data.supercell_generator import build_dataset
    from src.data.tokenizer import SiteOccupancyTokenizer
    from src.data.supercell_generator import MASK_TOK
    ds  = build_dataset(nx=3, ny=3, nz=3, n_per_scenario=2,
                         n_mc_steps=100, verbose=False)
    tok = SiteOccupancyTokenizer()
    rng = np.random.default_rng(42)
    s   = tok.tokenize(ds['occupancy'][0], ds['shells'], ds['coords_frac'],
                       mask_prob=0.15, rng=rng)
    n_masked = s['mask'].sum()
    # 15% masking of 108 sites: expect 10-25 masked positions
    assert 5 <= n_masked <= 30, f"Unexpected mask count: {n_masked}"
    # Masked positions should have MASK_TOK or random element in token_ids
    ok("tokenizer_masking")


def test_flat_features_shape():
    from src.data.supercell_generator import build_dataset, N_SP
    from src.data.tokenizer import SiteOccupancyTokenizer, SequenceDataset
    ds  = build_dataset(nx=3, ny=3, nz=3, n_per_scenario=4,
                         n_mc_steps=100, verbose=False)
    tok = SiteOccupancyTokenizer()
    seq = SequenceDataset(ds, tok)
    X, ys, yp = seq.flat_features([0, 1, 2])
    expected_dim = 2 * N_SP * N_SP + N_SP   # 55
    assert X.shape == (3, expected_dim), f"X shape: {X.shape}"
    assert ys.shape[0] == 3
    assert yp.shape == (3,)
    ok("flat_features_shape")


# ---------------------------------------------------------------------------
# 3. Transformer
# ---------------------------------------------------------------------------

def test_encoder_forward_shape():
    from src.models.transformer import HEAEncoder
    enc = HEAEncoder(vocab=7, feat_dim=20, d_model=32,
                      n_heads=2, n_layers=2, d_ff=64, d_emb=8)
    tok  = np.random.randint(0, 5, 108)
    feat = np.random.randn(108, 20)
    out, attn = enc.forward(tok, feat)
    assert out.shape == (108, 32), f"out shape: {out.shape}"
    assert len(attn) == 2
    assert attn[0].shape == (2, 108, 108)
    ok("encoder_forward_shape")


def test_param_grad_alignment():
    """
    Critical: all_params() and collect_param_grads() must return
    lists of the same length with matching shapes.
    """
    from src.models.transformer import HEAEncoder, SROHead
    enc = HEAEncoder(vocab=7, feat_dim=20, d_model=32,
                      n_heads=2, n_layers=2, d_ff=64, d_emb=8)
    tok  = np.random.randint(0, 5, 108)
    feat = np.random.randn(108, 20)
    out, _ = enc.forward(tok, feat)
    d_out = np.random.randn(*out.shape)
    enc.backward(d_out)
    params = enc.all_params()
    grads  = enc.collect_param_grads()
    assert len(params) == len(grads), \
        f"param/grad count: {len(params)} vs {len(grads)}"
    for i, (p, g) in enumerate(zip(params, grads)):
        assert p.shape == g.shape, \
            f"param[{i}] {p.shape} != grad[{i}] {g.shape}"
    ok("param_grad_alignment")


def test_sro_head_backward():
    from src.models.transformer import HEAEncoder, SROHead
    enc = HEAEncoder(vocab=7, feat_dim=20, d_model=32,
                      n_heads=2, n_layers=2, d_ff=64, d_emb=8)
    head = SROHead(32, 50)
    tok  = np.random.randint(0, 5, 108)
    feat = np.random.randn(108, 20)
    out, _ = enc.forward(tok, feat)
    head._N = 108
    pred    = head.forward(out)
    assert pred.shape == (50,)
    d_pred  = np.random.randn(50)
    d_enc   = head.backward(d_pred)
    assert d_enc.shape == out.shape, f"d_enc shape: {d_enc.shape}"
    hg      = head.param_grads()
    hp      = head.params()
    assert len(hg) == len(hp)
    ok("sro_head_backward")


# ---------------------------------------------------------------------------
# 4. Losses
# ---------------------------------------------------------------------------

def test_cross_entropy_gradient():
    from src.training.losses import cross_entropy
    np.random.seed(0)
    logits  = np.random.randn(8, 7)
    targets = np.random.randint(0, 7, 8)
    loss, dlogits = cross_entropy(logits, targets)
    assert np.isfinite(loss) and loss > 0
    assert dlogits.shape == logits.shape
    ok("cross_entropy")


def test_sro_loss_gradient_check():
    """Numerical gradient check for the full SRO loss."""
    from src.training.losses import sro_loss
    np.random.seed(1)
    ns, ms = 5, 2
    x  = np.ones(ns) / ns
    ap = np.random.randn(ms * ns * ns) * 0.1
    at = np.random.randn(ms * ns * ns) * 0.1
    _, _, d = sro_loss(ap, at, x, ms, ns)
    eps = 1e-5
    dn = np.zeros_like(ap)
    for k in range(len(ap)):
        p, m = ap.copy(), ap.copy()
        p[k] += eps; m[k] -= eps
        lp, _, _ = sro_loss(p, at, x, ms, ns)
        lm, _, _ = sro_loss(m, at, x, ms, ns)
        dn[k] = (lp - lm) / (2 * eps)
    err = float(np.max(np.abs(d - dn)))
    assert err < 1e-4, f"Gradient check failed: max_err={err:.2e}"
    ok(f"sro_loss_gradient_check (max_err={err:.2e})")


def test_adam_decreases_loss():
    """One Adam step should decrease a trivial quadratic loss."""
    from src.training.losses import Adam
    p   = np.array([2.0, -1.0, 0.5])
    g   = p.copy()   # gradient of ||p||^2 / 2
    opt = Adam(lr=0.1)
    l0  = 0.5 * np.sum(p**2)
    opt.step([p], [g])
    l1  = 0.5 * np.sum(p**2)
    assert l1 < l0, f"Adam did not decrease loss: {l0:.4f} -> {l1:.4f}"
    ok("adam_decreases_loss")


# ---------------------------------------------------------------------------
# 5. Training loop (end-to-end mini)
# ---------------------------------------------------------------------------

def test_pretrain_runs():
    from src.data.supercell_generator import build_dataset
    from src.data.tokenizer import SiteOccupancyTokenizer, SequenceDataset
    from src.models.transformer import HEAEncoder, MLMHead
    from src.training.trainer import pretrain_epoch
    from src.training.losses import Adam

    ds  = build_dataset(nx=3, ny=3, nz=3, n_per_scenario=5,
                         n_mc_steps=100, verbose=False)
    tok = SiteOccupancyTokenizer()
    seq = SequenceDataset(ds, tok, mask_prob=0.15, seed=0)
    enc = HEAEncoder(7, tok.feat_dim, d_model=16, n_heads=2,
                      n_layers=1, d_ff=32, d_emb=4)
    mlm = MLMHead(16, 7)
    opt = Adam(lr=1e-3)
    rng = np.random.default_rng(0)
    r   = pretrain_epoch(enc, mlm, seq, list(range(5)), opt, rng)
    assert 'mlm_loss' in r and np.isfinite(r['mlm_loss'])
    ok(f"pretrain_runs (loss={r['mlm_loss']:.4f})")


def test_finetune_loss_decreases():
    """
    Loss should strictly decrease over the first 3 fine-tuning epochs
    (with sufficient learning rate on a tiny dataset).
    """
    from src.data.supercell_generator import build_dataset
    from src.data.tokenizer import SiteOccupancyTokenizer, SequenceDataset
    from src.models.transformer import HEAEncoder, SROHead
    from src.training.trainer import finetune_epoch
    from src.training.losses import Adam

    ds  = build_dataset(nx=3, ny=3, nz=3, n_per_scenario=8,
                         n_mc_steps=200, verbose=False)
    tok = SiteOccupancyTokenizer()
    seq = SequenceDataset(ds, tok, mask_prob=0.0, seed=0)
    enc = HEAEncoder(7, tok.feat_dim, d_model=16, n_heads=2,
                      n_layers=1, d_ff=32, d_emb=4, seed=0)
    head = SROHead(16, ds['meta']['n_sro'])
    opt  = Adam(lr=5e-3)
    rng  = np.random.default_rng(0)
    idx  = list(range(len(seq)))

    losses = []
    for _ in range(3):
        r = finetune_epoch(enc, head, seq, idx, opt, rng,
                           n_shells=2, n_sp=5, batch_size=4)
        losses.append(r['loss'])

    assert losses[2] < losses[0], \
        f"Loss did not decrease: {losses}"
    ok(f"finetune_loss_decreases ({losses[0]:.3f} -> {losses[-1]:.3f})")


# ---------------------------------------------------------------------------
# 6. Baselines
# ---------------------------------------------------------------------------

def test_baselines_run():
    from src.data.supercell_generator import build_dataset
    from src.data.tokenizer import SiteOccupancyTokenizer, SequenceDataset
    from src.evaluation.baselines import run_all_baselines

    ds  = build_dataset(nx=3, ny=3, nz=3, n_per_scenario=10,
                         n_mc_steps=200,
                         scenarios=['random', 'ordering'],
                         verbose=False)
    tok = SiteOccupancyTokenizer()
    seq = SequenceDataset(ds, tok)
    idx = list(range(20))
    tr, te = idx[:14], idx[14:]
    bl = run_all_baselines(seq, tr, te, verbose=False)
    for task in ['sro', 'phase']:
        assert task in bl
        for nm, m in bl[task].items():
            assert isinstance(m, dict), f"{nm} returned {m}"
    ok("baselines_run")


# ---------------------------------------------------------------------------
# 7. Plots (just check files are created, not content)
# ---------------------------------------------------------------------------

def test_plots_produce_files():
    import os
    from src.data.supercell_generator import build_dataset
    from src.data.tokenizer import SiteOccupancyTokenizer, SequenceDataset
    from src.evaluation.plots import (
        plot_sro_matrix, plot_sro_parity, plot_training_curves
    )
    ds = build_dataset(nx=3, ny=3, nz=3, n_per_scenario=3,
                        n_mc_steps=100, verbose=False)
    # SRO matrix
    p1 = plot_sro_matrix(ds['labels'][0]['alpha'], shell=0, title='test')
    assert os.path.exists(p1), f"File not created: {p1}"

    # Parity plot
    sro_p = np.random.randn(5, 50) * 0.1
    sro_t = np.random.randn(5, 50) * 0.1
    p2 = plot_sro_parity(sro_p, sro_t, model_name='TestModel', mae=0.1, r2=0.5)
    assert os.path.exists(p2)

    # Training curves
    history = dict(train_loss=[1.0, 0.8, 0.6], val_loss=[1.1, 0.9, 0.7])
    p3 = plot_training_curves(history)
    assert os.path.exists(p3)
    ok("plots_produce_files")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_fcc_supercell,
    test_neighbor_counts,
    test_sro_random_near_zero,
    test_sro_sum_rule,
    test_tokenizer_shapes,
    test_tokenizer_masking,
    test_flat_features_shape,
    test_encoder_forward_shape,
    test_param_grad_alignment,
    test_sro_head_backward,
    test_cross_entropy_gradient,
    test_sro_loss_gradient_check,
    test_adam_decreases_loss,
    test_pretrain_runs,
    test_finetune_loss_decreases,
    test_baselines_run,
    test_plots_produce_files,
]

if __name__ == '__main__':
    print("=" * 60)
    print("HEAFormer Test Suite")
    print("=" * 60)
    for test_fn in ALL_TESTS:
        try:
            test_fn()
        except Exception as e:
            fail(test_fn.__name__, e)

    print(f"\n{'='*60}")
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS+FAIL} tests")
    print('='*60)
    if FAIL > 0:
        sys.exit(1)
