"""
Training loops for HEA Transformer.

Stage 1 -- MLM pre-training:
    Masked site prediction; trains the encoder to understand
    local chemical context without property labels.

Stage 2 -- Fine-tuning (physics-informed SRO regression):
    Joint SRO regression + phase classification with the
    Warren-Cowley physics constraints enforced via loss penalties.
    Gradients flow through the full encoder (not frozen).
"""

import numpy as np
from src.training.losses import cross_entropy, sro_loss, Adam
from src.models.transformer import gelu_grad


# ---------------------------------------------------------------------------
# MLM pre-training
# ---------------------------------------------------------------------------

def pretrain_epoch(encoder, mlm_head, seq_dataset,
                   train_idx, optimizer, rng):
    """
    One epoch of masked language model pre-training.
    Returns dict with mean mlm_loss and mlm_accuracy.
    """
    mlm_losses, mlm_accs = [], []

    for idx in rng.permutation(train_idx):
        s        = seq_dataset[idx]
        tok      = s['token_ids']
        feat     = s['features']
        mask     = s['mask']
        true_ids = s['true_ids']

        if not mask.any():
            continue

        # forward
        enc_out, _ = encoder.forward(tok, feat)
        logits     = mlm_head.proj.forward(enc_out[mask])   # (M, vocab)

        # loss
        loss, d_logits = cross_entropy(logits, true_ids[mask])
        mlm_losses.append(float(loss))
        mlm_accs.append(float((logits.argmax(1) == true_ids[mask]).mean()))

        # backward: mlm_head
        d_masked, d_mlm_grads = mlm_head.proj.grads_list(d_logits)

        # scatter back to full (N, d_model)
        d_enc = np.zeros_like(enc_out)
        d_enc[mask] = d_masked

        # backward: encoder (only enc params get updated in pre-training)
        encoder.backward(d_enc)
        enc_grads = encoder.collect_param_grads()

        # combined params + grads
        all_params = encoder.all_params() + mlm_head.params()
        all_grads  = enc_grads + d_mlm_grads
        optimizer.step(all_params, all_grads)

    return dict(mlm_loss=float(np.mean(mlm_losses)) if mlm_losses else 0.0,
                mlm_acc =float(np.mean(mlm_accs))   if mlm_accs  else 0.0)


# ---------------------------------------------------------------------------
# SRO fine-tuning
# ---------------------------------------------------------------------------

def finetune_epoch(encoder, sro_head, seq_dataset,
                   train_idx, optimizer, rng,
                   n_shells=2, n_sp=5, batch_size=16,
                   sro_kwargs=None):
    """
    One epoch of physics-informed SRO regression fine-tuning.

    Uses gradient accumulation over `batch_size` samples before
    calling optimizer.step(), which is equivalent to mini-batch SGD.
    """
    if sro_kwargs is None:
        sro_kwargs = dict(lam_mse=1.0, lam_comp=0.3,
                         lam_sym=0.3, lam_bounds=0.05)

    all_params = encoder.all_params() + sro_head.params()
    grad_accum = [np.zeros_like(p) for p in all_params]
    n_acc = 0

    ep_loss, ep_mae = [], []

    for step, idx in enumerate(rng.permutation(train_idx)):
        s       = seq_dataset[idx]
        tok     = s['token_ids']
        feat    = s['features']
        alpha_t = s['alpha_flat']
        xg      = s['x_global']
        N       = len(tok)

        # forward
        enc_out, _ = encoder.forward(tok, feat)
        sro_head._N = N
        sro_pred    = sro_head.forward(enc_out)

        # physics loss
        loss, terms, d_sro = sro_loss(sro_pred, alpha_t, xg,
                                       n_shells, n_sp, **sro_kwargs)
        ep_loss.append(float(loss))
        ep_mae.append(float(np.mean(np.abs(sro_pred - alpha_t))))

        # backward: SRO head
        d_enc = sro_head.backward(d_sro)
        sro_g = sro_head.param_grads()

        # backward: encoder
        encoder.backward(d_enc)
        enc_g = encoder.collect_param_grads()

        # accumulate
        all_grads = enc_g + sro_g
        for i, g in enumerate(all_grads):
            if i < len(grad_accum) and g is not None:
                if grad_accum[i].shape == g.shape:
                    grad_accum[i] += g
        n_acc += 1

        # optimizer step when batch full or last sample
        if n_acc >= batch_size or step == len(train_idx) - 1:
            neff = max(n_acc, 1)
            scaled = [g / neff for g in grad_accum]
            optimizer.step(all_params, scaled)
            for g in grad_accum:
                g[:] = 0.0
            n_acc = 0

    return dict(loss=float(np.mean(ep_loss)) if ep_loss else 0.0,
                sro_mae=float(np.mean(ep_mae)) if ep_mae else 0.0)


# ---------------------------------------------------------------------------
# Evaluation (no gradient updates)
# ---------------------------------------------------------------------------

def evaluate(encoder, sro_head, phase_head, seq_dataset,
             indices, n_shells=2, n_sp=5) -> dict:
    """
    Evaluate SRO regression + phase classification without parameter updates.

    Returns dict with sro_mae, sro_r2, phase_acc, phase_f1, attn_maps.
    """
    from scipy.stats import pearsonr
    from sklearn.metrics import f1_score

    sro_preds, sro_trues = [], []
    ph_preds, ph_trues   = [], []
    attn_maps            = []

    for idx in indices:
        s   = seq_dataset[idx]
        tok = s['token_ids']
        ft  = s['features']
        N   = len(tok)

        enc_out, attn = encoder.forward(tok, ft)

        # SRO prediction
        sro_head._N = N
        sro_preds.append(sro_head.forward(enc_out))
        sro_trues.append(s['alpha_flat'])
        attn_maps.append(attn[-1])   # last-layer attention

        # Phase prediction (if head provided)
        if phase_head is not None:
            phase_head._N = N
            ph_preds.append(int(phase_head.forward(enc_out).argmax()))
            ph_trues.append(s['phase_label'])

    sro_preds = np.array(sro_preds)
    sro_trues = np.array(sro_trues)

    diff  = sro_preds - sro_trues
    mae   = float(np.mean(np.abs(diff)))
    rmse  = float(np.sqrt(np.mean(diff**2)))
    ss_r  = float(np.sum(diff**2))
    ss_t  = float(np.sum((sro_trues - sro_trues.mean())**2))
    r2    = 1 - ss_r / (ss_t + 1e-12)
    pr    = float(pearsonr(sro_preds.ravel(), sro_trues.ravel())[0]) \
            if sro_trues.std() > 1e-8 else 0.0

    result = dict(sro_mae=mae, sro_rmse=rmse, sro_r2=r2,
                  sro_pearson=pr, attn_maps=attn_maps)

    if ph_preds:
        pa = np.array(ph_preds); pt = np.array(ph_trues)
        acc = float((pa == pt).mean())
        f1  = float(f1_score(pt, pa, average='macro', zero_division=0))
        result.update(phase_acc=acc, phase_f1=f1,
                      phase_pred=pa, phase_true=pt)

    return result


# ---------------------------------------------------------------------------
# Quick phase head training (3 epochs)
# ---------------------------------------------------------------------------

def quick_phase_train(encoder, phase_head, seq_dataset,
                      train_idx, lr=5e-4, epochs=3, rng=None):
    """Rapid 3-epoch training of the phase classification head."""
    if rng is None:
        rng = np.random.default_rng(0)
    opt = Adam(lr=lr)
    for ep in range(epochs):
        for idx in rng.permutation(train_idx):
            s   = seq_dataset[idx]
            enc_out, _ = encoder.forward(s['token_ids'], s['features'])
            phase_head._N = len(s['token_ids'])
            logits = phase_head.forward(enc_out)
            loss, d_logits = cross_entropy(logits[None,:],
                                           np.array([s['phase_label']]))
            d_enc = phase_head.backward(d_logits.squeeze(0))
            opt.step(phase_head.params(), phase_head.param_grads())
