"""Evaluation metrics for HEA Transformer."""

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, confusion_matrix


def compute_metrics(sro_pred, sro_true, phase_pred, phase_true,
                    split='test') -> dict:
    diff  = sro_pred - sro_true
    mae   = float(np.mean(np.abs(diff)))
    rmse  = float(np.sqrt(np.mean(diff**2)))
    ss_r  = float(np.sum(diff**2))
    ss_t  = float(np.sum((sro_true - sro_true.mean())**2))
    r2    = float(1 - ss_r / (ss_t + 1e-12))
    pr    = float(pearsonr(sro_pred.ravel(), sro_true.ravel())[0]) \
            if sro_true.std() > 1e-8 else 0.0

    phase_pred = np.array(phase_pred); phase_true = np.array(phase_true)
    acc = float((phase_pred == phase_true).mean())
    f1  = float(f1_score(phase_true, phase_pred,
                          average='macro', zero_division=0))
    cm  = confusion_matrix(phase_true, phase_pred,
                            labels=[0,1,2]).tolist()

    print(f"\n[{split.upper()}] N={len(sro_pred)}")
    print(f"  SRO  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  r={pr:.4f}")
    print(f"  Phase Acc={acc:.4f}  F1={f1:.4f}")
    return dict(sro_mae=mae, sro_rmse=rmse, sro_r2=r2, sro_pearson=pr,
                phase_acc=acc, phase_f1=f1, phase_cm=cm)


def physics_check(sro_pred, x_global, n_shells, n_sp) -> dict:
    comp_v, sym_v, bnd_v = [], [], []
    for af in sro_pred:
        a = af.reshape(n_shells, n_sp, n_sp)
        vc = np.einsum('mij,j->mi', a, x_global)
        comp_v.append(float(np.mean(np.abs(vc))))
        lhs = x_global[:,None] * a[0]
        rhs = x_global[None,:] * a[0].T
        sym_v.append(float(np.mean(np.abs(lhs - rhs))))
        bnd_v.append(float(np.mean((a < -1) | (a > 1))))
    return dict(comp_viol=float(np.mean(comp_v)),
                sym_viol =float(np.mean(sym_v)),
                bnd_frac =float(np.mean(bnd_v)))
