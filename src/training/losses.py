"""
Loss functions and Adam optimizer for HEA Transformer.

SRO physics constraints embedded in the loss:
  C1  sum_j  x_j * alpha_ij^m = 0          (composition normalization)
  C2  x_i * alpha_ij = x_j * alpha_ji       (symmetry)
  C3  -x_j/(1-x_j) <= alpha_ij <= 1        (physical bounds, soft)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Cross-entropy
# ---------------------------------------------------------------------------

def cross_entropy(logits: np.ndarray,
                  targets: np.ndarray) -> tuple:
    """
    Numerically stable softmax cross-entropy.

    Returns (loss_scalar, d_logits)
    """
    M, C = logits.shape
    shifted = logits - logits.max(1, keepdims=True)
    exp_l   = np.exp(shifted)
    probs   = exp_l / exp_l.sum(1, keepdims=True)
    loss    = -np.log(probs[np.arange(M), targets] + 1e-12).mean()
    dp      = probs.copy()
    dp[np.arange(M), targets] -= 1.0
    dp     /= M
    return loss, dp


# ---------------------------------------------------------------------------
# Warren-Cowley SRO physics-informed loss
# ---------------------------------------------------------------------------

def sro_loss(alpha_pred: np.ndarray,
             alpha_true: np.ndarray,
             x: np.ndarray,
             n_shells: int,
             n_sp: int,
             lam_mse: float = 1.0,
             lam_comp: float = 0.3,
             lam_sym: float = 0.3,
             lam_bounds: float = 0.05) -> tuple:
    """
    Combined SRO loss with physics constraint penalties.

    Returns
    -------
    total        : scalar
    terms        : dict of individual component losses
    d_alpha_pred : (sro_dim,) gradient w.r.t. alpha_pred
    """
    ns = n_sp
    ap = alpha_pred.reshape(n_shells, ns, ns)
    at = alpha_true.reshape(n_shells, ns, ns)

    # --- MSE ---
    d_mse  = ap - at
    l_mse  = 0.5 * np.sum(d_mse**2)

    # --- C1: composition normalization  sum_j x_j * alpha_ij = 0 ---
    viol_c = np.einsum('mij,j->mi', ap, x)           # (ns, ns) . (ns,) -> (ms, ns)
    l_comp = 0.5 * np.sum(viol_c**2)
    d_comp = np.einsum('mi,j->mij', viol_c, x)

    # --- C2: symmetry  x_i*alpha_ij - x_j*alpha_ji = 0 ---
    sym_v  = x[:,None]*ap - x[None,:]*ap.transpose(0,2,1)
    l_sym  = 0.5 * np.sum(sym_v**2)
    d_sym  = (x[:,None]*sym_v - x[None,:]*sym_v.transpose(0,2,1))

    # --- C3: soft bounds ---
    lb = np.full((ns, ns), -0.99)
    for j in range(ns):
        for i in range(ns):
            if i != j and x[j] < 1.0:
                lb[i,j] = -x[j] / max(1-x[j], 1e-6)
    lb3 = lb[None].repeat(n_shells, 0)
    ub3 = np.ones_like(lb3)
    below  = np.maximum(lb3 - ap, 0)
    above  = np.maximum(ap - ub3, 0)
    l_bnd  = 0.5 * np.sum(below**2 + above**2)
    d_bnd  = -below + above

    total  = lam_mse*l_mse + lam_comp*l_comp + lam_sym*l_sym + lam_bounds*l_bnd
    d_tot  = (lam_mse*d_mse + lam_comp*d_comp +
              lam_sym*d_sym + lam_bounds*d_bnd)

    terms  = dict(mse=float(l_mse), comp=float(l_comp),
                  sym=float(l_sym),  bounds=float(l_bnd),
                  total=float(total))
    return total, terms, d_tot.flatten()


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------

class Adam:
    """
    Adam optimizer (Kingma & Ba 2015).
    Operates in-place on a list of numpy parameter arrays.
    """

    def __init__(self, lr=1e-3, b1=0.9, b2=0.999,
                 eps=1e-8, wd=0.0, grad_clip=5.0):
        self.lr   = lr
        self.b1   = b1
        self.b2   = b2
        self.eps  = eps
        self.wd   = wd
        self.clip = grad_clip
        self.t    = 0
        self.m    = {}
        self.v    = {}

    def step(self, params: list, grads: list):
        self.t += 1
        lrt = self.lr * np.sqrt(1 - self.b2**self.t) / (1 - self.b1**self.t)
        for i, (p, g) in enumerate(zip(params, grads)):
            if g is None:
                continue
            if p.shape != g.shape:
                continue    # shape mismatch guard — should not happen
            gc = np.clip(g, -self.clip, self.clip)
            if self.wd:
                gc = gc + self.wd * p
            if i not in self.m:
                self.m[i] = np.zeros_like(p)
                self.v[i] = np.zeros_like(p)
            self.m[i] = self.b1 * self.m[i] + (1-self.b1) * gc
            self.v[i] = self.b2 * self.v[i] + (1-self.b2) * gc**2
            p -= lrt * self.m[i] / (np.sqrt(self.v[i]) + self.eps)


# ---------------------------------------------------------------------------
# Gradient check
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    np.random.seed(0)
    ns, nm = 5, 2
    x  = np.ones(ns)/ns
    ap = np.random.randn(nm*ns*ns) * 0.1
    at = np.random.randn(nm*ns*ns) * 0.1

    l, terms, d = sro_loss(ap, at, x, nm, ns)
    print(f"SRO loss={l:.4f}  terms={terms}")

    eps = 1e-5
    dn = np.zeros_like(ap)
    for k in range(len(ap)):
        p, m = ap.copy(), ap.copy()
        p[k] += eps; m[k] -= eps
        lp, _, _ = sro_loss(p, at, x, nm, ns)
        lm, _, _ = sro_loss(m, at, x, nm, ns)
        dn[k] = (lp - lm) / (2*eps)
    err = np.max(np.abs(d - dn))
    print(f"Gradient check max error: {err:.2e}  (should be < 1e-4)")
    assert err < 1e-4
    print("[PASS] losses.py")
