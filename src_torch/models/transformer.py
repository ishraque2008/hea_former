"""
PyTorch HEA Transformer — production backend.

Identical architecture to src/models/transformer.py but uses
torch.nn for GPU acceleration and automatic differentiation.

Usage
-----
    from src_torch.models.transformer import HEAEncoderTorch, SROHeadTorch

Requirements
------------
    pip install torch>=2.0

Key differences from NumPy version
------------------------------------
- Batch dimension supported: (B, N, d_model)
- Gradient flow via autograd (no manual backward required)
- Mixed-precision training via torch.cuda.amp
- Flash attention via F.scaled_dot_product_attention (PyTorch >=2.0)
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False
    raise ImportError(
        "PyTorch not installed.  Install with: pip install torch>=2.0\n"
        "For CPU-only use, the NumPy version in src/models/transformer.py "
        "is fully functional."
    )

import numpy as np


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

class HEAEmbeddingTorch(nn.Module):
    """
    Factorised embedding: element table + feature projection, concatenated.
    out = [E[token_ids]  ||  feat_proj(features)]  ->  (B, N, d_model)
    """
    def __init__(self, vocab: int, feat_dim: int,
                 d_emb: int, d_model: int):
        super().__init__()
        assert d_model >= d_emb
        self.elem_emb  = nn.Embedding(vocab, d_emb)
        self.feat_proj = nn.Linear(feat_dim, d_model - d_emb)
        nn.init.xavier_uniform_(self.feat_proj.weight)

    def forward(self, token_ids, features):
        # token_ids: (B, N) int   features: (B, N, feat_dim)
        emb  = self.elem_emb(token_ids)     # (B, N, d_emb)
        proj = self.feat_proj(features)      # (B, N, d_model-d_emb)
        return torch.cat([emb, proj], dim=-1)  # (B, N, d_model)


# ---------------------------------------------------------------------------
# Encoder layer (standard pre-norm transformer)
# ---------------------------------------------------------------------------

class HEAEncoderLayerTorch(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                           batch_first=True,
                                           dropout=dropout)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        # Pre-norm MHA + residual
        x_norm = self.ln1(x)
        a_out, a_weights = self.attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=True, average_attn_weights=False,
        )
        x = x + self.drop(a_out)
        # Pre-norm FF + residual
        x = x + self.drop(self.ff(self.ln2(x)))
        return x, a_weights   # (B,N,d) and (B,H,N,N)


# ---------------------------------------------------------------------------
# Full encoder
# ---------------------------------------------------------------------------

class HEAEncoderTorch(nn.Module):
    """
    HEA site-occupancy transformer encoder.

    Parameters
    ----------
    vocab    : vocabulary size
    feat_dim : continuous feature dimension
    d_model  : transformer hidden size
    n_heads  : attention heads
    n_layers : encoder layers
    d_ff     : feed-forward dimension
    d_emb    : element embedding dimension
    dropout  : dropout probability
    """

    def __init__(self, vocab: int, feat_dim: int,
                 d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 3, d_ff: int = 128,
                 d_emb: int = 16, dropout: float = 0.1):
        super().__init__()
        self.embedding = HEAEmbeddingTorch(vocab, feat_dim, d_emb, d_model)
        self.layers    = nn.ModuleList([
            HEAEncoderLayerTorch(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.d_model   = d_model

    def forward(self, token_ids, features, padding_mask=None):
        """
        Parameters
        ----------
        token_ids    : (B, N) int tensor
        features     : (B, N, feat_dim) float tensor
        padding_mask : (B, N) bool tensor, True = PAD positions

        Returns
        -------
        x         : (B, N, d_model) contextual representations
        attn_all  : list of (B, n_heads, N, N) attention weights per layer
        """
        x = self.embedding(token_ids, features)
        attn_all = []
        for layer in self.layers:
            x, A = layer(x, key_padding_mask=padding_mask)
            attn_all.append(A)
        return x, attn_all

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Prediction heads
# ---------------------------------------------------------------------------

class MLMHeadTorch(nn.Module):
    def __init__(self, d_model: int, vocab: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, enc_out, mask):
        # enc_out: (B, N, d_model)  mask: (B, N) bool
        # Returns: (B*N_masked, vocab)  -- flattened masked positions
        return self.proj(enc_out[mask])


class SROHeadTorch(nn.Module):
    """Mean-pool -> MLP -> SRO vector."""
    def __init__(self, d_model: int, sro_dim: int, d_hidden: int = None):
        super().__init__()
        if d_hidden is None:
            d_hidden = d_model * 2
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, sro_dim),
        )

    def forward(self, enc_out, padding_mask=None):
        # enc_out: (B, N, d_model)
        if padding_mask is not None:
            # Masked mean pooling
            valid = (~padding_mask).float().unsqueeze(-1)  # (B,N,1)
            pooled = (enc_out * valid).sum(1) / valid.sum(1).clamp(min=1)
        else:
            pooled = enc_out.mean(dim=1)  # (B, d_model)
        return self.mlp(pooled)            # (B, sro_dim)


class PhaseHeadTorch(nn.Module):
    def __init__(self, d_model: int, n_classes: int = 3):
        super().__init__()
        self.proj = nn.Linear(d_model, n_classes)

    def forward(self, enc_out, padding_mask=None):
        if padding_mask is not None:
            valid = (~padding_mask).float().unsqueeze(-1)
            pooled = (enc_out * valid).sum(1) / valid.sum(1).clamp(min=1)
        else:
            pooled = enc_out.mean(dim=1)
        return self.proj(pooled)


# ---------------------------------------------------------------------------
# Physics-informed SRO loss (torch)
# ---------------------------------------------------------------------------

class SROLossTorch(nn.Module):
    """
    Physics-informed Warren-Cowley SRO regression loss.

    Enforces:
      C1 (composition normalization): sum_j x_j * alpha_ij = 0
      C2 (symmetry):                  x_i * alpha_ij = x_j * alpha_ji
      C3 (bounds):                    alpha_ij in [-x_j/(1-x_j), 1]
    """
    def __init__(self, n_shells: int = 2, n_sp: int = 5,
                 lam_mse: float = 1.0, lam_comp: float = 0.3,
                 lam_sym: float = 0.3, lam_bnd: float = 0.05):
        super().__init__()
        self.n_shells = n_shells
        self.n_sp     = n_sp
        self.lam_mse  = lam_mse
        self.lam_comp = lam_comp
        self.lam_sym  = lam_sym
        self.lam_bnd  = lam_bnd

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                x: torch.Tensor) -> dict:
        """
        pred, target : (B, sro_dim)
        x            : (B, n_sp) or (n_sp,) composition
        """
        B       = pred.shape[0]
        ns, ms  = self.n_sp, self.n_shells
        ap = pred.view(B, ms, ns, ns)
        at = target.view(B, ms, ns, ns)

        if x.dim() == 1:
            x = x.unsqueeze(0).expand(B, -1)   # (B, ns)

        # MSE
        l_mse = 0.5 * F.mse_loss(ap, at, reduction='sum') / B

        # C1: composition normalization
        viol_c = torch.einsum('bmij,bj->bmi', ap, x)   # (B, ms, ns)
        l_comp = 0.5 * (viol_c**2).sum() / B

        # C2: symmetry
        x_i  = x.view(B, 1, ns, 1)    # center species
        x_j  = x.view(B, 1, 1, ns)    # neighbor species
        sym_v = x_i * ap - x_j * ap.transpose(-2, -1)
        l_sym = 0.5 * (sym_v**2).sum() / B

        # C3: soft bounds
        ub    = torch.ones_like(ap)
        lb    = torch.full_like(ap, -0.99)
        for j in range(ns):
            for i in range(ns):
                if i != j:
                    lb[:, :, i, j] = -x[:, j] / (1 - x[:, j] + 1e-6)
        below = F.relu(lb - ap)
        above = F.relu(ap - ub)
        l_bnd = 0.5 * (below**2 + above**2).sum() / B

        total = (self.lam_mse  * l_mse  +
                 self.lam_comp * l_comp +
                 self.lam_sym  * l_sym  +
                 self.lam_bnd  * l_bnd)

        return dict(total=total, mse=l_mse, comp=l_comp,
                    sym=l_sym, bounds=l_bnd)


# ---------------------------------------------------------------------------
# Complete model (encoder + heads)
# ---------------------------------------------------------------------------

class HEAFormerTorch(nn.Module):
    """
    Full HEAFormer model: encoder + SRO head + phase head + MLM head.

    This is the production model for GPU-accelerated training.
    Use src/models/transformer.py for CPU/NumPy training.
    """

    def __init__(self, vocab: int, feat_dim: int,
                 sro_dim: int, n_classes: int = 3,
                 d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 3, d_ff: int = 128,
                 d_emb: int = 16, dropout: float = 0.1):
        super().__init__()
        self.encoder    = HEAEncoderTorch(
            vocab, feat_dim, d_model, n_heads,
            n_layers, d_ff, d_emb, dropout
        )
        self.mlm_head   = MLMHeadTorch(d_model, vocab)
        self.sro_head   = SROHeadTorch(d_model, sro_dim)
        self.phase_head = PhaseHeadTorch(d_model, n_classes)

    def forward(self, token_ids, features,
                mlm_mask=None, padding_mask=None):
        enc_out, attn_all = self.encoder(
            token_ids, features, padding_mask
        )
        sro_pred   = self.sro_head(enc_out, padding_mask)
        phase_pred = self.phase_head(enc_out, padding_mask)
        mlm_logits = None
        if mlm_mask is not None and mlm_mask.any():
            mlm_logits = self.mlm_head(enc_out, mlm_mask)
        return dict(sro=sro_pred, phase=phase_pred,
                    mlm=mlm_logits, attn=attn_all)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    B, N = 4, 108
    vocab, feat_dim, sro_dim = 7, 20, 50
    model = HEAFormerTorch(vocab, feat_dim, sro_dim,
                            d_model=64, n_heads=4, n_layers=3,
                            d_ff=128, d_emb=16)
    print(f"HEAFormerTorch params: {model.n_params():,}")

    tok  = torch.randint(0, 5, (B, N))
    feat = torch.randn(B, N, feat_dim)
    mask = torch.zeros(B, N, dtype=torch.bool)
    mask[:, ::7] = True   # ~14% masked

    out = model(tok, feat, mlm_mask=mask)
    print(f"SRO pred:   {out['sro'].shape}")
    print(f"Phase pred: {out['phase'].shape}")
    if out['mlm'] is not None:
        print(f"MLM logits: {out['mlm'].shape}")

    x   = torch.ones(B, 5) / 5
    sro_criterion = SROLossTorch()
    sro_target    = torch.randn(B, sro_dim) * 0.1
    losses = sro_criterion(out['sro'], sro_target, x)
    losses['total'].backward()
    print(f"Loss: {losses['total'].item():.4f}")
    print("[PASS] src_torch/models/transformer.py")
