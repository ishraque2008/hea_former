"""
NumPy Transformer Encoder for HEA Site-Occupancy Modeling.

The entire forward + backward pass is implemented in NumPy so the
experiment can be run without PyTorch.  For GPU-accelerated production
use, see src_torch/models/transformer.py (identical architecture).

Layer order in all_params() / collect_param_grads():
  1. HEAEmbedding   :  E, feat_proj.W, feat_proj.b
  2. For each TransformerEncoderLayer (layer 0, 1, ...):
       Wq.W, Wq.b, Wk.W, Wk.b, Wv.W, Wv.b, Wo.W, Wo.b
       ff.lin1.W, ff.lin1.b, ff.lin2.W, ff.lin2.b
       ln1.gamma, ln1.beta, ln2.gamma, ln2.beta
"""

import numpy as np


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715*x**3)))

def gelu_grad(x):
    c  = np.sqrt(2/np.pi)
    t  = np.tanh(c * (x + 0.044715*x**3))
    dt = (1 - t**2) * c * (1 + 3*0.044715*x**2)
    return 0.5 * (1 + t) + 0.5 * x * dt

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------

class Linear:
    def __init__(self, in_d, out_d, bias=True):
        s = np.sqrt(2 / (in_d + out_d))
        self.W = np.random.randn(in_d, out_d) * s
        self.b = np.zeros(out_d) if bias else None
        self._x = None

    def forward(self, x):
        self._x = x
        out = x @ self.W
        if self.b is not None:
            out += self.b
        return out

    def backward(self, dout):
        """Returns (dx, dW, db-or-None)."""
        x  = self._x
        dx = dout @ self.W.T
        # reshape to 2-D for grad computation
        x2   = x.reshape(-1, x.shape[-1])
        do2  = dout.reshape(-1, dout.shape[-1])
        dW   = x2.T @ do2
        db   = do2.sum(axis=0) if self.b is not None else None
        return dx, dW, db

    def params(self):
        return [self.W, self.b] if self.b is not None else [self.W]

    def grads_list(self, dout):
        """Returns (dx, [dW, db])."""
        dx, dW, db = self.backward(dout)
        g = [dW, db] if db is not None else [dW]
        return dx, g


# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------

class LayerNorm:
    def __init__(self, d, eps=1e-6):
        self.gamma = np.ones(d)
        self.beta  = np.zeros(d)
        self.eps   = eps
        self._cache = None

    def forward(self, x):
        mu    = x.mean(-1, keepdims=True)
        var   = x.var(-1, keepdims=True)
        xhat  = (x - mu) / np.sqrt(var + self.eps)
        out   = self.gamma * xhat + self.beta
        self._cache = (x, xhat, var)
        return out

    def backward(self, dout):
        x, xhat, var = self._cache
        d    = x.shape[-1]
        si   = 1 / np.sqrt(var + self.eps)
        dgam = (dout * xhat).sum(tuple(range(dout.ndim-1)))
        dbet = dout.sum(tuple(range(dout.ndim-1)))
        dxh  = dout * self.gamma
        dvar = (dxh * (x - x.mean(-1,keepdims=True)) * (-0.5) * si**3
                ).sum(-1, keepdims=True)
        dmu  = (-dxh * si).sum(-1, keepdims=True) + \
               dvar * (-2*(x - x.mean(-1,keepdims=True))).mean(-1, keepdims=True)
        dx   = dxh * si + dvar * 2*(x - x.mean(-1,keepdims=True))/d + dmu/d
        return dx, dgam, dbet

    def params(self):
        return [self.gamma, self.beta]

    def grads_list(self, dout):
        dx, dg, db = self.backward(dout)
        return dx, [dg, db]


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention
# ---------------------------------------------------------------------------

class MHA:
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.nh      = n_heads
        self.dk      = d_model // n_heads
        self.Wq = Linear(d_model, d_model)
        self.Wk = Linear(d_model, d_model)
        self.Wv = Linear(d_model, d_model)
        self.Wo = Linear(d_model, d_model)
        self._c = None

    def forward(self, x, mask=None):
        # x: (N, d_model)
        N, d = x.shape
        h, dk = self.nh, self.dk
        Q = self.Wq.forward(x).reshape(N,h,dk).transpose(1,0,2)  # (h,N,dk)
        K = self.Wk.forward(x).reshape(N,h,dk).transpose(1,0,2)
        V = self.Wv.forward(x).reshape(N,h,dk).transpose(1,0,2)
        s = 1 / np.sqrt(dk)
        sc = Q @ K.transpose(0,2,1) * s                          # (h,N,N)
        if mask is not None:
            sc = sc + mask[None,:,:] * (-1e9)
        A  = softmax(sc, axis=-1)                                  # (h,N,N)
        O  = (A @ V).transpose(1,0,2).reshape(N,d)                # (N,d)
        out = self.Wo.forward(O)
        self._c = (x, Q, K, V, sc, A, mask, N, h, dk, s, O)
        return out, A

    def backward(self, dout):
        x, Q, K, V, sc, A, mask, N, h, dk, s, O = self._c
        # Wo backward
        dO, self._gWo  = self.Wo.grads_list(dout)                # (N,d)
        dO = dO.reshape(N,h,dk).transpose(1,0,2)                 # (h,N,dk)
        # attn output: O = A @ V
        dV = A.transpose(0,2,1) @ dO                             # (h,N,dk)
        dA = dO @ V.transpose(0,2,1)                             # (h,N,N)
        # softmax backward
        ds = A * dA - A * (A * dA).sum(-1, keepdims=True)
        ds = ds * s
        if mask is not None:
            ds[mask[None,:,:].repeat(h,0)] = 0
        dQ = ds @ K                                               # (h,N,dk)
        dK = ds.transpose(0,2,1) @ Q
        def r2(t): return t.transpose(1,0,2).reshape(N, h*dk)
        dQ2 = r2(dQ); dK2 = r2(dK); dV2 = r2(dV)
        dx_q, self._gWq = self.Wq.grads_list(dQ2)
        dx_k, self._gWk = self.Wk.grads_list(dK2)
        dx_v, self._gWv = self.Wv.grads_list(dV2)
        return dx_q + dx_k + dx_v

    def params(self):
        return (self.Wq.params() + self.Wk.params() +
                self.Wv.params() + self.Wo.params())

    def param_grads(self):
        return self._gWq + self._gWk + self._gWv + self._gWo


# ---------------------------------------------------------------------------
# Feed-Forward
# ---------------------------------------------------------------------------

class FF:
    def __init__(self, d_model, d_ff):
        self.l1 = Linear(d_model, d_ff)
        self.l2 = Linear(d_ff, d_model)
        self._h = None; self._h_raw = None

    def forward(self, x):
        h_raw    = self.l1.forward(x)
        self._h_raw = h_raw
        h        = gelu(h_raw)
        self._h  = h
        return self.l2.forward(h)

    def backward(self, dout):
        dh, self._g2 = self.l2.grads_list(dout)
        dh_raw = dh * gelu_grad(self._h_raw)
        dx, self._g1 = self.l1.grads_list(dh_raw)
        return dx

    def params(self):
        return self.l1.params() + self.l2.params()

    def param_grads(self):
        return self._g1 + self._g2


# ---------------------------------------------------------------------------
# Encoder Layer
# ---------------------------------------------------------------------------

class EncoderLayer:
    def __init__(self, d_model, n_heads, d_ff):
        self.attn = MHA(d_model, n_heads)
        self.ff   = FF(d_model, d_ff)
        self.ln1  = LayerNorm(d_model)
        self.ln2  = LayerNorm(d_model)
        self._c   = None

    def forward(self, x, mask=None):
        a_out, A  = self.attn.forward(x, mask)
        x1        = self.ln1.forward(x + a_out)
        ff_out    = self.ff.forward(x1)
        out       = self.ln2.forward(x1 + ff_out)
        self._c   = (x, x1, a_out, ff_out)
        return out, A

    def backward(self, dout):
        x, x1, a_out, ff_out = self._c
        dx1_ff, self._gln2 = self.ln2.grads_list(dout)
        d_ff  = self.ff.backward(dx1_ff)
        dx1   = dx1_ff + d_ff
        dx_a, self._gln1 = self.ln1.grads_list(dx1)
        d_attn = self.attn.backward(dx_a)
        return dx_a + d_attn

    def params(self):
        return (self.attn.params() + self.ff.params() +
                self.ln1.params() + self.ln2.params())

    def param_grads(self):
        return (self.attn.param_grads() + self.ff.param_grads() +
                self._gln1 + self._gln2)


# ---------------------------------------------------------------------------
# Embedding: element table + feature projection, concatenated
# ---------------------------------------------------------------------------

class HEAEmbedding:
    """
    out = concat( E[token_ids],  W_f @ features )
    shapes: (N, d_emb)  +  (N, d_model-d_emb)  =  (N, d_model)
    """
    def __init__(self, vocab, feat_dim, d_emb, d_model):
        assert d_model >= d_emb
        self.d_emb   = d_emb
        s = np.sqrt(2/d_emb)
        self.E       = np.random.randn(vocab, d_emb) * s
        self.fp      = Linear(feat_dim, d_model - d_emb)
        self._c      = None

    def forward(self, tok, feat):
        emb  = self.E[tok]                              # (N, d_emb)
        proj = self.fp.forward(feat)                    # (N, d_model-d_emb)
        out  = np.concatenate([emb, proj], axis=-1)    # (N, d_model)
        self._c = (tok, emb, proj)
        return out

    def backward(self, dout):
        tok, emb, proj = self._c
        de   = self.d_emb
        dE_emb  = dout[:, :de]
        dE_proj = dout[:, de:]
        dE_tbl  = np.zeros_like(self.E)
        np.add.at(dE_tbl, tok, dE_emb)
        _dfeat, dfp_grads = self.fp.grads_list(dE_proj)
        return [dE_tbl] + dfp_grads    # list aligned with params()

    def params(self):
        return [self.E] + self.fp.params()


# ---------------------------------------------------------------------------
# Full Encoder
# ---------------------------------------------------------------------------

class HEAEncoder:
    """
    HEA site-occupancy transformer encoder.

    Parameters
    ----------
    vocab     : vocabulary size (N_SP + 2 special tokens)
    feat_dim  : continuous feature dimension per site
    d_model   : transformer hidden dimension
    n_heads   : number of attention heads
    n_layers  : number of encoder layers
    d_ff      : feed-forward inner dimension
    d_emb     : element embedding dimension (factorised)
    """

    def __init__(self, vocab, feat_dim, d_model=64, n_heads=4,
                 n_layers=3, d_ff=128, d_emb=16, seed=42):
        np.random.seed(seed)
        self.emb    = HEAEmbedding(vocab, feat_dim, d_emb, d_model)
        self.layers = [EncoderLayer(d_model, n_heads, d_ff)
                       for _ in range(n_layers)]
        self.d_model = d_model
        self._emb_out = None

    def forward(self, tok, feat, mask=None):
        """
        Returns
        -------
        out      : (N, d_model) contextual site representations
        attn_all : list of (n_heads, N, N) attention per layer
        """
        x = self.emb.forward(tok, feat)
        self._emb_out = x
        attn_all = []
        for layer in self.layers:
            x, A = layer.forward(x, mask)
            attn_all.append(A)
        return x, attn_all

    def backward(self, dout):
        """
        Backpropagate dout through layers and embedding.
        Stores gradients internally; retrieve with collect_param_grads().
        """
        dx = dout
        for layer in reversed(self.layers):
            dx = layer.backward(dx)
        self._emb_grads = self.emb.backward(dx)   # list aligned with emb.params()

    def all_params(self):
        """Parameters in the SAME order as collect_param_grads()."""
        p = self.emb.params()
        for layer in self.layers:
            p = p + layer.params()
        return p

    def collect_param_grads(self):
        """
        Return gradients as a list aligned with all_params().
        Must be called AFTER backward().
        """
        g = list(self._emb_grads)
        for layer in self.layers:
            g = g + layer.param_grads()
        return g


# ---------------------------------------------------------------------------
# Prediction Heads
# ---------------------------------------------------------------------------

class MLMHead:
    """Predict element at masked positions."""
    def __init__(self, d_model, vocab):
        self.proj = Linear(d_model, vocab)

    def forward(self, enc_out, mask):
        # enc_out: (N, d_model);  mask: (N,) bool
        return self.proj.forward(enc_out[mask])   # (M, vocab)

    def params(self):
        return self.proj.params()

    def grads_list(self, dout):
        return self.proj.grads_list(dout)


class SROHead:
    """
    Mean-pool encoder output -> MLP -> SRO vector.
    2-layer MLP: d_model -> d_hidden (GELU) -> sro_dim
    """
    def __init__(self, d_model, sro_dim, d_hidden=None):
        if d_hidden is None:
            d_hidden = d_model * 2
        self.l1 = Linear(d_model, d_hidden)
        self.l2 = Linear(d_hidden, sro_dim)
        self._N  = 1
        self._c  = None

    def forward(self, enc_out):
        # enc_out: (N, d_model)
        self._N   = enc_out.shape[0]
        pooled    = enc_out.mean(0, keepdims=True)   # (1, d_model)
        h_raw     = self.l1.forward(pooled)
        self._c   = h_raw
        h         = gelu(h_raw)
        out       = self.l2.forward(h)               # (1, sro_dim)
        return out.squeeze(0)                         # (sro_dim,)

    def backward(self, d_out):
        """d_out: (sro_dim,).  Returns d_enc: (N, d_model)."""
        d = d_out[None, :]                           # (1, sro_dim)
        dh, self._g2 = self.l2.grads_list(d)
        dh_raw = dh * gelu_grad(self._c)
        d_pool, self._g1 = self.l1.grads_list(dh_raw)
        d_enc = np.tile(d_pool, (self._N, 1)) / self._N
        return d_enc

    def params(self):
        return self.l1.params() + self.l2.params()

    def param_grads(self):
        return self._g1 + self._g2


class PhaseHead:
    """Mean-pool -> Linear -> n_classes logits."""
    def __init__(self, d_model, n_classes=3):
        self.proj = Linear(d_model, n_classes)
        self._N   = 1

    def forward(self, enc_out):
        self._N  = enc_out.shape[0]
        pooled   = enc_out.mean(0, keepdims=True)   # (1, d_model)
        return self.proj.forward(pooled).squeeze(0)  # (n_classes,)

    def backward(self, d_out):
        d        = d_out[None, :]                   # (1, n_classes)
        d_pool, self._g = self.proj.grads_list(d)
        return np.tile(d_pool, (self._N, 1)) / self._N

    def params(self):
        return self.proj.params()

    def param_grads(self):
        return self._g


# ---------------------------------------------------------------------------
# Smoke test (run with: python src/models/transformer.py)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    np.random.seed(0)
    N, fd, V = 108, 18, 7
    tok  = np.random.randint(0, 5, N)
    feat = np.random.randn(N, fd)

    enc = HEAEncoder(V, fd, d_model=32, n_heads=2, n_layers=2,
                     d_ff=64, d_emb=8)
    out, attn = enc.forward(tok, feat)
    print(f"Encoder out: {out.shape}   attn[0]: {attn[0].shape}")

    sro_h   = SROHead(32, 50)
    sro_p   = sro_h.forward(out)
    print(f"SRO pred: {sro_p.shape}")

    ph_h    = PhaseHead(32, 3)
    ph_p    = ph_h.forward(out)
    print(f"Phase logits: {ph_p.shape}")

    # backward smoke test
    d_sro = np.random.randn(50)
    d_enc = sro_h.backward(d_sro)
    enc.backward(d_enc)
    g = enc.collect_param_grads()
    p = enc.all_params()
    assert len(g) == len(p), f"grad/param length mismatch: {len(g)} vs {len(p)}"
    for i, (pi, gi) in enumerate(zip(p, g)):
        assert pi.shape == gi.shape, \
            f"param[{i}] shape {pi.shape} != grad shape {gi.shape}"
    print(f"collect_param_grads: {len(g)} tensors, all shapes match.")
    print("[PASS] transformer.py smoke test")
