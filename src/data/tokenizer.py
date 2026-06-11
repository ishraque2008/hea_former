"""
Site-Occupancy Tokenizer.

Each lattice site -> token with:
  token_id   : int  (element index, or MASK / PAD)
  features   : float vector encoding local chemical environment

Feature layout (per site, dim = 3*N_SP + n_shells + 3):
  [0 : n_shells*N_SP]             shell-m composition vectors (m=1,2)
  [n_shells*N_SP : n_shells*N_SP + n_shells]  normalised coordination numbers
  [n_shells*N_SP+n_shells : ...]  global composition (repeated)
  [...  : ...+3]                  fractional coordinates (x, y, z)
"""

import numpy as np
from src.data.supercell_generator import (
    N_SP, CANTOR, VOCAB, MASK_TOK, PAD_TOK
)

FCC_CN_IDEAL = [12.0, 6.0]    # expected CN per shell for perfect FCC


class SiteOccupancyTokenizer:
    """Converts an HEA occupancy array to per-site feature tensors."""

    def __init__(self, n_shells: int = 2):
        self.n_shells  = n_shells
        self.n_species = N_SP
        self.mask_tok  = MASK_TOK
        self.pad_tok   = PAD_TOK
        self.vocab     = VOCAB
        # feature dim: shell-envs + CNs + global-comp + frac-coords
        self.feat_dim  = (n_shells * N_SP      # shell composition
                          + n_shells            # coord numbers
                          + N_SP               # global composition
                          + 3)                 # fractional pos

    # ------------------------------------------------------------------
    def tokenize(self, occ: np.ndarray, shells: list,
                 frac: np.ndarray,
                 mask_prob: float = 0.0,
                 rng: np.random.Generator = None) -> dict:
        """
        Parameters
        ----------
        occ       : (N,) int  species indices
        shells    : neighbor lists
        frac      : (N,3) fractional coordinates
        mask_prob : BERT masking probability
        rng       : numpy Generator (required when mask_prob > 0)

        Returns
        -------
        dict with token_ids, true_ids, mask, features, x_global
        """
        N  = len(occ)
        ns = N_SP
        ms = self.n_shells

        x_global = np.bincount(occ, minlength=ns) / N   # (ns,)

        # ---- per-site environment features ----
        env_feat = np.zeros((N, ms * ns))
        cn_feat  = np.zeros((N, ms))
        for i in range(N):
            for m in range(ms):
                nb = shells[i][m]
                cn = len(nb)
                cn_feat[i, m] = cn / FCC_CN_IDEAL[m] if m < len(FCC_CN_IDEAL) else 0
                if cn > 0:
                    nb_counts = np.bincount(occ[np.array(nb)], minlength=ns)
                    env_feat[i, m*ns:(m+1)*ns] = nb_counts / cn

        glob_feat = np.tile(x_global, (N, 1))            # (N, ns)
        features  = np.concatenate(
            [env_feat, cn_feat, glob_feat, frac], axis=1
        ).astype(np.float64)                             # (N, feat_dim)

        # ---- BERT masking ----
        true_ids  = occ.copy()
        token_ids = occ.copy()
        mask      = np.zeros(N, dtype=bool)

        if mask_prob > 0 and rng is not None:
            for i in range(N):
                r = rng.random()
                if r < mask_prob:
                    mask[i] = True
                    r2 = rng.random()
                    if r2 < 0.80:
                        token_ids[i] = MASK_TOK
                    elif r2 < 0.90:
                        token_ids[i] = rng.integers(0, ns)
                    # else keep original (still counted as masked)

        return dict(token_ids=token_ids, true_ids=true_ids,
                    mask=mask, features=features, x_global=x_global)

    # ------------------------------------------------------------------
    def decode(self, token_ids: np.ndarray) -> list:
        names = []
        for t in token_ids:
            if t == MASK_TOK:    names.append('[MASK]')
            elif t == PAD_TOK:   names.append('[PAD]')
            elif 0 <= t < NS:    names.append(CANTOR[t])
            else:                names.append(f'?{t}')
        return names

    def feature_names(self) -> list:
        ns = N_SP
        ms = self.n_shells
        names = []
        for m in range(ms):
            for el in CANTOR:
                names.append(f'sh{m+1}_{el}')
        for m in range(ms):
            names.append(f'cn{m+1}')
        for el in CANTOR:
            names.append(f'glob_{el}')
        names += ['fx', 'fy', 'fz']
        return names


NS = N_SP   # alias for decode


# ---------------------------------------------------------------------------
# SequenceDataset
# ---------------------------------------------------------------------------

class SequenceDataset:
    """
    Wraps a raw dataset dict into a per-sample iterable.
    Each __getitem__ returns a tokenised sample augmented with SRO labels.
    """

    def __init__(self, dataset: dict, tokenizer: SiteOccupancyTokenizer,
                 mask_prob: float = 0.15, seed: int = 0):
        self.tokenizer    = tokenizer
        self.mask_prob    = mask_prob
        self.rng          = np.random.default_rng(seed)
        self.occupancies  = dataset['occupancy']
        self.shells       = dataset['shells']
        self.frac         = dataset['coords_frac']
        self.alpha_flat   = dataset['alpha_flat']
        self.phase_labels = dataset['phase_labels']
        self.mean_alpha   = dataset['mean_alpha']
        self.scenario_ids = dataset['scenario_ids']

    def __len__(self):
        return len(self.occupancies)

    def __getitem__(self, idx):
        tok = self.tokenizer.tokenize(
            self.occupancies[idx], self.shells, self.frac,
            self.mask_prob, self.rng,
        )
        tok['alpha_flat']   = self.alpha_flat[idx]
        tok['phase_label']  = int(self.phase_labels[idx])
        tok['mean_alpha']   = float(self.mean_alpha[idx])
        tok['scenario_id']  = int(self.scenario_ids[idx])
        return tok

    def flat_features(self, indices=None):
        """
        Return (X, y_sro, y_phase) averaged local-env features for sklearn.

        X : (M, 2*NS*NS + NS)  -- avg env1/env2 per species + global comp
        """
        if indices is None:
            indices = list(range(len(self)))
        ns = N_SP
        X_list, ys_list, yp_list = [], [], []
        for idx in indices:
            occ  = self.occupancies[idx]
            N    = len(occ)
            sh   = self.shells
            env1 = np.zeros((ns, ns));  cnt1 = np.zeros(ns)
            env2 = np.zeros((ns, ns));  cnt2 = np.zeros(ns)
            for i in range(N):
                si  = occ[i]
                nb1 = sh[i][0]; nb2 = sh[i][1] if len(sh[i]) > 1 else []
                if nb1:
                    c = np.bincount(occ[np.array(nb1)], minlength=ns)
                    env1[si] += c / len(nb1); cnt1[si] += 1
                if nb2:
                    c = np.bincount(occ[np.array(nb2)], minlength=ns)
                    env2[si] += c / len(nb2); cnt2[si] += 1
            for si in range(ns):
                if cnt1[si]: env1[si] /= cnt1[si]
                if cnt2[si]: env2[si] /= cnt2[si]
            x_g  = np.bincount(occ, minlength=ns) / N
            X_list.append(np.concatenate([env1.ravel(), env2.ravel(), x_g]))
            ys_list.append(self.alpha_flat[idx])
            yp_list.append(int(self.phase_labels[idx]))
        return np.array(X_list), np.array(ys_list), np.array(yp_list)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.data.supercell_generator import build_dataset

    ds  = build_dataset(nx=3, ny=3, nz=3, n_per_scenario=4,
                        n_mc_steps=200, verbose=True)
    tok = SiteOccupancyTokenizer(n_shells=2)
    print(f"feat_dim={tok.feat_dim}  names={tok.feature_names()[:6]}...")

    s = tok.tokenize(ds['occupancy'][0], ds['shells'],
                     ds['coords_frac'], mask_prob=0.15,
                     rng=np.random.default_rng(0))
    print(f"seq_len={len(s['token_ids'])}  "
          f"masked={s['mask'].sum()}  feat={s['features'].shape}")

    sds = SequenceDataset(ds, tok, mask_prob=0.15)
    X, ys, yp = sds.flat_features(list(range(4)))
    print(f"flat X={X.shape}  y_sro={ys.shape}  y_phase={yp}")
    print("Tokenizer OK")
