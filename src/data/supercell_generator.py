"""
HEA Supercell Dataset Generator.

Generates Cantor-alloy (CrMnFeCoNi) FCC supercells across five ordering
scenarios by MC swap sampling with different pairwise interaction matrices.
Warren-Cowley SRO parameters are computed as regression labels.
"""

import numpy as np
from collections import Counter

from src.physics.sro import (
    build_fcc_supercell, nearest_neighbors,
    warren_cowley_sro, sro_to_flat, mc_config,
)

# ---------------------------------------------------------------------------
# Element vocabulary
# ---------------------------------------------------------------------------
CANTOR   = ['Cr', 'Mn', 'Fe', 'Co', 'Ni']
N_SP     = 5          # number of species
MASK_TOK = N_SP       # 5
PAD_TOK  = N_SP + 1   # 6
VOCAB    = N_SP + 2   # 7


def random_occupancy(N: int, x: np.ndarray = None,
                     rng: np.random.Generator = None) -> np.ndarray:
    """
    Draw a random equiatomic (or target-composition) occupancy.
    Composition is enforced exactly by integer counts.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if x is None:
        x = np.ones(N_SP) / N_SP
    counts = np.floor(x * N).astype(int)
    counts[np.argmax(x)] += N - counts.sum()   # distribute remainder
    occ = np.repeat(np.arange(N_SP), counts)
    rng.shuffle(occ)
    return occ


# ---------------------------------------------------------------------------
# Interaction matrices for each scenario
# ---------------------------------------------------------------------------
_J = {
    'random': np.zeros((N_SP, N_SP)),
}

def _J_ordering():
    J = np.ones((N_SP, N_SP)) * 0.10    # unlike-pair preference
    np.fill_diagonal(J, -0.10)
    return J

def _J_cluster():
    J = -np.ones((N_SP, N_SP)) * 0.08   # like-pair preference
    np.fill_diagonal(J, 0.18)
    return J

def _J_mixed():
    J = np.zeros((N_SP, N_SP))
    J[0, 2] = J[2, 0] = 0.15    # Cr-Fe ordering
    J[4, 4] = -0.15              # Ni clustering
    J[3, 3] = -0.10
    return J

def _J_CrMn():
    J = np.zeros((N_SP, N_SP))
    J[0, 1] = J[1, 0] = -0.20   # Cr-Mn pair-ordering preference
    J[2, 4] = J[4, 2] = 0.12    # Fe-Ni clustering
    return J

_J['ordering']     = _J_ordering()
_J['cluster']      = _J_cluster()
_J['mixed']        = _J_mixed()
_J['CrMn_ordering'] = _J_CrMn()

_T = {'random': 100.0, 'ordering': 0.30, 'cluster': 0.30,
      'mixed': 0.50, 'CrMn_ordering': 0.40}


# ---------------------------------------------------------------------------
# SRO labelling
# ---------------------------------------------------------------------------
def label_config(occ: np.ndarray, shells: list,
                 n_shells: int = 2) -> dict:
    """Compute all SRO-related labels for one configuration."""
    alpha      = warren_cowley_sro(occ, shells, N_SP, n_shells)
    alpha_flat = sro_to_flat(alpha)

    # Phase label: 0=random, 1=weak, 2=strong
    off = [abs(alpha[0, i, j])
           for i in range(N_SP) for j in range(N_SP) if i != j]
    maa = float(np.mean(off))
    phase = 0 if maa < 0.05 else (1 if maa < 0.15 else 2)

    x = np.bincount(occ, minlength=N_SP) / len(occ)
    return dict(alpha=alpha, alpha_flat=alpha_flat,
                mean_abs_alpha=maa, phase_label=phase, composition=x)


# ---------------------------------------------------------------------------
# Full dataset builder
# ---------------------------------------------------------------------------
def build_dataset(nx=3, ny=3, nz=3,
                  n_per_scenario=80,
                  n_mc_steps=8000,
                  scenarios=None,
                  n_shells=2,
                  seed=0,
                  verbose=True) -> dict:
    """
    Build a labeled HEA supercell dataset.

    A 3x3x3 FCC supercell (108 atoms) is the minimum for correct 2-shell SRO
    under periodic boundary conditions (box side = 3a, 2NN dist = a, ratio=3).

    Returns
    -------
    dict with keys: occupancy, coords_frac, box, shells, dists,
                    labels, alpha_flat, phase_labels, mean_alpha,
                    scenario_ids, meta
    """
    if scenarios is None:
        scenarios = ['random', 'ordering', 'cluster', 'mixed', 'CrMn_ordering']

    rng = np.random.default_rng(seed)
    coords_frac, box, N = build_fcc_supercell(nx, ny, nz)

    if verbose:
        print(f"[Dataset] {nx}x{ny}x{nz} FCC supercell: {N} atoms")
        print("[Dataset] Building neighbor lists ...", end=" ", flush=True)

    shells, dists = nearest_neighbors(coords_frac, box, n_shells=n_shells)
    nn1 = np.mean([len(shells[i][0]) for i in range(N)])
    nn2 = np.mean([len(shells[i][1]) for i in range(N)]) if n_shells > 1 else 0
    if verbose:
        print(f"done  (1NN={nn1:.0f}, 2NN={nn2:.0f})")

    occupancies, labels_list, scenario_ids = [], [], []

    for s_idx, sc in enumerate(scenarios):
        J = _J[sc]
        T = _T[sc]
        if verbose:
            print(f"[Dataset] '{sc}': {n_per_scenario} configs ...",
                  end=" ", flush=True)

        for k in range(n_per_scenario):
            sub_rng = np.random.default_rng(seed * 100_000 + s_idx * 10_000 + k)
            occ0 = random_occupancy(N, rng=sub_rng)
            if sc == 'random':
                occ = occ0
            else:
                occ = mc_config(occ0, shells, J, n_mc_steps, T, sub_rng)
            lbl = label_config(occ, shells, n_shells)
            occupancies.append(occ)
            labels_list.append(lbl)
            scenario_ids.append(s_idx)

        if verbose:
            phases = [labels_list[-(n_per_scenario - k)]['phase_label']
                      for k in range(n_per_scenario)]
            print(f"done  phases={dict(Counter(phases))}")

    alpha_flat   = np.array([l['alpha_flat']    for l in labels_list])
    phase_labels = np.array([l['phase_label']   for l in labels_list])
    mean_alpha   = np.array([l['mean_abs_alpha'] for l in labels_list])
    scenario_ids = np.array(scenario_ids)
    n_sro        = alpha_flat.shape[1]

    if verbose:
        print(f"\n[Dataset] Total={len(occupancies)} | "
              f"atoms={N} | SRO_dim={n_sro}")
        print(f"          alpha range=[{alpha_flat.min():.3f}, "
              f"{alpha_flat.max():.3f}]")
        vc = dict(zip(*np.unique(phase_labels, return_counts=True)))
        print(f"          phase counts={vc}")

    return dict(
        occupancy=occupancies, coords_frac=coords_frac,
        box=box, shells=shells, dists=dists,
        labels=labels_list, alpha_flat=alpha_flat,
        phase_labels=phase_labels, mean_alpha=mean_alpha,
        scenario_ids=scenario_ids,
        meta=dict(nx=nx, ny=ny, nz=nz, N=N, n_shells=n_shells,
                  n_species=N_SP, elements=CANTOR,
                  scenarios=scenarios, n_sro=n_sro),
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    ds = build_dataset(nx=3, ny=3, nz=3, n_per_scenario=5,
                       n_mc_steps=500, verbose=True)
    print("\nSample alpha[0] (shell-1 SRO):")
    a = ds['labels'][0]['alpha']
    for i, ei in enumerate(CANTOR):
        for j, ej in enumerate(CANTOR):
            print(f"  alpha[{ei},{ej}] = {a[0,i,j]:+.4f}")
