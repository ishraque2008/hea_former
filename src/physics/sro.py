"""
Warren-Cowley Short-Range Order (SRO) and FCC geometry utilities.

Warren-Cowley SRO:
    alpha_ij^m = 1 - P(j|i;m) / x_j

    alpha = 0  -> random solid solution (RSS)
    alpha < 0  -> unlike-pair preference (chemical ordering tendency)
    alpha > 0  -> like-pair preference (clustering / phase-separation)

Sum rules that the model must respect:
    sum_j  x_j * alpha_ij^m = 0      (composition normalization)
    x_i * alpha_ij = x_j * alpha_ji  (symmetry)
"""

import numpy as np
from itertools import product as iproduct


# ---------------------------------------------------------------------------
# FCC geometry
# ---------------------------------------------------------------------------

def build_fcc_supercell(nx: int, ny: int, nz: int, a: float = 3.6):
    """
    Build an FCC supercell of nx x ny x nz conventional unit cells.

    Returns
    -------
    coords_frac : (N, 3)  fractional coordinates in the supercell box
    box         : (3, 3)  supercell lattice vectors (rows), Angstrom
    N           : int     total atoms  (= 4 * nx * ny * nz)
    """
    basis = np.array([[0.0, 0.0, 0.0],
                      [0.5, 0.5, 0.0],
                      [0.0, 0.5, 0.5],
                      [0.5, 0.0, 0.5]])
    positions = []
    for ix, iy, iz in iproduct(range(nx), range(ny), range(nz)):
        for b in basis:
            positions.append([(b[0] + ix) / nx,
                               (b[1] + iy) / ny,
                               (b[2] + iz) / nz])
    coords_frac = np.array(positions)
    box = np.diag([nx * a, ny * a, nz * a])
    return coords_frac, box, len(positions)


def frac_to_cart(frac: np.ndarray, box: np.ndarray) -> np.ndarray:
    return frac @ box


def nearest_neighbors(coords_frac: np.ndarray, box: np.ndarray,
                      n_shells: int = 2, a: float = 3.6):
    """
    Minimum-image nearest-neighbor lists for an FCC supercell.

    Returns
    -------
    shells : list[N][n_shells]  -- atom index lists per shell
    dists  : list[N][n_shells]  -- distance lists per shell
    """
    N = len(coords_frac)
    cart = frac_to_cart(coords_frac, box)
    inv_box = np.linalg.inv(box)

    # FCC ideal shell radii in Angstrom
    ideal_r = [a / np.sqrt(2), a, a * np.sqrt(1.5)]
    tol = 0.08          # 8% tolerance

    shells = [[[] for _ in range(n_shells)] for _ in range(N)]
    dists  = [[[] for _ in range(n_shells)] for _ in range(N)]

    # Build full MIC distance matrix
    diff      = cart[:, None, :] - cart[None, :, :]   # (N,N,3)
    diff_frac = diff @ inv_box
    diff_frac -= np.round(diff_frac)
    diff_mic  = diff_frac @ box
    D = np.sqrt(np.sum(diff_mic**2, axis=-1))         # (N,N)
    np.fill_diagonal(D, np.inf)

    for i in range(N):
        d_i = D[i]
        for m in range(n_shells):
            r0 = ideal_r[m]
            lo, hi = r0 * (1 - tol), r0 * (1 + tol)
            idx = np.where((d_i >= lo) & (d_i <= hi))[0]
            shells[i][m] = idx.tolist()
            dists[i][m]  = d_i[idx].tolist()

    return shells, dists


# ---------------------------------------------------------------------------
# Warren-Cowley SRO
# ---------------------------------------------------------------------------

def warren_cowley_sro(occupancy: np.ndarray, shells: list,
                      n_species: int, n_shells: int = 2) -> np.ndarray:
    """
    Compute Warren-Cowley SRO parameters.

    Parameters
    ----------
    occupancy : (N,) int  species index 0..n_species-1
    shells    : neighbor lists from nearest_neighbors()
    n_species : number of chemical species
    n_shells  : number of neighbor shells

    Returns
    -------
    alpha : (n_shells, n_species, n_species)
    """
    N    = len(occupancy)
    n_sp = n_species
    x    = np.bincount(occupancy, minlength=n_sp) / N

    alpha = np.zeros((n_shells, n_sp, n_sp))

    for m in range(n_shells):
        pair_cnt   = np.zeros((n_sp, n_sp))
        center_cnt = np.zeros(n_sp)

        for i in range(N):
            sp_i = occupancy[i]
            nb   = shells[i][m]
            center_cnt[sp_i] += len(nb)
            for j in nb:
                pair_cnt[sp_i, occupancy[j]] += 1

        for si in range(n_sp):
            if center_cnt[si] > 0:
                P = pair_cnt[si] / center_cnt[si]
            else:
                P = np.zeros(n_sp)
            for sj in range(n_sp):
                if x[sj] > 1e-12:
                    alpha[m, si, sj] = 1.0 - P[sj] / x[sj]

    return alpha


def sro_to_flat(alpha: np.ndarray) -> np.ndarray:
    """Flatten SRO tensor (n_shells, n_sp, n_sp) to 1-D."""
    return alpha.flatten()


# ---------------------------------------------------------------------------
# MC swap to generate controlled SRO configurations
# ---------------------------------------------------------------------------

def pairwise_energy(occ: np.ndarray, shells: list,
                    J: np.ndarray) -> float:
    """E = sum_{unique pairs <i,j>} J[sp_i, sp_j]  (1st shell only)."""
    E = 0.0
    for i in range(len(occ)):
        for j in shells[i][0]:
            if j > i:
                E += J[occ[i], occ[j]]
    return E


def mc_config(occ_start: np.ndarray, shells: list,
              J: np.ndarray, n_steps: int,
              T: float, rng: np.random.Generator) -> np.ndarray:
    """
    Metropolis MC atomic swap to generate a configuration driven
    by the pair-interaction matrix J.

    Negative J_ij favors unlike pairs (ordering).
    Positive J_ij favors like pairs (clustering).
    """
    occ = occ_start.copy()
    N   = len(occ)
    E   = pairwise_energy(occ, shells, J)

    for _ in range(n_steps):
        i = rng.integers(0, N)
        j = rng.integers(0, N)
        if occ[i] == occ[j]:
            continue

        # Local energy change: only atoms neighboring i or j are affected
        affected = set(shells[i][0]) | set(shells[j][0])
        affected.discard(i)
        affected.discard(j)
        dE = 0.0
        for k in affected:
            in_i = k in shells[i][0]
            in_j = k in shells[j][0]
            sk   = occ[k]
            # before swap
            dE -= J[occ[i], sk] * in_i + J[occ[j], sk] * in_j
            # after swap (i and j exchanged)
            dE += J[occ[j], sk] * in_i + J[occ[i], sk] * in_j
        if j in shells[i][0]:
            dE += J[occ[j], occ[i]] - J[occ[i], occ[j]]

        if dE <= 0 or rng.random() < np.exp(-dE / max(T, 1e-9)):
            occ[i], occ[j] = occ[j], occ[i]
            E += dE

    return occ
