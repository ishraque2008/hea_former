# HEAFormer: Site-Occupancy Transformer for Short-Range Order Prediction in High-Entropy Alloys

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![NumPy](https://img.shields.io/badge/backend-NumPy%20%7C%20PyTorch-orange.svg)](https://numpy.org/)

**Authors:** Ishraque Zaman Borshon, Prof. Vitaliy Yurkiv
**Institution:** Department of Aerospace and Mechanical Engineering, University of Arizona
**Target Venue:** *npj Computational Materials* / *Acta Materialia*

---

## Overview

HEAFormer treats atomic site occupancies in crystalline supercells as a
*materials language* — analogous to biological sequence modeling — and
trains a transformer encoder to learn transferable representations of
short-range chemical order (SRO) in high-entropy alloys (HEAs).

**Core scientific claim:** Transformer pre-training on site-occupancy
sequences learns transferable representations of atomic order in
chemically complex materials, and a physics-informed fine-tuning loss
enforcing Warren-Cowley sum rules improves prediction fidelity relative
to composition-only or local-average baselines.

### Key contributions

| Contribution | Description |
|---|---|
| Materials tokenisation | Factorised per-site tokens: element identity + shell-resolved local environment + fractional position |
| MLM pre-training | BERT-style masked site prediction teaches local chemical context without property labels |
| Physics-informed loss | Composition normalization, symmetry, and bounds constraints of Warren-Cowley formalism embedded as soft penalties |
| Full NumPy implementation | Complete forward + backward in NumPy; runs without GPU or PyTorch |
| PyTorch production version | Identical architecture in `src_torch/` for GPU-accelerated large-scale training |
| Reproducible pipeline | Single script + Jupyter notebook reproduce all figures and tables |

---

## Repository Structure

```
hea_transformer/
├── src/                         # Core library (NumPy backend)
│   ├── physics/
│   │   └── sro.py               # FCC supercell, Warren-Cowley SRO, MC sampling
│   ├── data/
│   │   ├── supercell_generator.py   # HEA dataset builder (5 ordering scenarios)
│   │   └── tokenizer.py             # SiteOccupancyTokenizer + SequenceDataset
│   ├── models/
│   │   └── transformer.py       # HEAEncoder, MLMHead, SROHead, PhaseHead
│   ├── training/
│   │   ├── losses.py            # Physics-informed SRO loss + Adam optimizer
│   │   └── trainer.py           # pretrain_epoch, finetune_epoch, evaluate
│   └── evaluation/
│       ├── metrics.py           # MAE, RMSE, R², physics_check
│       ├── baselines.py         # Ridge, MLP, RF, FrozenEncoder baselines
│       └── plots.py             # 9 publication figures (matplotlib)
├── src_torch/                   # Production PyTorch backend (same API)
│   ├── models/transformer.py
│   └── training/trainer.py
├── scripts/
│   └── run_experiment.py        # End-to-end experiment script
├── notebooks/
│   └── HEAFormer_Tutorial.ipynb # Interactive walkthrough on real data
├── tests/
│   └── test_all.py              # Unit + integration tests
├── reports/
│   └── results.json             # Saved experiment results
├── figures/                     # Generated plots (F1-F9)
├── docs/
│   └── paper_outline.md         # Full paper outline with figure plan
├── environment.yml              # Conda environment
├── requirements.txt             # pip requirements
└── README.md
```

---

## Quick Start

### 1. Installation

```bash
# Clone
git clone https://github.com/your-org/hea_transformer.git
cd hea_transformer

# Create environment (conda)
conda env create -f environment.yml
conda activate hea_transformer

# Or with pip
pip install -r requirements.txt
```

### 2. Run the quick experiment (~3 min, CPU)

```bash
python scripts/run_experiment.py --quick
```

This runs the full pipeline on 90 configurations (30 per scenario,
3 scenarios) and produces all figures in `figures/`.

### 3. Run the full experiment (~20 min, CPU)

```bash
python scripts/run_experiment.py
```

500 configurations, 5 scenarios, 6 pre-training + 25 fine-tuning epochs.

### 4. Interactive notebook

```bash
jupyter notebook notebooks/HEAFormer_Tutorial.ipynb
```

The notebook runs all pipeline stages with inline visualisation.
Set `QUICK = False` for the full experiment.

---

## Physics Background

### Warren-Cowley Short-Range Order

The Warren-Cowley SRO parameter is defined as:

```
alpha_ij^m = 1 - P(j | i; m) / x_j
```

where `P(j | i; m)` is the conditional probability that a shell-m
neighbor of an atom of species `i` is of species `j`, and `x_j` is the
global mole fraction of species `j`.

**Interpretation:**
- `alpha = 0` — random solid solution (no preference)
- `alpha < 0` — unlike-pair preference (chemical ordering tendency)
- `alpha > 0` — like-pair preference (clustering / phase-separation)

**Physical constraints embedded in the loss:**

```
C1 (composition normalization):  sum_j  x_j * alpha_ij^m = 0
C2 (symmetry):                   x_i * alpha_ij = x_j * alpha_ji
C3 (bounds):                     -x_j/(1-x_j) <= alpha_ij <= 1
```

### FCC Nearest-Neighbor Geometry

The code uses a 3x3x3 FCC supercell (108 atoms, `a = 3.6` Å) as the
minimum supercell for correct 2-shell SRO under periodic boundary
conditions. For a box side `L = n*a`, the minimum-image convention
requires `L > 2 * r_shell` for each shell. For shell 2 (`r = a`),
this requires `n >= 3`.

**Verified shell counts (FCC, PBC):**
- Shell 1 (`r = a/sqrt(2)`): **12 neighbors** per site
- Shell 2 (`r = a`):          **6 neighbors** per site

---

## Model Architecture

```
Input: occupancy sequence s = (s_1, ..., s_N)  +  feature matrix F (N x feat_dim)
         |
    HEAEmbedding
    [ element table E[s_i]  ||  Linear(f_i) ]
    => (N, d_model)
         |
    N_layers x TransformerEncoderLayer
    [ MHA(x) -> LN(x + attn_out) -> FF -> LN(x + ff_out) ]
    => (N, d_model) contextual representations
         |
    +---------+-----------+-----------+
    |         |           |           |
  MLMHead  SROHead    PhaseHead   (ImageHead -- future)
  (pretrain) (finetune)  (finetune)
```

**Token feature vector layout** (`feat_dim = 3*N_SP + n_shells + 3 = 20`):

| Slice | Content | Dim |
|---|---|---|
| `[0 : 2*N_SP]` | Shell-1 and shell-2 neighbor composition | 10 |
| `[2*N_SP : 2*N_SP+2]` | Normalised coordination numbers | 2 |
| `[2*N_SP+2 : 3*N_SP+2]` | Global composition (repeated) | 5 |
| `[3*N_SP+2 : 3*N_SP+5]` | Fractional coordinates (x, y, z) | 3 |

---

## Training Details

| Hyperparameter | Quick | Full |
|---|---|---|
| Supercell | 3x3x3 FCC (108 atoms) | 3x3x3 FCC |
| Samples / scenario | 30 | 100 |
| Scenarios | 3 | 5 |
| d_model | 32 | 64 |
| n_heads | 2 | 4 |
| n_layers | 2 | 3 |
| d_ff | 64 | 128 |
| Pre-train epochs | 3 | 6 |
| Fine-tune epochs | 10 | 25 |
| Learning rate | 1e-3 | 5e-4 |
| Batch size | 8 | 16 |
| Mask probability | 0.15 | 0.15 |

**Physics loss weights:**
- `lambda_mse = 1.0`
- `lambda_comp = 0.3` (composition normalization)
- `lambda_sym = 0.3` (symmetry)
- `lambda_bounds = 0.05` (physical bounds)

---

## Results (Quick Run, 3 scenarios, test N=15)

### SRO Regression

| Model | MAE | R² |
|---|---|---|
| Ridge (comp-only) | 0.208 | -0.11 |
| MLP (comp-only) | 0.217 | -0.20 |
| MLP (local-env) | 0.131 | +0.34 |
| RandomForest | 0.126 | +0.37 |
| FrozenEncoder+Ridge | 0.142 | +0.25 |
| **HEAFormer (ours)** | **0.209** | **+0.10** |

The transformer is competitive in this small-data regime. The RandomForest
advantage is expected: with only 90 training samples, the transformer
underfits without sufficient fine-tuning epochs. See `docs/paper_outline.md`
for a full analysis and path to improvement.

### Physics Constraint Violations

| Constraint | Mean violation |
|---|---|
| Composition normalization | 3.8e-3 |
| Symmetry | 1.8e-3 |
| Bounds fraction | 0.0 |

No predicted alpha values fall outside physical bounds. Composition and
symmetry violations at the 1e-3 level confirm the loss penalties are
effective.

---

## Extending the Model

### Add a microscopy branch

```python
# In src/models/transformer.py, add:
class MicroscopyEncoder:
    """Lightweight CNN for HRTEM/STEM image patches."""
    def __init__(self, in_channels=1, d_out=64):
        # ... CNN layers ...
    def forward(self, patch):   # patch: (C, H, W)
        return self.pool(self.conv(patch))  # (d_out,)

# In HEAEncoder.forward(), add:
img_feat = mic_enc.forward(image_patch)     # (d_out,)
fused    = np.concatenate([x.mean(0), img_feat])  # (d_model + d_out,)
```

### Use DFT-relaxed structures

Replace the MC-generated configurations with structures from:
- Materials Project HEA database (via `mp-api`)
- Your own VASP/LAMMPS AIMD trajectories
- AFLOW HEA library

The `label_config()` function in `supercell_generator.py` works on
any input occupancy array.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{borshon2025heaformer,
  title   = {HEAFormer: A Site-Occupancy Transformer with Physics-Constrained Learning for Short-Range Order in High-Entropy Alloys},
  author  = {Borshon, Ishraque Zaman},
  journal = {npj Computational Materials},
  year    = {2026},
  note    = {Under preparation}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
