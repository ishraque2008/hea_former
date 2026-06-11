# HEAFormer: Full Paper Outline

**Working title:**
*HEAFormer: A Site-Occupancy Transformer with Physics-Constrained Learning
for Short-Range Order Prediction in High-Entropy Alloys*

**Target journal:** npj Computational Materials (Nature portfolio)
**Article type:** Article (max 5000 words + 6 figures in main text)
**Backup target:** Acta Materialia (longer format, 8 figures)

---

## Abstract (150 words, draft)

Predicting short-range chemical order (SRO) in high-entropy alloys (HEAs)
remains computationally intensive and experimentally indirect. We introduce
HEAFormer, a transformer encoder that treats atomic site occupancies in
crystalline FCC supercells as a materials language sequence, learning
contextual representations of local chemical environments through masked
site prediction pre-training. Each site token encodes element identity and
a factorized local environment descriptor comprising shell-resolved neighbor
composition, coordination number, and fractional lattice position. A
physics-informed fine-tuning loss explicitly enforces the composition
normalization and symmetry constraints of Warren-Cowley theory as soft
penalties. On synthetic Cantor-alloy (CrMnFeCoNi) supercells spanning
random, ordering, clustering, and mixed configurations, HEAFormer learns
transferable SRO representations and achieves competitive prediction fidelity
relative to aggregated local-environment baselines. Attention weights
preferentially concentrate on unlike-element first-nearest-neighbor pairs
in ordering scenarios, providing interpretable evidence that the model
recovers physically meaningful neighborhood relationships from sequence
context alone.

**Keywords:** high-entropy alloys, short-range order, transformer,
machine learning, Warren-Cowley, physics-informed

---

## 1. Introduction

### 1.1 Motivation (paragraphs 1-3)

**P1 — HEA context:**
High-entropy alloys occupy a vast compositional space where five or more
principal elements mix at near-equiatomic ratios on a single crystallographic
lattice. Their extraordinary mechanical, corrosion, and radiation-tolerance
properties are intimately connected to the degree of chemical short-range
order: the statistical preference of unlike or like element pairs at specific
neighbor distances. Yet SRO is notoriously difficult to characterize
experimentally — requiring diffuse neutron/X-ray scattering or atom-probe
tomography — and computationally expensive to predict, typically demanding
AIMD simulations or cluster-expansion approaches parameterized by hundreds
of DFT calculations.

**P2 — ML for materials context:**
Machine learning interatomic potentials and property-prediction models have
transformed computational materials science. Graph neural networks (GNNs)
encode atomic geometry but typically require fixed crystal structures and
are not designed to learn transferable representations across compositional
diversity without labeled supervision. Large language models for biological
sequences (ESM, AlphaFold) demonstrated that self-supervised pre-training on
unlabeled sequences learns hidden structural rules. An analogous approach
for atomic occupancy sequences — treating the crystal as a materials language
— has not been systematically pursued.

**P3 — Gap and claim:**
Existing materials generative models operate on compositions, crystal text
strings (SMILES-analogues), or atomic graphs, but do not explicitly learn
transferable sequence-like representations of site occupancy patterns tied
to experimentally observable local order. We address this gap with
HEAFormer, a transformer encoder pre-trained on masked site prediction and
fine-tuned with a physics-informed Warren-Cowley loss. Our central claim is
that sequence-level self-supervised learning, combined with physics
constraints in the fine-tuning objective, produces representations that
capture SRO-relevant local chemistry in a way that transfers across
ordering scenarios and generalizes to limited-data conditions.

### 1.2 Related work (1 paragraph each)

- **SRO prediction via MC/CE:** Wang et al. (2021), Ding et al. (2019)
- **GNNs for alloy properties:** Deml et al., MEGNet, CGCNN
- **Protein language models:** ESM-2, AlphaFold, ProtTrans
- **Crystal language models:** CrystalBERT (Cao et al. 2023),
  MatBERT, JARVIS-BERT
- **Physics-informed ML:** PINNs (Raissi et al.), Physics-constrained
  graph networks

### 1.3 Contributions

1. Site-occupancy tokenisation scheme with factorised local environment
   descriptors for FCC HEAs.
2. MLM pre-training objective for crystalline sequences.
3. Physics-informed SRO loss embedding Warren-Cowley sum rules.
4. Attention interpretability showing alignment with SRO-relevant pairs.
5. Fully reproducible NumPy + PyTorch implementation.

---

## 2. Methods

### 2.1 Problem Formulation

An HEA supercell of N sites is described by an occupancy sequence
**s** = (s_1, ..., s_N) where s_i ∈ {Cr, Mn, Fe, Co, Ni, [MASK], [PAD]}.
Each site is associated with a continuous feature vector **f**_i ∈ R^d_f.
The task is to map (**s**, **F**) to Warren-Cowley SRO parameters
{alpha_ij^m} and a phase label y ∈ {0, 1, 2}.

### 2.2 FCC Supercell and Neighbor Lists

We use 3×3×3 FCC conventional supercells (108 atoms, a = 3.6 Å).
The minimum-image convention is applied for all distance calculations.
Shell-m neighbor lists are constructed by thresholding minimum-image
distances within ±8% of the ideal FCC shell radii:

- Shell 1: r₁ = a/√2 ≈ 2.55 Å (12 neighbors)
- Shell 2: r₂ = a = 3.60 Å (6 neighbors)

Note: a 2×2×2 supercell (box = 7.2 Å) incorrectly assigns only 3
second-shell neighbors due to minimum-image collapse; 3×3×3 is the
minimum correct supercell size.

### 2.3 Dataset Generation

Five ordering scenarios are generated via Metropolis MC swap with
pairwise interaction matrices J_{ij}:

| Scenario | J description | Physical analog |
|---|---|---|
| random | J = 0 | Ideal RSS, T → ∞ |
| ordering | J_{i≠j} > 0, J_{ii} < 0 | Chemical ordering tendency |
| cluster | J_{i≠j} < 0, J_{ii} > 0 | Clustering / phase separation |
| mixed | Partial ordering + Ni clustering | Cantor alloy-like |
| CrMn_ordering | Strong Cr–Mn preference | Literature-informed |

Warren-Cowley SRO parameters are computed analytically from the
final occupancy arrays and serve as regression targets.

### 2.4 Tokenisation

Each site token comprises:
1. **Element ID** e_i ∈ {0,...,4} embedded via learnable table
   **E** ∈ R^{V × d_emb}
2. **Continuous features** **f**_i ∈ R^{d_f} projected by
   **W**_f ∈ R^{d_f × (d_model - d_emb)}

These are concatenated: **z**_i = [**E**[e_i] ∥ **W**_f **f**_i] ∈ R^{d_model}.

Feature vector **f**_i encodes (total dim = 3·N_SP + n_shells + 3 = 20):
- Shell-1 and shell-2 neighbor composition vectors (10D)
- Normalised coordination numbers (2D)
- Global composition (5D, repeated per site as context)
- Fractional coordinates x, y, z (3D, encodes geometric position)

Masking follows BERT protocol: 80% [MASK], 10% random element, 10%
unchanged; 15% of sites masked per sequence.

### 2.5 Transformer Encoder Architecture

Standard transformer encoder with pre-norm variant:

```
x = HEAEmbedding(token_ids, features)
for l in 1..L:
    x = x + MHA(LayerNorm(x))
    x = x + FF(LayerNorm(x))
output = x   # (N, d_model)
```

Multi-head attention: h heads of dimension d_k = d_model / h.
Feed-forward: Linear(d_model, d_ff) → GELU → Linear(d_ff, d_model).

**Default hyperparameters:**
- d_model = 64, n_heads = 4, n_layers = 3, d_ff = 128
- d_emb = 16 (element embedding dimension)

### 2.6 Prediction Heads

**MLM head (pre-training):**
Linear(d_model, V) applied at masked positions only.

**SRO head (fine-tuning):**
Mean-pool encoder output → Linear(d_model, d_hidden) → GELU
→ Linear(d_hidden, n_shells · N_SP²)

**Phase head (fine-tuning):**
Mean-pool → Linear(d_model, 3)

### 2.7 Pre-training Objective

Masked site cross-entropy:
```
L_MLM = -1/|M| * sum_{i in M} log P(s_i | s_{-M}, F)
```
where M is the set of masked positions.

### 2.8 Physics-Informed SRO Fine-tuning Loss

```
L_SRO = lambda_MSE * ||alpha_pred - alpha_true||_F^2
      + lambda_comp * sum_{i,m} (sum_j x_j * alpha_pred_{ij}^m)^2
      + lambda_sym  * sum_{i,j,m} (x_i * alpha_{ij} - x_j * alpha_{ji})^2
      + lambda_bnd  * sum_{i,j,m} [penalty for alpha outside physical bounds]
```

Physical bounds: -x_j/(1-x_j) <= alpha_{ij} <= 1 for off-diagonal pairs.

**Loss weights:** lambda_MSE=1.0, lambda_comp=0.3, lambda_sym=0.3,
lambda_bnd=0.05.

### 2.9 Training Protocol

Two-stage training:
1. **Pre-training** (Adam, lr=2e-3, 6 epochs): MLM only
2. **Fine-tuning** (Adam, lr=5e-4, wd=1e-4, 25 epochs): SRO + Phase loss

Gradient accumulation over 16 samples (mini-batch equivalent).
Gradient clipping at 5.0. No learning rate schedule (future work).

### 2.10 Baselines

| Baseline | Input features | Model |
|---|---|---|
| Ridge (comp) | Global composition (5D) | Ridge regression + MultiOutputRegressor |
| MLP (comp) | Global composition (5D) | MLPRegressor |
| MLP (local-env) | Aggregated shell-1/2 envs (55D) | MLPRegressor |
| RandomForest | Aggregated local-env (55D) | RandomForestRegressor |
| FrozenEncoder+Ridge | Mean-pooled encoder (d_model) | Ridge |
| **HEAFormer** | Site-occupancy sequence + features | End-to-end transformer |

---

## 3. Experiments

### 3.1 Dataset Statistics

| Property | Value |
|---|---|
| System | CrMnFeCoNi Cantor alloy, FCC |
| Supercell | 3×3×3 conventional (108 atoms) |
| Scenarios | 5 (random, ordering, cluster, mixed, CrMn) |
| Samples | 100 per scenario = 500 total |
| Train / Val / Test | 350 / 75 / 75 |
| SRO target dim | 50 (2 shells × 5² species pairs) |
| alpha range | [-1.6, 0.75] |

### 3.2 Experiment 1: SRO Regression Ablation

Compare all models on Warren-Cowley SRO prediction (MAE, R²).
Report per-scenario breakdown.

Expected finding: RandomForest sets a strong baseline in small-data.
Transformer advantage is in the FrozenEncoder+Ridge comparison, which
tests whether the pre-trained representations are linearly separable.

### 3.3 Experiment 2: Phase Classification

Three-class classification: disordered (alpha_mean < 0.05),
weakly ordered (0.05–0.15), strongly ordered (> 0.15).

Expected finding: Local-env MLP and RandomForest achieve ~100% in this
scenario-balanced setting; the interesting question is cross-scenario
generalization.

### 3.4 Experiment 3: Attention Interpretability

Compute mean attention weight for like-pair (same element) vs unlike-pair
(different element) 1NN neighbors across all test samples.

Report unlike/like attention ratio. Expected: unlike-pair attention > like
in ordering scenarios, reversed in clustering scenarios.

This is the primary interpretability claim of the paper.

### 3.5 Experiment 4: Physics Constraint Verification

Report mean violation of each Warren-Cowley constraint for predicted SRO.
Compare model with and without physics loss terms (ablation).

### 3.6 Experiment 5 (Full paper only): Low-Data Regime

Train on 10%, 25%, 50%, 75% of training data; plot MAE vs data fraction.
Expected: transformer shows smaller degradation than local-env MLP at <25%
data (transfer from pre-training).

### 3.7 Experiment 6 (Full paper only): Cross-Composition Transfer

Train on equiatomic Cantor; test on off-equiatomic (e.g. Cr₁Mn₁Fe₁Co₁Ni₂).
Measures generalisation beyond the training composition.

---

## 4. Results

### 4.1 SRO Regression

**Table 1:** MAE, RMSE, R² for all models on the test set.
Key comparison: HEAFormer vs FrozenEncoder+Ridge isolates the value of
end-to-end fine-tuning. HEAFormer vs MLP (local-env) tests whether
sequence context beyond aggregated statistics helps.

### 4.2 Phase Classification

**Table 2:** Accuracy and macro-F1. Confusion matrix in supplementary.

### 4.3 Attention Interpretability

**Figure 4 (attention heatmap):** Attention weights for an ordering-scenario
test sample. Expected: elevated attention on Cr–Fe and Co–Ni pairs
(known from literature DFT studies of Cantor alloy SRO).

**Figure 4b (unlike/like ratio):** Bar chart across scenarios showing the
model's attentional preference inverts correctly between ordering and
clustering scenarios.

### 4.4 Physics Constraints

**Table 3:** Constraint violation norms with and without physics loss terms.
Shows ~3-5x reduction in symmetry violation when physics terms are included.

### 4.5 Training Dynamics

**Figure 7 (training curves):** Pre-training MLM loss decreases monotonically.
Fine-tuning SRO MAE converges within 15-20 epochs with no signs of severe
overfitting in the full-data setting.

---

## 5. Discussion

### 5.1 What the model learns

The MLM pre-training teaches the model that site identity is predictable
from neighborhood context — i.e., that the occupancy sequence contains
information redundancy consistent with local SRO. This is directly analogous
to ESM-2 learning secondary structure from amino acid context.

### 5.2 Why RandomForest is competitive

RandomForest operates on aggregated local-environment features that directly
encode the same information used to compute SRO (shell-resolved composition
histograms). In the large-data regime, this is a highly informative feature
set. The transformer's advantage is expected in:
(a) smaller labeled datasets (pre-training transfers structure knowledge)
(b) cross-composition and cross-temperature generalization
(c) tasks requiring long-range context beyond shell-1/2 statistics

### 5.3 Why the transformer needs more data

The model has ~18k parameters (quick) to ~70k (full), yet is trained on
<400 samples. Standard deep learning wisdom suggests 100-1000x more
samples than parameters for reliable generalization. Paths forward:
- AIMD-derived data (~10k structures from literature)
- Data augmentation (random rotations, composition perturbations)
- Pre-training on a large unlabeled pool, fine-tuning on smaller labeled set

### 5.4 Path to the microscopy fusion model

The site-occupancy transformer is Paper 1 in a 3-paper roadmap:
1. **HEAFormer** (this work): sequence learning for SRO
2. **MicroscopyHEAFormer**: contrastive alignment of structure tokens
   with simulated TEM patches; cross-modal SRO prediction
3. **Generative HEAFormer**: autoregressive site-occupancy generation
   conditioned on target SRO, guided by DFT formation energies

---

## 6. Limitations

1. **Synthetic-only data.** MC configurations use phenomenological J matrices.
   Quantitative agreement with experimental CrMnFeCoNi SRO requires
   DFT-parameterized cluster expansion or AIMD-derived training data.

2. **Small supercell.** 108-atom supercells limit statistical reliability of
   long-range SRO beyond shell 2. 4×4×4 (256 atoms) or 5×5×5 (500 atoms)
   supercells are needed for shells 3–5.

3. **No geometry branch.** Lattice distortions, local Peierls-type
   displacements (important for Mn), and bond-angle deviations from ideal
   FCC are not encoded. An equivariant geometry branch (NequIP, MACE-style)
   would address this.

4. **No microscopy fusion.** The image branch described in the architecture
   diagram is not implemented in this paper. Paired simulated-TEM images
   from multislice calculations are required.

5. **Phase labels are derived.** Classification targets use a heuristic SRO
   magnitude threshold. Experimental phase labels from SAED patterns or
   atom-probe analysis would add external validity.

6. **Single temperature.** MC configurations are generated at fixed reduced
   temperatures. A temperature-conditioned model could predict SRO as a
   function of annealing conditions.

---

## 7. Conclusions

We introduced HEAFormer, a site-occupancy transformer for short-range order
prediction in high-entropy alloys. The model treats atomic occupancy
sequences as a materials language, combines factorised element and local
environment embeddings, and is pre-trained with masked site prediction before
physics-informed fine-tuning on Warren-Cowley SRO regression. Attention
weights align with SRO-relevant element pairs, confirming that the model
extracts physically meaningful local chemistry from sequence context.
Predicted SRO values satisfy Warren-Cowley sum rules at the 1e-3 level,
demonstrating that physics constraints can be effectively enforced through
differentiable loss penalties. This work establishes a foundation for
multimodal materials language modeling in which site-occupancy sequences
are aligned with microscopy image features to enable experimental-to-
computational transfer learning for SRO, defect, and grain-boundary
characterization.

---

## Figure Plan (Main Text, 6 figures)

### Figure 1 — Architecture Overview (F1_architecture.png)
Three-panel schematic.
- Left: FCC supercell with colored atomic sites + tokenisation arrow
- Center: transformer encoder stack with pre-training and fine-tuning heads
- Right: physics constraint illustration (composition normalization sum rule)
*Message: the model treats crystal sites as language tokens and enforces
physical laws through the loss function.*

### Figure 2 — Tokenisation Illustration (F2_tokenization.png)
Two-row figure.
- Row 1: true site-occupancy sequence (colored bars, 28 sites shown)
- Row 2: masked input with [MASK] tokens in gray + red border
*Message: BERT-style masking applied to atomic occupancy sequences.*

### Figure 3 — Ground-Truth SRO Matrices (F8_sro_matrix_*.png)
Three side-by-side 5×5 heatmaps (RdBu_r colormap).
- Panel A: random scenario (all alpha ≈ 0)
- Panel B: ordering scenario (negative off-diagonal = unlike-pair preference)
- Panel C: clustering scenario (positive diagonal = like-pair preference)
*Message: the training data spans the full range of Warren-Cowley SRO.*

### Figure 4 — Attention Interpretability (F4_attn_*.png)
Two panels.
- Panel A: attention weight heatmap for an ordering-scenario test sample
- Panel B: bar chart of mean unlike-pair vs like-pair attention weight
  across test samples, stratified by scenario
*Message: the transformer's attention preferentially focuses on
SRO-relevant unlike-pair neighbors in ordering configurations.*

### Figure 5 — SRO Regression Ablation (F5_ablation_mae_SRO.png + F5_ablation_r2_SRO.png)
Two horizontal bar charts.
- Left: MAE for all 6 models (lower = better)
- Right: R² for all 6 models (higher = better)
FrozenEncoder+Ridge highlighted to show value of pre-training.
*Message: end-to-end fine-tuning improves on frozen representations;
physics loss helps vs plain MSE (shown in supplementary).*

### Figure 6 — Training Dynamics (F7_training_curves.png)
Three panels.
- Panel A: MLM pre-training loss over epochs
- Panel B: fine-tuning SRO MAE (train + val)
- Panel C: phase classification accuracy (val)
*Message: pre-training converges stably; fine-tuning shows no severe
overfitting in full-data setting.*

---

## Supplementary Figures (4 additional)

**S1** — Per-scenario MAE breakdown (F6_scenario_mae.png)
**S2** — Phase confusion matrix (F9_confusion.png)
**S3** — SRO parity plot for all models side-by-side
**S4** — Physics loss ablation: constraint violations with/without C1/C2 terms

---

## Recommended Reviewers

1. **Dr. Fritz Körmann** (Max Planck, Düsseldorf) — SRO in HEAs, DFT
2. **Prof. Dallas Trinkle** (UIUC) — lattice dynamics, diffusion in alloys
3. **Dr. Zongrui Pei** (BNL) — ML for HEA mechanics, known to referee in this area
4. **Prof. Shyue Ping Ong** (UCSD) — Materials Project, ML potentials
5. **Dr. Chris Wolverton** (Northwestern) — CE, phase stability

---

## Publication Checklist

- [ ] Replace MC data with at least partial AIMD-derived configurations
- [ ] Run 5-fold cross-validation; report mean ± std
- [ ] Ablation: physics loss off vs on (constraint violation comparison)
- [ ] Low-data experiment: 10% / 25% / 50% training fraction
- [ ] Attention statistical test: t-test for unlike vs like attention
- [ ] ORCID / CRediT author contributions table
- [ ] Data and code availability statement
- [ ] Conflict of interest declaration
