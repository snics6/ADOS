# Losses and prediction heads

Input to the model is four task vectors
\(x_3, x_4, x_{11}, x_{12}\) (pretend play, interactive play, birthday, snack),
each built as in `docs/features.md` (child + dyad scalars; optional text
embedding). One participant = one multi-task prediction.

Explanation is part of the protocol, not optional reporting.

---

## 1. Architecture

```
x_k ∈ R^F     for k ∈ {3, 4, 11, 12}
     │
     ▼
task encoder e_k = MLP_enc(x_k) ∈ R^H     (shared weights across tasks)
     │
     ▼
task attention α^{(τ)} = softmax(q_τ^T e_k / √H)   per target head τ
     │
     ▼
context c_τ = Σ_k α_k^{(τ)} e_k
     │
     ▼
shared trunk (optional): h_τ = MLP_shared(c_τ)   or h_τ = c_τ
     │
     ├── SA head   (regression)
     ├── RRB head  (regression)
     ├── C2 head   (ordinal 0–3)
     ├── B1 head   (binary)
     └── B12 head  (ordinal 0–2)
```

| Block | Role |
|-------|------|
| `MLP_enc` | Maps heterogeneous task features to a common space |
| Task attention | Soft selection of which task window matters for each target |
| Per-target heads | Different output types / losses |

Default sizes (small-N): \(H=64\), encoder 2-layer MLP with LayerNorm + GELU,
dropout 0.1–0.2, trunk optional 1-layer. Prefer under-capacity over large nets.

### Why per-target attention

Shared attention across all heads would mix “C2 should look at pretend play”
with “SA should look at social tasks.” Separate query vectors \(q_τ\) make
task-mass readable per target and support the C2 requirement below.

---

## 2. Heads and losses

Let \(y\) be the label and \(\hat{y}\) the prediction (defined per head).

### 2.1 SA, RRB — regression

- Head: linear layer → scalar \(\hat{y}\)
- Loss: SmoothL1 (Huber, \(\beta=1\)) on z-scored labels within the **training fold**

\[
\mathcal{L}_{\mathrm{SA}} = \mathrm{SmoothL1}(\hat{y}_{\mathrm{SA}},\; z(y_{\mathrm{SA}}))
\]

Same for RRB. Z-scoring is fit on the training fold only and inverted at
evaluation when reporting MAE on the original scale.

**Why SmoothL1:** N=20, outliers (e.g. SA=13) should not dominate MSE.

### 2.2 C2 — ordinal (0–3)

Levels \(\{0,1,2,3\}\). Use **ordinal regression via cumulative logits**
(CORAL-style or equivalent binary thresholds):

- Head outputs \(K-1=3\) logits \(g_1,g_2,g_3\) for thresholds \(y\ge1,\;y\ge2,\;y\ge3\)
- \(P(y \ge r) = \sigma(g_{r})\)
- Loss: sum of binary cross-entropies on the three thresholds (with optional
  class-balanced positives for rare high scores)

Point prediction for metrics:

\[
\hat{y}_{\mathrm{C2}} = \sum_{r=1}^{3} \mathbf{1}[\sigma(g_r) > 0.5]
\]

(or expected value \(\sum_r \sigma(g_r)\), rounded for discrete MAE).

**Default decode:** thresholded sum (integer 0–3). Report both integer MAE and
Spearman on the expected-value score.

### 2.3 B12 — ordinal (0–2)

Same scheme with \(K-1=2\) thresholds (\(y\ge1\), \(y\ge2\)).

### 2.4 B1 — binary (0 vs 2)

Map labels: \(0 \mapsto 0\), \(2 \mapsto 1\). No score-1 in this cohort.

- Head: linear → logit
- Loss: BCE with logits, **pos_weight** = \(n_{\mathrm{neg}}/n_{\mathrm{pos}}\) on the training fold
  (here roughly \(5:15\) globally → weight ≈ 1/3 for the positive class, or
  inverse-frequency as implemented)

---

## 3. Multi-task objective

\[
\mathcal{L} =
\lambda_{\mathrm{SA}}\mathcal{L}_{\mathrm{SA}} +
\lambda_{\mathrm{RRB}}\mathcal{L}_{\mathrm{RRB}} +
\lambda_{\mathrm{C2}}\mathcal{L}_{\mathrm{C2}} +
\lambda_{\mathrm{B1}}\mathcal{L}_{\mathrm{B1}} +
\lambda_{\mathrm{B12}}\mathcal{L}_{\mathrm{B12}}
\]

### Default λ (starting point)

| Target | λ | Rationale |
|--------|---|-----------|
| SA | 1.0 | Primary continuous severity |
| RRB | 0.5 | Narrower range; avoid dominating SA |
| C2 | 1.0 | Focal ordinal target + explanation |
| B1 | 0.5 | Binary; scaled so it does not swamp regression |
| B12 | 0.75 | Ordinal social quality |

After a dry run, optionally rebalance so that mean train-fold loss terms are
within ~2× of each other (uncertainty weighting is allowed later; not required
for v1).

Homoscedastic uncertainty weighting (Kendall et al.) is **optional v2**, not
default.

---

## 4. Explanation requirements (mandatory)

### 4.1 Task attention (always logged)

For every fold and every target \(τ\), save mean test-set attention
\(\barα_k^{(τ)}\) over tasks \(k\).

**C2 hard check (required to pass reporting):**

\[
\barα_{3}^{(\mathrm{C2})} \;\ge\; \max_{k\in\{4,11,12\}} \barα_k^{(\mathrm{C2})}
\]

i.e. pretend play has the largest average attention mass among the four tasks
for the C2 head on the held-out fold average (report per-fold and mean).

If this fails, the run is still recorded, but the C2 explanation criterion is
marked **FAIL** and the pretend-play ablation (4.2) becomes decisive.

### 4.2 Pretend-play ablation for C2 (required)

Train two models under the same fold splits:

| Model | Input |
|-------|--------|
| Full | all four task vectors |
| No-pretend | \(x_3\) zeroed (or removed) at train and test; attention over {4,11,12} only |

Required comparison on the same folds:

- \(\mathrm{MAE}(\mathrm{C2})_{\mathrm{no\text{-}pretend}} - \mathrm{MAE}(\mathrm{C2})_{\mathrm{full}} \ge 0\)
  (prefer strictly greater on mean across folds)
- Secondary: Spearman ρ should not improve when removing pretend play

Mark **PASS** if mean fold MAE worsens (or stays equal only if attention check
already PASS and degradation is ~0 within noise). Mark **FAIL** if removing
pretend play *improves* C2 — then C2 is not grounded in pretend play.

### 4.3 Dyad ablation (required interaction check)

Same folds, zero all **dyad_*** features (pose/face/speech dyad), keep child_* .

Report ΔMAE / ΔAUC for SA, B1, B12 (interaction-related targets).  
Not a hard pass/fail gate for v1, but must appear in the run summary.

### 4.4 What we do *not* count as explanation

- Post-hoc saliency on quality fields (they are not inputs)
- Examiner-only feature attributions
- Single qualitative case study without fold-level attention / ablation tables

---

## 5. Metrics (evaluation)

| Target | Primary | Secondary |
|--------|---------|-------------|
| SA | MAE (original scale), Spearman ρ | within-1 accuracy |
| RRB | MAE, Spearman ρ | within-1 |
| C2 | ordinal MAE (integer decode), Spearman ρ | exact match |
| B1 | balanced accuracy, ROC-AUC | F1 |
| B12 | ordinal MAE | macro-F1 |

All metrics: mean ± sd over 4 folds. Participant-level predictions only.

---

## 6. Optimization

- AdamW, lr \(10^{-3}\), weight decay \(10^{-4}\)
- Batch size 4–8 (full fold fit is fine given N≈15 train)
- Early stopping on mean validated multi-task score, e.g.

\[
S = -\big(
\mathrm{MAE}_{SA} + \mathrm{MAE}_{RRB} + \mathrm{MAE}_{C2} + \mathrm{MAE}_{B12}
\big)/4
+ \mathrm{BalAcc}_{B1}
\]

  (higher better; computed on val fold)
- Max epochs 100, patience 20
- Feature imputation: train-fold median for NaNs; then standardize continuous
  inputs with train-fold mean/std

---

## 7. Baseline (required companion)

Before trusting the neural multi-task model, fit **independent** sklearn
baselines on concatenated \([x_3;x_4;x_{11};x_{12}]\):

| Target | Baseline |
|--------|----------|
| SA, RRB | Ridge / ElasticNet |
| C2, B12 | Ordinal / logistic as multinomial with ordered penalty if available; else LogisticRegression on levels |
| B1 | LogisticRegression (class_weight=balanced) |

Same 4 folds. Neural results are reported **next to** these baselines.
Explanation for baselines: permutation importance or SHAP on dyad vs child and
on pretend-play block for C2.

---

## 8. Config keys

See `configs/default.yaml` → `model`, `loss_weights`, `explain`.

Implementation modules (to be filled):

- `src/ados_ml/models/multitask.py` — encoder, attention, heads
- `src/ados_ml/train/losses.py` — SmoothL1, ordinal BCE, BCE
- `src/ados_ml/eval/explain.py` — attention dump, ablations
