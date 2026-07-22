# Experimental design

## Goal

Multi-task prediction of ADOS-2 Module 2 scores from multimodal features
extracted from room-camera sessions (face, pose, speech text).

## Cohort

See `configs/cohorts/four_tasks_20.yaml`.

Inclusion: participant IDs without a `d` prefix for which automatic task
detection found all of: pretend play, interactive play, birthday party, snack.

## Time windows

Only frames / speech overlapping the four task intervals on the concatenated
session timeline (`session_start_sec`, `session_end_sec`). No between-task or
out-of-task context.

If a task has multiple detected segments for one ID, use all of them (union)
unless a later ablation specifies otherwise.

## Roles

| Stream | Use |
|--------|-----|
| Child | Primary behavioral predictors |
| Dyad (child–examiner) | Interaction predictors |
| Examiner alone | Not used as predictors (may appear only inside dyad definitions) |

Labels are child ADOS scores. Interpretations should stay in terms of child
behavior and child–examiner interaction.

## Quality / confidence

Fields such as `pose_quality`, `visible_joint_ratio`, `det_score`,
`role_confidence`, and `speaker_confidence` are **not** model inputs.

They are used only to compute soft weights when aggregating behavioral
features over time (and to decide whether a dyad feature is defined at a
timestep). See `docs/features.md`.

## Targets

| Name | Supervision | Evaluation (primary) |
|------|-------------|----------------------|
| SA | regression | MAE, Spearman ρ |
| RRB | regression | MAE, Spearman ρ |
| C2 | ordinal (0–3) | ordinal MAE, Spearman ρ |
| B1 | binary 0 vs 2 | balanced accuracy, AUC |
| B12 | ordinal (0–2) | ordinal MAE, macro-F1 (secondary) |

Heads, losses, task attention, and mandatory explanation / ablation rules:
**`docs/losses_and_heads.md`**.

### C2 explanation requirement (summary)

1. Task attention: pretend play should carry the largest mass for the C2 head.
2. Ablation: removing pretend-play inputs must not improve C2 MAE.
3. Dyad ablation reported for SA / B1 / B12 (interaction check).

## Cross-validation

- 4-fold CV at the **participant** level (no segment from a test ID in train).
- Prefer light stratification by SA tertile when forming folds.
- Report mean ± sd across folds.

## Hardware

Training / embedding extraction target: NVIDIA A5000.

## Leakage notes

- Normalize label IDs (`d1-402` → `d1_402`) before joining.
- For `ce_swap` IDs, vision `role` is swapped; `speech_segments.speaker` is not.
  Dyad speech features must use diarization speaker labels; dyad vision
  features must use vision roles. Do not naively align the two label systems
  without an explicit mapping rule.
