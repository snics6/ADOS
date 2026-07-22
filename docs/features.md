# Feature list

Features are computed **inside each of the four task intervals**, then used as
per-task vectors (later fused by the model). Aggregation uses confidence
weights; quality variables themselves are not predictors.

Notation:

- \(t\) indexes 0.25 s timeline samples (pose/face) or speech segments.
- \(w_t\) is a soft confidence weight in \([0, 1]\).
- Weighted mean: \(\mathrm{wmean}(x) = \sum_t w_t x_t / \sum_t w_t\) (undefined if \(\sum w_t = 0\)).
- Weighted std: analogous, population form over the same weights.
- Rate features: count of events / task duration (seconds), unless noted.

Missing: if a feature is undefined for a task (no valid weights), store NaN and
apply a fixed imputation policy at train time (median of training fold only).

---

## 1. Confidence weights (not predictors)

### Pose sample weight

For a pose instance with role `child` or entering a dyad pair:

\[
w^{\mathrm{pose}}_t =
\sigma(\mathrm{pose\_quality}_t) \cdot
\sigma(\mathrm{role\_confidence}_t)
\]

Use \(\sigma(u)=\mathrm{clip}(u, 0, 1)\) when scores are already in \([0,1]\).

**Dyad pose validity** at \(t\): both child and examiner pose instances exist and

\[
\min(w^{\mathrm{pose,child}}_t, w^{\mathrm{pose,ex}}_t) \ge \tau_{\mathrm{pose}}
\]

Default \(\tau_{\mathrm{pose}} = 0.25\). Dyad pose features are computed only on
valid \(t\); their aggregation weight is
\(\min(w^{\mathrm{pose,child}}_t, w^{\mathrm{pose,ex}}_t)\).

### Face sample weight

\[
w^{\mathrm{face}}_t =
\sigma(\mathrm{det\_score}_t) \cdot
\sigma(\mathrm{role\_confidence}_t)
\]

Dyad face validity: both roles present and
\(\min(w^{\mathrm{face,child}}_t, w^{\mathrm{face,ex}}_t) \ge \tau_{\mathrm{face}}\)
(default \(0.25\)).

### Speech segment weight

\[
w^{\mathrm{sp}}_s = \sigma(\mathrm{speaker\_confidence}_s)
\]

Optional hard drop if \(w^{\mathrm{sp}}_s < \tau_{\mathrm{sp}}\) (default \(0.3\)).

### Explicitly excluded from the feature vector

`pose_quality`, `visible_joint_ratio`, `det_score`, `role_confidence`,
`speaker_confidence`, two-person / visibility rates, bbox size / image location,
raw track IDs.

---

## 2. Pose — child

Source: `*_pose_v2.json` / session timeline `pose[]` with `role == child`,
restricted to the task window. Use `features` and `events`.

| ID | Feature | Definition |
|----|---------|------------|
| P-C01 | `child_left_elbow_angle_wmean` | wmean of `left_elbow_angle_deg` |
| P-C02 | `child_left_elbow_angle_wstd` | wstd of `left_elbow_angle_deg` |
| P-C03 | `child_right_elbow_angle_wmean` | wmean of `right_elbow_angle_deg` |
| P-C04 | `child_right_elbow_angle_wstd` | wstd of `right_elbow_angle_deg` |
| P-C05 | `child_left_wrist_speed_wmean` | wmean of `left_wrist_speed` (skip nulls) |
| P-C06 | `child_right_wrist_speed_wmean` | wmean of `right_wrist_speed` (skip nulls) |
| P-C07 | `child_wrist_speed_wmean` | mean of available left/right speed wmeans |
| P-C08 | `child_hand_to_face_dist_wmean` | wmean of `hand_to_face_dist` |
| P-C09 | `child_hand_to_face_dist_wstd` | wstd of `hand_to_face_dist` |
| P-C10 | `child_torso_lean_wmean` | wmean of `torso_lean_deg` |
| P-C11 | `child_torso_lean_wstd` | wstd of `torso_lean_deg` |
| P-C12 | `child_head_drop_wmean` | wmean of `head_drop` |
| P-C13 | `child_joint_jitter_wmean` | wmean of `joint_jitter` |
| P-C14 | `child_arm_extended_rate` | duration covered by `events` type `arm_extended` / task_dur |

Weights: \(w^{\mathrm{pose}}_t\) for that child instance. Null scalar fields are
skipped inside the weighted sum (renormalize over non-null \(t\)).

---

## 3. Pose — dyad

Computed only on dyad-valid timesteps.

| ID | Feature | Definition |
|----|---------|------------|
| P-D01 | `dyad_distance_wmean` | wmean of `inter_person_distance` (from either instance; same value) |
| P-D02 | `dyad_distance_wstd` | wstd of `inter_person_distance` |
| P-D03 | `dyad_distance_slope` | robust slope of distance vs time on valid \(t\) (e.g. Theil–Sen) |
| P-D04 | `dyad_close_frac` | fraction of valid \(t\) with distance < cohort median of **positive** distances |
| P-D05 | `dyad_facing_angle_diff_wmean` | wmean of `facing_angle_diff_deg` |
| P-D06 | `dyad_facing_angle_diff_wstd` | wstd of `facing_angle_diff_deg` |
| P-D07 | `dyad_face_each_other_frac` | fraction of valid \(t\) with `facing_angle_diff_deg` < 45° |
| P-D08 | `dyad_valid_time_frac` | valid dyad pose time / task_dur |

`dyad_valid_time_frac` is a coverage descriptor of the interaction signal, not a
camera visibility rate of a single person; keep it. Do **not** add single-person
visibility rates.

**Data note:** `inter_person_distance` is 0 on most frames in this release.
Distance features (P-D01–P-D04) have limited dynamic range; facing-angle dyad
features (P-D05–P-D07) are preferable for interaction until distance is fixed.

---

## 4. Face — child

Source: face samples with `role == child`. AU / emotions are unused (empty in
this release).

| ID | Feature | Definition |
|----|---------|------------|
| F-C01 | `child_mar_wmean` | wmean of `mar` |
| F-C02 | `child_mar_wstd` | wstd of `mar` |
| F-C03 | `child_mar_p90` | weighted 90th percentile of `mar` |
| F-C04 | `child_yaw_wmean` | wmean of `head_pose.yaw` |
| F-C05 | `child_yaw_wstd` | wstd of yaw |
| F-C06 | `child_pitch_wmean` | wmean of pitch |
| F-C07 | `child_pitch_wstd` | wstd of pitch |
| F-C08 | `child_roll_wstd` | wstd of roll |
| F-C09 | `child_yaw_abs_wmean` | wmean of \|yaw\| |

---

## 5. Face — dyad

On dyad-valid face timesteps (both roles).

| ID | Feature | Definition |
|----|---------|------------|
| F-D01 | `dyad_yaw_diff_wmean` | wmean of \|yaw_child − yaw_ex\| |
| F-D02 | `dyad_yaw_diff_wstd` | wstd of \|yaw_child − yaw_ex\| |
| F-D03 | `dyad_mutual_orient_frac` | frac. where \|yaw_c−yaw_e\| \< 30° and both \|pitch\| \< 25° |
| F-D04 | `dyad_mar_crosscorr` | Pearson corr(mar_child, mar_ex) on overlapping valid \(t\) (NaN if \< 8 points) |
| F-D05 | `dyad_face_valid_time_frac` | valid dyad face time / task_dur |

---

## 6. Speech — child

Source: `speech_segments` with `speaker == child` overlapping the task interval
(segment overlap length \> 0). Text embeddings: frozen sentence encoder
(configured in code; not fine-tuned in the default setup).

| ID | Feature | Definition |
|----|---------|------------|
| S-C01 | `child_speech_time_frac` | sum of overlap durations / task_dur |
| S-C02 | `child_n_utterances` | number of overlapping child segments |
| S-C03 | `child_utt_dur_wmean` | wmean segment duration (weight \(w^{\mathrm{sp}}\)) |
| S-C04 | `child_chars_per_sec` | total overlapping characters / task_dur |
| S-C05 | `child_emb_mean` | mean of segment embedding vectors (L2-normalized per segment), shape \(d\) |
| S-C06 | `child_emb_std` | std of embeddings across segments (scalar mean of dims, or full \(d\) — implement as mean-pooled std scalar + keep `emb_mean` vector) |

Default: store `child_emb_mean` as a vector (\(d\), e.g. 768) and
`child_emb_disp` as mean feature-wise std (scalar) instead of full std vector.

---

## 7. Speech — dyad

Use diarization `speaker` labels (`child` / `examiner`). Do not substitute vision
roles.

Sort overlapping segments by start time within the task.

| ID | Feature | Definition |
|----|---------|------------|
| S-D01 | `dyad_turn_count` | number of speaker changes between consecutive segments |
| S-D02 | `dyad_turn_rate` | turn_count / task_dur |
| S-D03 | `child_response_latency_wmean` | wmean of (child_start − previous_examiner_end) for child segments that follow an examiner segment; negatives clipped to 0; max gap 10 s or NaN |
| S-D04 | `examiner_to_child_frac` | fraction of child utterances that immediately follow an examiner utterance |
| S-D05 | `speech_overlap_frac` | fraction of task time where child and examiner speech intervals overlap |
| S-D06 | `child_vs_examiner_time_ratio` | child_speech_time / max(examiner_speech_time, ε) |
| S-D07 | `dyad_emb_diff_norm` | \|mean_emb_child − mean_emb_examiner\|₂ (semantic divergence) |

---

## 8. Per-task vector layout

For each task \(k \in \{3,4,11,12\}\):

```
x_k = [
  pose_child (P-C*),
  pose_dyad  (P-D*),
  face_child (F-C*),
  face_dyad  (F-D*),
  speech_child scalars (S-C01..04, S-C06),
  speech_dyad scalars (S-D*),
  speech_child_emb_mean (S-C05, dim d),
]
```

Approximate scalar count (excluding embedding): ~14 + 8 + 9 + 5 + 5 + 7 ≈ **48**
scalars per task, plus embedding \(d\).

Four tasks → model input as either:

- concatenated \([x_3; x_4; x_{11}; x_{12}]\), or
- stacked \((4, F)\) with a task-attention encoder (preferred for C2 explanations).

---

## 9. Task-specific emphasis (reporting, not hard masking)

| Target | Expected informative tasks |
|--------|----------------------------|
| C2 | Pretend play (3) — **required** in attention / ablation reports |
| B1, B12, SA | Interactive play, birthday, snack (4, 11, 12) |
| RRB | All four; no single-task requirement |

Default training still feeds all four task vectors to every head; explanation
protocols test whether C2 relies on pretend play.

---

## 10. Implementation mapping (data files)

| Need | File |
|------|------|
| Task intervals | `<id>_tasks_multimodal_v2_session.csv` |
| Pose features / events | `<id>_multimodal_session_v1.json` `timeline[].pose` (preferred) or `*_pose_v2.json` |
| Face | same session JSON `timeline[].face` or `*_face_v2.json` |
| Speech | session JSON `speech_segments` |
| Labels | `data/ADOS2_result_2.xlsx` |

Prefer `*_multimodal_session_v1.json` so task membership and full pose/face live
on one timeline.
