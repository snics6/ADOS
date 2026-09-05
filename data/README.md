# ADOS_feature — 本番 v4 マルチモーダル特徴量（70 ID）

**現行実験（実験1〜3）が使う課題区間は、この ZIP 内の自動 14 課題 CSV ではない。** 正本はリポジトリの `data/task_segments.json`（人手、task_id 1–10）。以下は配布 JSON のファイル説明である。

ADOS 収録の **参加者 ID 単位**（room カメラを時系列結合した 1 セッション）について、本番パイプライン v4 が **正常完了** した **70 ID** の解析 JSON をまとめた配布用フォルダです（2026-07-31 時点。再検出中の `d1_506` / `d1_517` は未収録）。

- **元データ**: `~/research/autism/data/processed/<ID>/multimodal_v3/`（または下記 CE 反転版）
- **動画 MP4 は含みません**（オーバーレイ QA 用 mp4 は除外）
- **1 ID = 1 フォルダ**、ファイル名の stem は **`participant_id` そのもの**（例: `d1_504_pose_v2.json`）

---

## ディレクトリ構成

```
ADOS_feature/
├── README.md                    ← 本ファイル
├── 020401/
│   ├── 020401_multimodal_unified_v1.json
│   ├── 020401_pose_v2.json
│   ├── 020401_face_v2.json
│   ├── 020401_calibration_v4.json
│   ├── 020401_parts.json
│   ├── 020401_multimodal_session_v1.json        ← ★ これ 1 本で全体把握（推奨）
│   ├── 020401_multimodal_unified_v1.json
│   ├── 020401_tasks_multimodal_v2_session.csv   ← 課題区間（結合タイムライン座標）
│   ├── 020401_tasks_session_manifest.json
│   ├── task_identification/
│   │   ├── video1_tasks_multimodal_v2.csv       ← 課題区間（各 video ローカル座標）
│   │   └── video2_tasks_multimodal_v2.csv
│   └── …（その他 JSON、mp4 なし）
├── 020404/                      ← CE ラベル反転版（ce_swap）
│   └── …
└── d1_504/
    └── …
```

| ルール | 内容 |
|--------|------|
| 第 1 階層 | 参加者 ID（`020401`, `d1_504`, `g1_408` など） |
| 第 2 階層 | その ID の **セッション全体** に対する JSON（stem = ID） |
| 時間軸 | **video1 先頭 = t=0**。複数 room 動画は結合タイムライン（境界は `*_parts.json`） |

---

## Child / Examiner ラベル反転（ce_swap）について

以下 **17 ID** は、Pose・Face の `role`（child ↔ examiner）を **全区間反転** した **`multimodal_v3_ce_swap`** の内容を収録しています。

| 付与日 | ID |
|--------|----|
| 2026-07-22 | 020404, 020412, 020422, 020443, 020534, 020541, d1_402, d1_405, d1_420, d1_446, d1_450 |
| 2026-07-31 | d1_509, g1_407, g1_408, g1_419, g1_474, g1_525 |

**重要（機械学習で混在しないこと）**

| ファイル | ce_swap ID での状態 |
|----------|-------------------|
| `*_pose_v2.json` | **反転済み** |
| `*_face_v2.json` | **反転済み** |
| `*_multimodal_unified_v1.json` | **timeline の pose/face role を反転 pose/face から再生成** |
| `*_multimodal_unified_v1.json` の `speech_segments` | **反転していない**（話者ラベルは音声 diarize のまま） |
| `*_calibration_v4.json` 他 | **反転していない** |

→ **姿勢・顔の role** は pose_v2 / face_v2 / unified の timeline を一貫して使える。発話の `speaker` を child/examiner 視覚ラベルと突き合わせる場合は、ce_swap ID では **speech と vision のラベル体系が一致しない** 点に注意。

ce_swap ID には `*_ce_swap_manifest.json` も入っています。

---

## 収録 ID 一覧（70）

020401, 020404, 020410, 020411, 020412, 020422, 020423, 020426, 020428, 020430, 020436, 020442, 020443, 020452, 020456, 020460, 020471, 020472, 020473, 020475, 020486, 020489, 020491, 020495, 020505, 020521, 020522, 020529, 020534, 020537, 020538, 020539, 020540, 020541, 020542, 020543, d1_402, d1_405, d1_406, d1_420, d1_446, d1_447, d1_450, d1_451, d1_453, d1_454, d1_458, d1_464, d1_467, d1_470, d1_476, d1_479, d1_501, d1_502, d1_503, d1_504, d1_509, d1_512, d1_515, d1_516, d1_520, d1_526, d1_527, g1_407, g1_408, g1_419, g1_444, g1_474, g1_524, g1_525

**2026-07-31 追加（14 / 予定 16）**: d1_509, d1_512, d1_515, d1_516, d1_520, d1_526, d1_527, g1_407, g1_408, g1_419, g1_444, g1_474, g1_524, g1_525  
（`d1_506`, `d1_517` は再検出完了後に追加予定）

---

## 共通の時間軸・単位

| 項目 | 値 |
|------|-----|
| サンプリング間隔 | **0.25 秒**（約 4 Hz） |
| 時刻単位 | **秒**（結合セッション先頭 = 0） |
| 元動画境界 | `*_parts.json` の `parts[].start_sec` / `end_sec` |
| 発話区間 | `speech_segments[].start/end`（連続区間。0.25 s グリッドとは独立） |
| 720×480 上画角 | 結合前に **除外済み**（room カメラのみ） |

### `*_parts.json`（セッション構成）

```json
{
  "participant_id": "d1_504",
  "session_stem": "d1_504",
  "parts": [
    {
      "video_stem": "video1",
      "start_sec": 0.0,
      "end_sec": 1788.787,
      "duration_sec": 1788.787
    },
    {
      "video_stem": "video2",
      "start_sec": 1788.787,
      "end_sec": 1992.275,
      "duration_sec": 203.488
    }
  ],
  "duration_sec": 1992.275
}
```

**用途**: 結合タイムライン上の任意時刻 `t` が元のどの ADOS 動画に属するかを復元する。タスク区間解析・leave-one-video-out 等に使う。

---

## ファイル一覧と役割

| ファイル（パターン） | 必須度 | 役割 |
|---------------------|--------|------|
| `*_multimodal_session_v1.json` | ★★★ | **統合マスター**（課題 + 完全 pose/face + 発話、1 時間軸） |
| `*_multimodal_unified_v1.json` | ★★☆ | 統合タイムライン + 発話（スリム版 pose/face） |
| `*_pose_v2.json` | ★★☆ | 骨格・派生特徴の **完全版** |
| `*_face_v2.json` | ★★☆ | 顔・MAR・AU/emotions の **完全版** |
| `*_calibration_v4.json` | ★☆☆ | child/examiner スロット校正（原型ベクトル・例画像パス） |
| `*_parts.json` | ★★☆ | 元 video1..N の時間境界 |
| `*_tasks_multimodal_v2_session.csv` | ★★☆ | **課題区間（結合タイムライン座標）** |
| `task_identification/*_tasks_multimodal_v2.csv` | ★★☆ | 課題区間（**各 video ローカル座標**・原本） |
| `*_tasks_session_manifest.json` | ★☆☆ | 課題 CSV の座標系説明 |
| `*_anchor_scenes_v6.json` | ★☆☆ | 話者分離用アンカー区間 |
| `*_voice_prototypes_v6.json` | ★☆☆ | ECAPA 声紋プロトタイプ |
| `*_calibration_audit.json` | ☆☆☆ | キャリブレーション診断ログ |
| `*_ce_swap_manifest.json` | ce_swap のみ | ラベル反転のメタデータ |

---

## 0. 統合マスター JSON — `<ID>_multimodal_session_v1.json`（推奨）

**役割**: 課題分割・pose_v2 完全版・face_v2 完全版・発話を **1 つの結合タイムライン**（video1 先頭 = `t=0`）に載せた **単一エントリポイント**。このファイルだけ読めば ML 実験の全体像が把握できます。

### トップレベル

| フィールド | 説明 |
|------------|------|
| `pipeline_version` | `"multimodal_session_v1"` |
| `time_axis` | 時間軸の定義（concat session, 秒） |
| `source_parts` | video1/2/3 の境界 |
| `task_segments` | ADOS Module 2 課題区間一覧（`session_start_sec` / `session_end_sec`） |
| `speech_segments` | 発話区間 + 重なり **`tasks`** |
| `timeline` | 0.25 s 刻みの **pose + face + 課題 + 元 video** |
| `calibration` / `speaker_counts` / `quality_summary` | 要約メタ |

### `timeline[]` 各要素（visual がある時刻のみ）

```json
{
  "time_sec": 120.25,
  "video_stem": "video1",
  "time_sec_local": 120.25,
  "tasks": [{"task_id": 1, "task_name": "構成課題", "multimodal_confidence": 0.571}],
  "pose": [{ "...": "pose_v2 完全版（features, events, keypoints 含む）" }],
  "face": [{ "...": "face_v2 完全版（aus, emotions 含む）" }]
}
```

- `time_sec` = pose_v2 の `sample_time_sec` / face_v2 の `time_sec` と一致。
- `tasks` = その瞬間にアクティブな課題（`task_segments` から `[start, end)` で判定）。
- `video_stem` / `time_sec_local` = `source_parts` から復元した元動画内位置。

### ML 利用例

```python
import json
doc = json.load(open("020401/020401_multimodal_session_v1.json"))

# 課題単位に timeline を切る
task1 = next(t for t in doc["task_segments"] if t["task_id"] == 1)
frames = [f for f in doc["timeline"]
          if task1["session_start_sec"] <= f["time_sec"] < task1["session_end_sec"]]

# 1 フレームで pose 派生量 + 課題 + 顔
for fr in frames:
    tid = fr["tasks"][0]["task_id"] if fr.get("tasks") else None
    for p in fr.get("pose", []):
        feats = p.get("features", {})
    for f in fr.get("face", []):
        mar = f.get("mar")

# 発話も課題ラベル付き
for seg in doc["speech_segments"]:
    print(seg["tasks"], seg["text"])
```

**サイズ目安**: 1 ID あたりおおよそ 50–120 MB（セッション長による）。56 ID 全体では数 GB 級。

**unified_v1 との違い**: session_v1 は **課題ラベル** + **pose/face 完全版** を含む。unified_v1 は軽量スリム版。

---

## 1. 統合 JSON — `<ID>_multimodal_unified_v1.json`

**役割**: pose_v2・face_v2・Whisper・diarize v6.1 を 0.25 s グリッドでマージした **一次利用向け** 成果物。

### トップレベル

| フィールド | 型 | 説明 |
|------------|-----|------|
| `video_id` | string | 参加者 ID |
| `video_stem` | string | 本番 v4 では ID と同一 |
| `duration_sec` | float | 結合セッション長（秒） |
| `pipeline_version` | string | `"multimodal_unified_v1"` |
| `sample_interval_sec` | float | 通常 `0.25` |
| `calibration` | object | キャリブ要約 |
| `speaker_counts` | object | `{child, examiner, other}` 発話セグメント数 |
| `timeline` | array | **0.25 s 刻み** pose/face |
| `speech_segments` | array | **発話区間**（テキスト + 話者） |
| `source_parts` | array | 元 video 境界（parts.json 相当） |
| `provenance` | object | 元ファイルパス |
| `quality_summary` | object | timeline フレーム数等 |

### `timeline[]` 各要素

```json
{
  "time_sec": 0.0,
  "pose": [
    {
      "track_id": 1,
      "role": "child",
      "role_confidence": 0.58,
      "pose_quality": 0.55,
      "visible_joint_ratio": 0.71,
      "keypoints_px": { "nose": {"x": 640, "y": 360, "visibility": 0.9}, ... }
    }
  ],
  "face": [
    {
      "track_id": 1,
      "role": "child",
      "role_confidence": 0.99,
      "bbox": [x1, y1, x2, y2],
      "det_score": 0.85,
      "mar": 0.12,
      "head_pose": {"pitch": -5.2, "yaw": 12.1, "roll": 1.3}
    }
  ]
}
```

- `pose` / `face` は検出が無い時刻ではキー省略可。
- 人物は配列（最大 2 人想定）。`role` ∈ `{child, examiner, unknown}`。
- **統合 pose はスリム版**: `keypoints_px` のみ（正規化 keypoints・features・events なし）。
- **統合 face はスリム版**: `aus` / `emotions` **なし**。

### `speech_segments[]` 各要素

| フィールド | 説明 |
|------------|------|
| `start`, `end` | 発話区間（秒） |
| `text` | Whisper 転写 |
| `speaker` | `child` / `examiner` / `other` |
| `speaker_method` | 例: `audio_primary`, `lip_hint` |
| `speaker_confidence` | float |
| `words` | 単語タイムスタンプ（任意） |
| `diarize_meta` | 音声・口唇スコア等（任意） |

### ML 利用例（統合 JSON）

```python
import json
doc = json.load(open("d1_504/d1_504_multimodal_unified_v1.json"))
# 時系列特徴: 0.25s グリッド
for frame in doc["timeline"]:
    t = frame["time_sec"]
    for p in frame.get("pose", []):
        role, kps = p["role"], p["keypoints_px"]
    for f in frame.get("face", []):
        mar, hp = f.get("mar"), f.get("head_pose")
# 発話と視覚の同期
for seg in doc["speech_segments"]:
    t_mid = (seg["start"] + seg["end"]) / 2
    # t_mid に最も近い timeline フレームを nearest で結合
```

---

## 2. Pose JSON — `<ID>_pose_v2.json`

**役割**: YOLOv8-pose + identity slot による **骨格時系列の完全版**。

### トップレベル（主要）

| フィールド | 説明 |
|------------|------|
| `duration` / `part_duration_sec` | セッション長 |
| `fps_sampled` / `frame_interval_sec` | 4.0 / 0.25 |
| `segment_count` | セグメント数 |
| `role_counts` | `{child, examiner, unknown}` |
| `quality_summary` | 二人検出率等 |
| `segments` | **主配列** |

### `segments[]` 各要素

| フィールド | ML 用途 |
|------------|---------|
| `sample_time_sec` | **主キー**（時刻） |
| `role` | child / examiner |
| `track_id` | 同一人物追跡 ID |
| `keypoints` | 正規化骨格 `{joint: {x,y,visibility}}` |
| `keypoints_px` | ピクセル骨格 |
| `bbox` | 人物 bbox |
| `features` | 派生（肘角、手首速度、手–顔距離等） |
| `events` | 離散イベント（例: `arm_extended`） |
| `pose_quality`, `visible_joint_ratio` | 品質フィルタ用 |
| `slot_scores` | `{child: float, examiner: float}` 役割スコア |

**ce_swap ID**: `role` と `role_counts` は反転済み。

---

## 3. Face JSON — `<ID>_face_v2.json`

**役割**: YOLO person + InsightFace マッチによる **顔時系列の完全版**。

### トップレベル（主要）

| フィールド | 説明 |
|------------|------|
| `sample_interval_sec` | 0.25 |
| `pyfeat_interval_sec` | 通常 1.0（AU/emotions はこの間隔のみ非 null） |
| `sample_count` / `role_counts` | 件数 |
| `samples` | **主配列** |

### `samples[]` 各要素

| フィールド | ML 用途 |
|------------|---------|
| `time_sec` | **主キー** |
| `role` | child / examiner |
| `bbox`, `det_score` | 検出品質 |
| `mar` | Mouth Aspect Ratio（発話・口開き） |
| `head_pose` | `{pitch, yaw, roll}` |
| `aus` | Action Units（**1 s 間隔のみ。それ以外 null 多**） |
| `emotions` | 感情スコア（同上） |
| `age_estimate` | 任意 |

**ce_swap ID**: `role` と `role_counts` は反転済み。

---

## 4. Calibration — `<ID>_calibration_v4.json`

child / examiner の **顔・上半身プロトタイプ**（InsightFace 512-d + CLIP）と例画像パス。

| ブロック | 内容 |
|----------|------|
| `slots.child` / `slots.examiner` | `face_prototype`, `clip_prototype`, `face_samples[]`, `body_samples[]` |
| `prototype_cosine_sim` | 二原型の分離度 |
| `calibration_window_sec` | 校正対象の動画長 |

**注意**: ce_swap では **校正スロット自体は反転していない**。pose/face の per-frame `role` のみ反転。

---

## 5. その他 JSON

### `<ID>_anchor_scenes_v6.json`
話者分離（diarize v6.1）で lip / 声紋アンカーに使った時間区間。

### `<ID>_voice_prototypes_v6.json`
ECAPA-TDNN 声紋ベクトル（child / examiner 原型）。

### `<ID>_calibration_audit.json`
キャリブレーション失敗診断（スロット割当・保存フィルタ内訳）。品質調査用。

---

## どのファイルを使うか（クイック参照）

| やりたいこと | 使うファイル |
|--------------|--------------|
| **1 ファイルで全部把握（推奨）** | `*_multimodal_session_v1.json` |
| 軽量な統合（スリム pose/face + 発話） | `*_multimodal_unified_v1.json` |
| 骨格の正規化座標・派生 features/events | `*_pose_v2.json` |
| AU / emotions（Py-Feat） | `*_face_v2.json`（**本 ZIP 56 ID ではほぼ全サンプル null**） |
| 発話テキスト + 話者 | 統合 JSON `speech_segments` |
| 元 ADOS 動画（video1/2/3）への時間復元 | `*_parts.json` または unified `source_parts` |
| **ADOS Module 2 課題区間**（14 課題） | `*_tasks_multimodal_v2_session.csv`（結合座標）または `task_identification/`（ローカル座標） |
| child/examiner 原型・例顔画像 | `*_calibration_v4.json` |
| 声紋 / lip アンカー区間 | `*_voice_prototypes_v6.json`, `*_anchor_scenes_v6.json` |
| ce_swap ID の姿勢・顔 role | pose_v2 / face_v2 / unified `timeline`（いずれも反転済みで一致） |

### unified に**ない**特徴量（別ファイルが必要）

| カテゴリ | unified に入る | unified に**入らない** |
|----------|---------------|----------------------|
| **Pose** | `keypoints_px`, `role`, 品質スコア | 正規化 `keypoints`, `bbox`, **`features`**（肘角・手首速度・二人距離等）, **`events`**（`arm_extended` 等）, `slot_scores` |
| **Face** | `mar`, `head_pose`, `bbox` | **`aus`**, **`emotions`**, `age_estimate` |
| **Speech** | テキスト・話者・words・diarize_meta | Whisper 品質（`avg_logprob` 等）、diarize ファイル全体の監査ログ |
| **校正** | 要約のみ | 原型ベクトル・例画像パス（`calibration_v4`） |
| **その他** | — | 声紋原型、アンカー区間、キャリブ監査、task CSV |

---

## タスク分割（2 層構造）

本 ZIP の特徴量は **1 ID = 1 本の結合タイムライン**（`t=0` は video1 先頭）です。区切りは **2 種類** あり、混同しないこと。

### 層 1: 収録ファイル境界 — `*_parts.json` / `source_parts`

| 項目 | 内容 |
|------|------|
| **何を区切るか** | 元の room カメラ動画 `video1`, `video2`, `video3` … |
| **本 ZIP に含まれるか** | **含まれる**（各 ID フォルダ + unified `source_parts`） |
| **時刻の意味** | 結合セッション上の `start_sec` / `end_sec`（秒） |
| **作り方** | v4 で room カメラを時系列 concat。720×480 上画角は除外 |

例: d1_504 は video1（0–1788 s）→ video2（1788–3155 s）の 2 パート。

**用途**: 元動画単位の解析、leave-one-video-out、結合タイムライン ↔ 元 stem の相互変換。

```python
def part_for_time(parts, t_sec):
    for p in parts:
        if p["start_sec"] <= t_sec < p["end_sec"]:
            return p["video_stem"]
    return None

def to_session_time(parts, video_stem, local_sec):
    for p in parts:
        if p["video_stem"] == video_stem:
            return p["start_sec"] + local_sec
    raise KeyError(video_stem)
```

### 層 2: ADOS Module 2 課題区間 — `task_identification/` + session CSV

| 項目 | 内容 |
|------|------|
| **何を区切るか** | ADOS-2 Module 2 の **14 課題**（構成課題、ごっこ遊び、おやつ …） |
| **本 ZIP に含まれるか** | **含まれる**（下記 2 形式） |
| **検出パイプライン** | `segment_tasks_multimodal.py` — **各 video stem 単位**で書き起こし + 映像融合 |
| **v4 特徴量との関係** | 特徴量は ID 単位で concat 済み。課題 CSV は **従来どおり per-video で検出**し、エクスポート時に `parts.json` オフセットで **結合座標に整合** |

**2 種類の CSV（座標系が違う）**

| ファイル | 時刻の意味 | 用途 |
|----------|-----------|------|
| `task_identification/<stem>_tasks_multimodal_v2.csv` | **その video 内ローカル秒**（`start_sec`, `end_sec`） | 原本の確認・video 単位の再現 |
| `<ID>_tasks_multimodal_v2_session.csv` | 上に加え **`session_start_sec` / `session_end_sec`**（結合タイムライン） | **unified / pose_v2 / face_v2 と直接突合** |

`<ID>_tasks_session_manifest.json` に座標系の定義を記載。

**整合性の考え方**

1. 課題検出は **video1, video2, … それぞれ単独**で実行（Whisper も stem 単位の原本を使用）。
2. v4 は room カメラを concat し、Whisper も **同じオフセット**（`parts[].start_sec`）で結合。
3. session CSV は `session_start = part_offset + start_sec_local` で生成 → **結合タイムラインと一致**。
4. 低信頼候補は `*_rejected.csv` も `task_identification/` に同梱（採用区間のみ session CSV にマージ）。

**14 課題（task_id）**: 1 構成課題, 2 呼名反応, 3 ごっこ遊び, 4 共同遊び, 5 会話, 6 共同注意, 7 実演, 8 絵の叙述, 9 本のストーリー, 10 自由遊び, 11 誕生パーティ, 12 おやつ, 13 ルーティン期待, 14 しゃぼん玉。

session CSV 主要列: 原本列 + `part_offset_sec`, `start_sec_local`, `end_sec_local`, `session_start_sec`, `session_end_sec`, `task_id`, `task_name`, `multimodal_confidence`。

### ML 利用例（課題 × 結合タイムライン）

```python
import csv, json

doc = json.load(open("d1_504/d1_504_multimodal_unified_v1.json"))
with open("d1_504/d1_504_tasks_multimodal_v2_session.csv") as f:
    tasks = list(csv.DictReader(f))

for task in tasks:
    t0, t1 = float(task["session_start_sec"]), float(task["session_end_sec"])
    frames = [fr for fr in doc["timeline"] if t0 <= fr["time_sec"] < t1]
    speech = [s for s in doc["speech_segments"] if float(s["start"]) < t1 and float(s["end"]) > t0]
    # task_id, task_name, video_stem で課題単位特徴量を集計
```

**注意**:

- 課題 CSV は **動画 1 本ごと** に検出 → session CSV で全 stem をマージ。
- 同一課題が 1 動画内で複数行になることがある。
- **parts.json は課題境界ではない**（収録ファイルの切れ目のみ）。
- 課題境界は自動推定のため、境界付近は `multimodal_confidence` でフィルタ推奨。

---

## ラベル定義

| ラベル | 意味 |
|--------|------|
| `child` | ADOS 被験児（子ども）スロット |
| `examiner` | 検査者スロット |
| `unknown` / `other` | 未割当・その他 |

話者 `speaker` は diarize v6.1（音声 + lip ヒント）に基づく。視覚 `role` は calibration v4 スロットマッチ。

---

## 既知の制約（ML 設計時）

1. **AU/emotions**: スキーマ上は face_v2 にあるが、**本 ZIP 56 ID では実データはほぼ全 null**。
2. **ce_swap と speech**: 11 ID で timeline の pose/face role は反転済みだが、`speech_segments[].speaker` は未反転。
3. **結合セッション**: 1 ID が 30–60 分級。`parts.json` で元動画単位、`tasks_multimodal_v2_session.csv` で課題単位に分割可能。
4. **欠損**: timeline に pose/face が無い 0.25 s ビンあり（検出なし）。
5. **本 ZIP に transcripts 原本なし**: Whisper・diarize 生 JSON は含まない（統合 `speech_segments` に要約済み）。
6. **課題境界は自動推定**: 手動アノテーションではない。`multimodal_confidence` を参照。

---

## 生成情報

- パイプライン: 本番 v4（ID 単位 room カメラ結合、`multimodal_v3`）
- 完了条件: `multimodal_v3/<ID>_overlay_production_audio.mp4` 存在（**本 ZIP には mp4 非収録**）
- エクスポート日: 2026-07-22
- スクリプト: `scripts/multimodal/export_ados_feature_desktop.py`
