"""Load ADOS labels for the cohort."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ados_ml.utils.ids import normalize_participant_id

LABEL_COLUMNS = {
    "SA": "SA(対人的感情)",
    "RRB": "RRB(限定的・反復的行動)",
    "C2": "C-2_想像力/創造性",
    "B1": "B-1_普通ではないアイコンタクト",
    "B12": "B-12_全体的なラポールの質",
}


def load_labels(
    xlsx_path: Path,
    *,
    sheet: str,
    participant_ids: list[str] | None = None,
) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    df = df.copy()
    df["participant_id"] = df["ID"].map(normalize_participant_id)
    out = pd.DataFrame({"participant_id": df["participant_id"]})
    for key, col in LABEL_COLUMNS.items():
        out[key] = pd.to_numeric(df[col], errors="coerce")
    # B1 binary: 0 vs 2 -> 0/1
    out["B1_bin"] = out["B1"].map({0: 0, 2: 1})
    out = out.drop_duplicates("participant_id", keep="first")
    if participant_ids is not None:
        ids = [normalize_participant_id(x) for x in participant_ids]
        out = out[out["participant_id"].isin(ids)].copy()
        missing = set(ids) - set(out["participant_id"])
        if missing:
            raise ValueError(f"Labels missing for: {sorted(missing)}")
        out = out.set_index("participant_id").loc[ids].reset_index()
    return out
