from pathlib import Path

from ados_ml.data.tasks import Interval, load_task_intervals, merge_intervals


def test_merge_intervals():
    merged = merge_intervals(
        [Interval(0, 2), Interval(1.5, 3), Interval(5, 6)]
    )
    assert merged == [Interval(0, 3), Interval(5, 6)]


def test_load_task_intervals_020401():
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "data" / "020401" / "020401_tasks_multimodal_v2_session.csv"
    if not csv_path.exists():
        return
    by = load_task_intervals(csv_path)
    assert set(by) >= {3, 4, 11, 12}
    assert all(len(by[t]) >= 1 for t in (3, 4, 11, 12))
