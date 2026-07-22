from ados_ml.utils.ids import normalize_participant_id


def test_normalize_participant_id():
    assert normalize_participant_id("020401") == "020401"
    assert normalize_participant_id("d1-402") == "d1_402"
    assert normalize_participant_id("d1_402") == "d1_402"
    assert normalize_participant_id(20401) == "020401"
