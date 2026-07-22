from ados_ml.features.stats import theil_sen_slope, weighted_mean, weighted_std


def test_weighted_mean_std():
    assert abs(weighted_mean([1, 3], [1, 1]) - 2.0) < 1e-9
    assert abs(weighted_std([1, 3], [1, 1]) - 1.0) < 1e-9


def test_theil_sen_line():
    # y = 2x
    t = [0, 1, 2, 3]
    v = [0, 2, 4, 6]
    assert abs(theil_sen_slope(t, v) - 2.0) < 1e-9
