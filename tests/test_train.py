import numpy as np

from credit_scoring.train import choose_threshold, prepare_data


def test_prepare_data_keeps_all_rows(synthetic_data):
    X_train, X_test, y_train, y_test = prepare_data(synthetic_data)

    assert len(X_train) + len(X_test) == len(synthetic_data)
    assert len(y_train) + len(y_test) == len(synthetic_data)


def test_split_is_stratified(synthetic_data):
    _, _, y_train, y_test = prepare_data(synthetic_data)

    assert abs(y_train.mean() - y_test.mean()) < 0.02


def test_split_is_deterministic(synthetic_data):
    first = prepare_data(synthetic_data)
    second = prepare_data(synthetic_data)

    for a, b in zip(first, second, strict=True):
        assert a.index.equals(b.index)


def test_choose_threshold_is_a_valid_probability():
    y = np.array([0, 0, 1, 1, 0, 1])
    proba = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])

    assert 0 <= choose_threshold(y, proba) <= 1
