import numpy as np

from credit_scoring.pipeline import FEATURES, TARGET, build_pipeline


def test_pipeline_fits_and_returns_probabilities(synthetic_data):
    X, y = synthetic_data[FEATURES], synthetic_data[TARGET]
    proba = build_pipeline().fit(X, y).predict_proba(X)[:, 1]

    assert proba.shape == (len(X),)
    assert np.all((proba >= 0) & (proba <= 1))


def test_column_order_does_not_change_predictions(synthetic_data):
    X, y = synthetic_data[FEATURES], synthetic_data[TARGET]
    pipe = build_pipeline().fit(X, y)

    shuffled = X[list(reversed(FEATURES))]

    np.testing.assert_allclose(pipe.predict_proba(X), pipe.predict_proba(shuffled))
