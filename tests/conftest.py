import joblib
import numpy as np
import pandas as pd
import pytest

from credit_scoring.pipeline import FEATURES, TARGET, build_pipeline


@pytest.fixture
def synthetic_data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame(
        {
            "age": rng.integers(21, 70, n),
            "income": rng.uniform(5, 200, n),
            "education": rng.integers(0, 2, n).astype(bool),
            "work": rng.integers(0, 2, n).astype(bool),
            "car": rng.integers(0, 2, n).astype(bool),
        }
    )
    logit = -1.5 - 0.01 * df["income"] - 0.5 * df["work"]
    df[TARGET] = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


@pytest.fixture
def model_path(tmp_path, synthetic_data, monkeypatch):
    pipe = build_pipeline().fit(synthetic_data[FEATURES], synthetic_data[TARGET])
    path = tmp_path / "model.pkl"
    joblib.dump({"pipeline": pipe, "threshold": 0.5, "features": FEATURES}, path)
    monkeypatch.setenv("TARGET_PATH", str(path))
    return path
