import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

FEATURES = ["age", "income", "education", "work", "car"]
TARGET = "default"


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        (
                            "log_income",
                            make_pipeline(
                                FunctionTransformer(np.log10), StandardScaler()
                            ),
                            ["income"],
                        ),
                        ("age", StandardScaler(), ["age"]),
                        ("binary", "passthrough", ["education", "work", "car"]),
                    ],
                ),
            ),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]
    )
