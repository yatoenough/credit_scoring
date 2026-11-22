from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

from credit_scoring.data import load_sql
from credit_scoring.pipeline import FEATURES, TARGET, build_pipeline


def prepare_data(data):
    X = data[FEATURES]
    y = data[TARGET]

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train_model(X_train, y_train):
    pipe = build_pipeline().fit(X_train, y_train)
    return pipe


def get_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return {"precision": precision, "recall": recall}


def main():
    data = load_sql("sqlite:///data/scoring.db")
    X_train, X_test, y_train, y_test = prepare_data(data)

    pipe = train_model(X_train, y_train)
    threshold = 0.5
    y_pred = (pipe.predict_proba(X_test)[:, 1] >= threshold).astype(int)

    metrics = get_metrics(y_test, y_pred)

    print(f"Declined: {np.mean(y_pred) * 100:.0f}%")
    print(f"Precision: {metrics['precision'] * 100:.0f}%")
    print(f"Recall: {metrics['recall'] * 100:.0f}%")

    model_path = "models/model.pkl"
    Path("models").mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {"pipeline": pipe, "threshold": threshold, "features": FEATURES}, model_path
    )

    print(f"Saved model to {model_path}")
