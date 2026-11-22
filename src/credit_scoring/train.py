import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)

from credit_scoring.data import load_sql
from credit_scoring.pipeline import FEATURES, TARGET, build_pipeline

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
CV_SCORING = ["roc_auc", "average_precision", "precision", "recall", "f1"]


def prepare_data(data):
    X = data[FEATURES]
    y = data[TARGET]

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train_model(X_train, y_train):
    pipe = build_pipeline().fit(X_train, y_train)
    return pipe


def build_dummy():
    return DummyClassifier(strategy="stratified", random_state=42)


def cross_validate_model(estimator, X, y):
    scores = cross_validate(estimator, X, y, cv=CV, scoring=CV_SCORING)
    return {
        name: {
            "mean": float(np.mean(scores[f"test_{name}"])),
            "std": float(np.std(scores[f"test_{name}"])),
        }
        for name in CV_SCORING
    }


def choose_threshold(y_true, proba):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # The last precision/recall point has no matching threshold.
    precision, recall = precision[:-1], recall[:-1]
    denom = precision + recall
    f1 = np.divide(
        2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0
    )
    return float(thresholds[np.argmax(f1)])


def get_metrics(y_true, proba, threshold):
    y_pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "approval_rate": float(1 - np.mean(y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
    }


def print_cv(cv_model, cv_dummy):
    print(f"{'5-fold CV':<20}{'model':>18}{'dummy':>18}")
    for name in CV_SCORING:
        m, d = cv_model[name], cv_dummy[name]
        print(
            f"{name:<20}"
            f"{m['mean']:>10.3f} ± {m['std']:.3f}"
            f"{d['mean']:>10.3f} ± {d['std']:.3f}"
        )


def print_test(test_model, test_dummy):
    print(f"{'Test set':<20}{'model':>18}{'dummy':>18}")
    for name in ["roc_auc", "pr_auc", "precision", "recall", "f1", "approval_rate"]:
        print(f"{name:<20}{test_model[name]:>18.3f}{test_dummy[name]:>18.3f}")
    (tn, fp), (fn, tp) = test_model["confusion_matrix"]
    print("\nConfusion matrix (rows = actual, cols = predicted):")
    print(f"{'':<12}{'no default':>12}{'default':>12}")
    print(f"{'no default':<12}{tn:>12}{fp:>12}")
    print(f"{'default':<12}{fn:>12}{tp:>12}")


def main():
    data = load_sql("sqlite:///data/scoring.db")
    X_train, X_test, y_train, y_test = prepare_data(data)

    cv_model = cross_validate_model(build_pipeline(), X_train, y_train)
    cv_dummy = cross_validate_model(build_dummy(), X_train, y_train)
    print_cv(cv_model, cv_dummy)

    oof_proba = cross_val_predict(
        build_pipeline(), X_train, y_train, cv=CV, method="predict_proba"
    )[:, 1]
    threshold = choose_threshold(y_train, oof_proba)
    print(f"\nThreshold (max F1 on out-of-fold predictions): {threshold:.3f}\n")

    pipe = train_model(X_train, y_train)
    dummy = build_dummy().fit(X_train, y_train)

    test_model = get_metrics(y_test, pipe.predict_proba(X_test)[:, 1], threshold)
    test_dummy = get_metrics(y_test, dummy.predict_proba(X_test)[:, 1], threshold)
    print_test(test_model, test_dummy)
    y_pred = (pipe.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    print("\n" + classification_report(y_test, y_pred, zero_division=0))

    metrics = {
        "threshold": threshold,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "base_rate": float(np.mean(y_train)),
        "cv": {"model": cv_model, "dummy": cv_dummy},
        "test": {"model": test_model, "dummy": test_dummy},
    }

    target_path = Path(os.environ.get("TARGET_PATH", "target/model.pkl"))
    target_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "pipeline": pipe,
            "threshold": threshold,
            "features": FEATURES,
            "metrics": metrics,
        },
        target_path,
    )
    metrics_path = Path(os.environ.get("METRICS_PATH", "metrics/metrics.json"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"Saved model to {target_path}")
    print(f"Saved metrics to {metrics_path}")
