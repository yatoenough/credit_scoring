import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

from utils.load import load_sql


def prepare_data(data):
    X = data.drop(columns=["default", "index"]).values
    y = data["default"].values

    return train_test_split(X, y, test_size=0.2)

def train_model(X_train, y_train):
    model = LogisticRegression(class_weight="balanced")
    model.fit(X_train, y_train)
    return model

def get_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return {"precision": precision, "recall": recall}

def main():
    data = load_sql("sqlite:///data/scoring.db")
    X_train, X_test, y_train, y_test = prepare_data(data)

    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = get_metrics(y_test, y_pred)

    print(f"Declined: {np.mean(y_pred) * 100:.0f}%")
    print(f"Precision: {metrics["precision"] * 100:.0f}%")
    print(f"Recall: {metrics["recall"] * 100:.0f}%")

    model_path =  "models/model.pkl"

    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")
