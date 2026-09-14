from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_PATH = PROJECT_DIR / "train_features_v3.csv"
TEST_PATH = PROJECT_DIR / "test_features_v3.csv"
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_PATH = PROJECT_DIR / "model_results_xgboost.csv"
THRESHOLD_GRID = np.linspace(0.10, 0.90, 161)
RECALL_FLOOR = 0.75
HORIZONS = [1, 2, 3]
TRAIN_SPLIT = 0.8
NON_FEATURE_COLS = [
    "No", "year", "month", "day", "hour", "wd", "datetime",
    "spike_t+1", "spike_t+2", "spike_t+3",
]

MODELS_DIR.mkdir(exist_ok=True)


def get_features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def encode_features(train_df: pd.DataFrame, other_df: pd.DataFrame, target_col: str):
    numeric_cols = [c for c in get_features(train_df) if c != "station"]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_station = encoder.fit_transform(train_df[["station"]])
    other_station = encoder.transform(other_df[["station"]])
    station_cols = [f"station_{value}" for value in encoder.categories_[0]]

    train_x = pd.concat(
        [train_df[numeric_cols].reset_index(drop=True), pd.DataFrame(train_station, columns=station_cols)], axis=1
    )
    other_x = pd.concat(
        [other_df[numeric_cols].reset_index(drop=True), pd.DataFrame(other_station, columns=station_cols)], axis=1
    )
    return train_x, train_df[target_col].astype(int).reset_index(drop=True), other_x, other_df[target_col].astype(int).reset_index(drop=True), list(encoder.categories_[0])


def get_best_threshold(y_true, y_proba):
    candidates = []
    for threshold in THRESHOLD_GRID:
        prediction = (y_proba >= threshold).astype(int)
        recall = recall_score(y_true, prediction, zero_division=0)
        f1 = f1_score(y_true, prediction, zero_division=0)
        precision = precision_score(y_true, prediction, zero_division=0)
        if recall >= RECALL_FLOOR:
            candidates.append(((0.7 * f1) + (0.3 * recall), f1, threshold, recall, precision))
    if not candidates:
        candidates = [
            (f1_score(y_true, (y_proba >= threshold).astype(int), zero_division=0),
             f1_score(y_true, (y_proba >= threshold).astype(int), zero_division=0), threshold,
             recall_score(y_true, (y_proba >= threshold).astype(int), zero_division=0),
             precision_score(y_true, (y_proba >= threshold).astype(int), zero_division=0))
            for threshold in THRESHOLD_GRID
        ]
    _, f1, threshold, recall, precision = max(candidates)
    return float(threshold), float(f1), float(recall), float(precision)


def model_search(X_train, y_train):
    model = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", tree_method="hist", random_state=42, n_jobs=-1
    )
    parameter_space = {
        "n_estimators": [400, 600, 900, 1200],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.08],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "min_child_weight": [1, 3, 5, 10],
        "gamma": [0, 0.1, 0.5],
    }
    search = RandomizedSearchCV(
        model, parameter_space, n_iter=6, scoring="average_precision",
        cv=TimeSeriesSplit(n_splits=3), random_state=42, n_jobs=1, refit=True
    )
    search.fit(X_train, y_train)
    return search


def train_horizon(train_df: pd.DataFrame, valid_df: pd.DataFrame, horizon: int):
    target_col = f"spike_t+{horizon}"
    X_train, y_train, X_valid, y_valid, station_categories = encode_features(train_df, valid_df, target_col)
    search = model_search(X_train, y_train)
    model = search.best_estimator_
    model.fit(X_train, y_train)

    valid_proba = model.predict_proba(X_valid)[:, 1]
    threshold, valid_f1, valid_recall, valid_precision = get_best_threshold(y_valid, valid_proba)
    valid_pred = (valid_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_valid, valid_pred).ravel()
    metrics = {
        "horizon": horizon, "model": "XGBoost", "best_threshold": threshold,
        "valid_f1": valid_f1, "valid_recall": valid_recall, "valid_precision": valid_precision,
        "f1": f1_score(y_valid, valid_pred), "recall": recall_score(y_valid, valid_pred),
        "precision": precision_score(y_valid, valid_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_valid, valid_proba), "pr_auc": average_precision_score(y_valid, valid_proba),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "best_cv_ap": search.best_score_,
        "best_params": str(search.best_params_),
    }
    joblib.dump(
        {"model": model, "threshold": threshold, "feature_columns": list(X_train.columns), "station_categories": station_categories},
        MODELS_DIR / f"xgb_spike_t{horizon}.joblib",
    )
    return metrics


def main():
    train = pd.read_csv(TRAIN_PATH)
    pd.read_csv(TEST_PATH)
    train["datetime"] = pd.to_datetime(train["datetime"])
    train = train.sort_values(["station", "datetime"]).reset_index(drop=True)
    split = int(len(train) * TRAIN_SPLIT)
    train_df, valid_df = train.iloc[:split].copy(), train.iloc[split:].copy()
    metrics = []
    for horizon in HORIZONS:
        print(f"Training XGBoost for t+{horizon}h ...")
        result = train_horizon(train_df, valid_df, horizon)
        metrics.append(result)
        print(f"  threshold={result['best_threshold']:.3f} F1={result['f1']:.4f} recall={result['recall']:.4f} precision={result['precision']:.4f} PR-AUC={result['pr_auc']:.4f}")
    results = pd.DataFrame(metrics)
    results.to_csv(RESULTS_PATH, index=False)
    print(f"Saved XGBoost metrics to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()