from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
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
RESULTS_PATH = PROJECT_DIR / "model_comparison_random_search.csv"
THRESHOLD_GRID = np.linspace(0.10, 0.90, 161)
RECALL_FLOOR = 0.75
HORIZONS = [1, 2, 3]
TRAIN_SPLIT = 0.8
NON_FEATURE_COLS = [
    "No",
    "year",
    "month",
    "day",
    "hour",
    "wd",
    "datetime",
    "spike_t+1",
    "spike_t+2",
    "spike_t+3",
]


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def prep_xy(df: pd.DataFrame, target_col: str):
    feature_cols = get_feature_cols(df)
    X = df[feature_cols].copy()
    y = df[target_col].astype(int)
    return X, y


def prep_xy_ohe(train_df: pd.DataFrame, val_df: pd.DataFrame, target_col: str):
    train_features = train_df[[c for c in train_df.columns if c not in NON_FEATURE_COLS and c != "station"]].copy()
    val_features = val_df[[c for c in val_df.columns if c not in NON_FEATURE_COLS and c != "station"]].copy()

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_station = enc.fit_transform(train_df[["station"]])
    val_station = enc.transform(val_df[["station"]])

    train_station_df = pd.DataFrame(
        train_station,
        columns=[f"station_{v}" for v in enc.categories_[0]],
        index=train_df.index,
    )
    val_station_df = pd.DataFrame(
        val_station,
        columns=[f"station_{v}" for v in enc.categories_[0]],
        index=val_df.index,
    )

    X_train = pd.concat([train_features.reset_index(drop=True), train_station_df.reset_index(drop=True)], axis=1)
    X_val = pd.concat([val_features.reset_index(drop=True), val_station_df.reset_index(drop=True)], axis=1)

    y_train = train_df[target_col].astype(int).reset_index(drop=True)
    y_val = val_df[target_col].astype(int).reset_index(drop=True)
    return X_train, y_train, X_val, y_val


def get_best_threshold(y_true, y_proba, thresholds=None):
    if thresholds is None:
        thresholds = THRESHOLD_GRID

    best_threshold = 0.5
    best_f1 = -1.0
    best_recall = -1.0
    best_precision = -1.0
    best_score = -1.0

    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)

        if recall < RECALL_FLOOR:
            continue

        score = (0.7 * f1) + (0.3 * recall)
        if score > best_score or (np.isclose(score, best_score) and f1 > best_f1):
            best_threshold = float(thr)
            best_f1 = float(f1)
            best_recall = float(recall)
            best_precision = float(precision)
            best_score = float(score)

    if best_score < 0:
        for thr in thresholds:
            y_pred = (y_proba >= thr).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            precision = precision_score(y_true, y_pred, zero_division=0)
            if (f1 > best_f1) or (np.isclose(f1, best_f1) and recall > best_recall):
                best_threshold = float(thr)
                best_f1 = float(f1)
                best_recall = float(recall)
                best_precision = float(precision)

    return best_threshold, best_f1, best_recall, best_precision


def lightgbm_model_for_horizon(horizon: int):
    target_col = f"spike_t+{horizon}"
    # tuned version from the current pipeline
    scale_pos_weight = None
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1500,
        learning_rate=0.025,
        num_leaves=80,
        min_child_samples=40,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=1.5,
        max_depth=-1,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
        feature_fraction_bynode=0.8,
    )


def xgb_model_for_horizon(horizon: int):
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )


def catboost_model_for_horizon(horizon: int):
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=False,
        random_seed=42,
        auto_class_weights="Balanced",
        thread_count=-1,
        allow_writing_files=False,
    )


def tune_and_score(model_name: str, model, param_dist: dict, X_train, y_train, X_val, y_val):
    cv = TimeSeriesSplit(n_splits=3)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=6,
        scoring="average_precision",
        cv=cv,
        random_state=42,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    best_model.fit(X_train, y_train)

    y_proba = best_model.predict_proba(X_val)[:, 1]
    best_thr, best_f1, best_recall, best_precision = get_best_threshold(y_val, y_proba)
    y_pred = (y_proba >= best_thr).astype(int)

    metrics = {
        "model": model_name,
        "best_params": search.best_params_,
        "best_cv_ap": search.best_score_,
        "best_threshold": best_thr,
        "valid_f1": best_f1,
        "valid_recall": best_recall,
        "valid_precision": best_precision,
        "f1": f1_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_val, y_proba),
        "pr_auc": average_precision_score(y_val, y_proba),
    }

    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    metrics.update({"tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return metrics


def tune_xgboost(X_train, y_train, X_val, y_val):
    model = xgb_model_for_horizon(1)
    param_dist = {
        "n_estimators": [400, 600, 900, 1200],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.08],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "min_child_weight": [1, 3, 5, 10],
        "gamma": [0, 0.1, 0.5],
    }
    return tune_and_score("XGBoost", model, param_dist, X_train, y_train, X_val, y_val)


def tune_catboost(X_train, y_train, X_val, y_val):
    model = catboost_model_for_horizon(1)
    param_dist = {
        "iterations": [400, 600, 900, 1200],
        "depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.03, 0.05, 0.08],
        "l2_leaf_reg": [1, 3, 5, 9],
        "subsample": [0.7, 0.85, 1.0],
        "min_data_in_leaf": [20, 50, 100],
        "colsample_bylevel": [0.7, 0.85, 1.0],
    }
    return tune_and_score("CatBoost", model, param_dist, X_train, y_train, X_val, y_val)


def train_lightgbm(X_train, y_train, X_val, y_val):
    model = lightgbm_model_for_horizon(1)
    model.fit(
        X_train,
        y_train,
        categorical_feature=["station"],
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(60, verbose=False)],
    )
    y_proba = model.predict_proba(X_val)[:, 1]
    best_thr, best_f1, best_recall, best_precision = get_best_threshold(y_val, y_proba)
    y_pred = (y_proba >= best_thr).astype(int)

    metrics = {
        "model": "LightGBM",
        "best_params": model.get_params(),
        "best_cv_ap": None,
        "best_threshold": best_thr,
        "valid_f1": best_f1,
        "valid_recall": best_recall,
        "valid_precision": best_precision,
        "f1": f1_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_val, y_proba),
        "pr_auc": average_precision_score(y_val, y_proba),
    }
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    metrics.update({"tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return metrics


def evaluate_horizon(horizon: int):
    df = pd.read_csv(TRAIN_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["station", "datetime"]).copy()

    split = int(len(df) * TRAIN_SPLIT)
    train_df = df.iloc[:split].copy()
    val_df = df.iloc[split:].copy()

    train_df_lgb = train_df.copy()
    val_df_lgb = val_df.copy()
    train_df_lgb["station"] = train_df_lgb["station"].astype("category")
    val_df_lgb["station"] = val_df_lgb["station"].astype("category")

    target_col = f"spike_t+{horizon}"
    X_train_lgb, y_train_lgb = prep_xy(train_df_lgb, target_col)
    X_val_lgb, y_val_lgb = prep_xy(val_df_lgb, target_col)

    X_train_xgb, y_train_xgb, X_val_xgb, y_val_xgb = prep_xy_ohe(train_df, val_df, target_col)
    X_train_cat, y_train_cat, X_val_cat, y_val_cat = prep_xy_ohe(train_df, val_df, target_col)

    rows = []
    rows.append(train_lightgbm(X_train_lgb, y_train_lgb, X_val_lgb, y_val_lgb))
    rows.append(tune_xgboost(X_train_xgb, y_train_xgb, X_val_xgb, y_val_xgb))
    rows.append(tune_catboost(X_train_cat, y_train_cat, X_val_cat, y_val_cat))

    result = pd.DataFrame(rows)
    result.insert(0, "horizon", horizon)
    return result


def main():
    results = []
    for horizon in HORIZONS:
        results.append(evaluate_horizon(horizon))

    combined = pd.concat(results, ignore_index=True)
    combined = combined[
        [
            "horizon",
            "model",
            "best_threshold",
            "valid_f1",
            "valid_recall",
            "valid_precision",
            "f1",
            "recall",
            "precision",
            "roc_auc",
            "pr_auc",
            "tp",
            "fp",
            "fn",
            "tn",
            "best_cv_ap",
        ]
    ]
    combined = combined.sort_values(["horizon", "valid_f1"], ascending=[True, False]).reset_index(drop=True)
    combined.to_csv(RESULTS_PATH, index=False)

    print(combined.round(4).to_string(index=False))
    print(f"\nSaved comparison results to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()