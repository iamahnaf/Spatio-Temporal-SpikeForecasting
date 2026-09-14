from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_V2 = PROJECT_DIR / "train_features_v2.csv"
TEST_V2 = PROJECT_DIR / "test_features_v2.csv"
TRAIN_V3 = PROJECT_DIR / "train_features_v3.csv"
TEST_V3 = PROJECT_DIR / "test_features_v3.csv"
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_PATH = PROJECT_DIR / "model_results_v3.csv"
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

MODELS_DIR.mkdir(exist_ok=True)


def add_advanced_horizon_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["station", "datetime"]).copy()

    df["pm25_delta_1"] = df["PM2.5"] - df.groupby("station")["PM2.5"].shift(1)
    df["pm25_delta_3"] = df["PM2.5"] - df.groupby("station")["PM2.5"].shift(3)
    df["pm25_change_6"] = df["PM2.5"] - df.groupby("station")["PM2.5"].shift(6)
    df["pm25_roc_3"] = df["pm25_delta_3"] / 3.0

    if "pm25_roll_mean_6" in df.columns:
        df["pm25_trend_6"] = df["PM2.5"] - df["pm25_roll_mean_6"]

    if "idw_pm25_neighbors" in df.columns:
        df["pm25_vs_neighbor"] = df["PM2.5"] - df["idw_pm25_neighbors"]
        df["neighbor_pressure"] = df["idw_pm25_neighbors"] / (df["PM2.5"] + 1e-6)

    if "dayofweek" in df.columns:
        df["weekend_flag"] = df["dayofweek"].isin([5, 6]).astype(int)

    station_mean = df.groupby("station")["PM2.5"].transform(
        lambda s: s.rolling(24, min_periods=12).mean()
    )
    station_std = df.groupby("station")["PM2.5"].transform(
        lambda s: s.rolling(24, min_periods=12).std().fillna(0.0)
    )
    df["station_pm25_zscore_24"] = (df["PM2.5"] - station_mean) / station_std.replace(0, np.nan)
    df["station_pm25_zscore_24"] = df["station_pm25_zscore_24"].fillna(0.0)

    if "regional_elevated_frac" in df.columns:
        df["regional_gap"] = df["PM2.5"] - df["regional_elevated_frac"]
    else:
        df["regional_gap"] = 0.0

    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def get_feature_cols(df: pd.DataFrame):
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def prep_xy(df: pd.DataFrame, feature_cols, target_col):
    X = df[feature_cols].copy()
    X["station"] = X["station"].astype("category")
    y = df[target_col].astype(int)
    return X, y


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


def train_one_horizon(train_df, val_df, horizon):
    target_col = f"spike_t+{horizon}"
    feature_cols = get_feature_cols(train_df)
    X_train, y_train = prep_xy(train_df, feature_cols, target_col)
    X_val, y_val = prep_xy(val_df, feature_cols, target_col)

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = lgb.LGBMClassifier(
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
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
        feature_fraction_bynode=0.8,
    )
    model.fit(
        X_train,
        y_train,
        categorical_feature=["station"],
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(60, verbose=False)],
    )

    y_proba = model.predict_proba(X_val)[:, 1]
    best_threshold, best_f1, best_recall, best_precision = get_best_threshold(y_val, y_proba)
    y_pred = (y_proba >= best_threshold).astype(int)

    metrics = {
        "horizon": horizon,
        "best_threshold": best_threshold,
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
    joblib_path = MODELS_DIR / f"lgbm_spike_t{horizon}.joblib"
    import joblib

    joblib.dump(model, joblib_path)
    return metrics, model, y_proba


def main():
    if not TRAIN_V3.exists() or not TEST_V3.exists():
        train_v2 = pd.read_csv(TRAIN_V2)
        test_v2 = pd.read_csv(TEST_V2)
        train_v3 = add_advanced_horizon_features(train_v2)
        test_v3 = add_advanced_horizon_features(test_v2)
        train_v3.to_csv(TRAIN_V3, index=False)
        test_v3.to_csv(TEST_V3, index=False)
        print(f"Created enhanced files: {TRAIN_V3.name} and {TEST_V3.name}")

    train = pd.read_csv(TRAIN_V3)
    test = pd.read_csv(TEST_V3)
    train["datetime"] = pd.to_datetime(train["datetime"])
    test["datetime"] = pd.to_datetime(test["datetime"])

    split = int(len(train) * TRAIN_SPLIT)
    train_df = train.iloc[:split].copy()
    val_df = train.iloc[split:].copy()

    rows = []
    for h in HORIZONS:
        metrics, _, _ = train_one_horizon(train_df, val_df, h)
        rows.append(metrics)

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_PATH, index=False)
    print(results.round(4).to_string(index=False))
    print(f"\nSaved metrics: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
