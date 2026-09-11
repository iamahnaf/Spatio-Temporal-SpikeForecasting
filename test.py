"""
05_modeling.py
Trains gradient boosting (LightGBM) classifiers to predict PM2.5 spikes
at t+1, t+2, t+3 hours ahead, handles the ~0.88% class imbalance with
scale_pos_weight and (optionally) SMOTE, evaluates with F1/Recall/ROC-AUC/
PR-AUC (NOT accuracy), and runs SHAP for explainability.

Run: python 05_modeling.py
Output: metrics printed + saved to model_results.csv, SHAP plots saved to
        ./shap_plots/, trained models saved to ./models/
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.metrics import (
    f1_score, recall_score, precision_score, roc_auc_score,
    average_precision_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "notebooks"
TRAIN_PATH = DATA_DIR / "train_features_v2.csv"
TEST_PATH = DATA_DIR / "test_features_v2.csv"
HORIZONS = [1, 2, 3]
USE_SMOTE = True          # set False to rely on scale_pos_weight only
MODELS_DIR = DATA_DIR / "models"
SHAP_DIR = DATA_DIR / "shap_plots"

# Columns that are identifiers, raw target leakage risks, or already
# re-encoded elsewhere (wd -> wind_u/v, year/month/day/hour -> cyclical +
# already used inside lag/rolling features) - excluded from the feature set.
NON_FEATURE_COLS = [
    "No", "year", "month", "day", "hour", "wd", "datetime",
    "spike_t+1", "spike_t+2", "spike_t+3",
]


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    required_cols = {"datetime", "station", *[f"spike_t+{h}" for h in HORIZONS]}
    missing_train = required_cols.difference(train.columns)
    missing_test = required_cols.difference(test.columns)
    if missing_train or missing_test:
        raise ValueError(
            "Feature files are missing required columns: "
            f"train={sorted(missing_train)}, test={sorted(missing_test)}"
        )
    train["datetime"] = pd.to_datetime(train["datetime"])
    test["datetime"] = pd.to_datetime(test["datetime"])
    return train, test


def get_feature_cols(df):
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def prep_xy(df, feature_cols, target_col, station_categories):
    X = df[feature_cols].copy()
    y = df[target_col].astype(int)
    if not y.isin([0, 1]).all():
        raise ValueError(f"{target_col} must contain only binary 0/1 labels")
    X["station"] = pd.Categorical(
        X["station"], categories=station_categories
    )
    return X, y


def train_one_horizon(train, test, horizon):
    target_col = f"spike_t+{horizon}"
    feature_cols = get_feature_cols(train)
    station_categories = sorted(
        set(train["station"].dropna()).union(test["station"].dropna())
    )

    X_train, y_train = prep_xy(
        train, feature_cols, target_col, station_categories
    )
    X_test, y_test = prep_xy(
        test, feature_cols, target_col, station_categories
    )
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(
            f"{target_col} must contain both classes in train and test sets"
        )

    print(f"\n{'='*60}\nHorizon t+{horizon}h | train={len(X_train)} test={len(X_test)} "
          f"| train spike rate={y_train.mean()*100:.3f}%")

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    if USE_SMOTE and y_train.value_counts().min() >= 2:
        # SMOTE can't handle the categorical 'station' column directly ->
        # one-hot it just for the resampling step, then rebuild the frame.
        X_train_ohe = pd.get_dummies(X_train, columns=["station"])
        minority_count = y_train.value_counts().min()
        sm = SMOTE(
            random_state=42,
            k_neighbors=min(5, minority_count - 1),
        )
        X_res, y_res = sm.fit_resample(X_train_ohe, y_train)
        # collapse one-hot station back to a single categorical column for LightGBM
        station_cols = [c for c in X_res.columns if c.startswith("station_")]
        X_res["station"] = X_res[station_cols].idxmax(axis=1).str.replace("station_", "", regex=False)
        X_res = X_res.drop(columns=station_cols)
        X_res["station"] = X_res["station"].astype("category")
        print(f"After SMOTE: {len(X_res)} rows, spike rate={y_res.mean()*100:.2f}%")
    else:
        X_res, y_res = X_train, y_train

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        scale_pos_weight=1.0 if USE_SMOTE else scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        X_res, y_res,
        categorical_feature=["station"],
        eval_set=[(X_test, y_test)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        "horizon": horizon,
        "f1": f1_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "pr_auc": average_precision_score(y_test, y_pred_proba),
    }
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    metrics.update({"tp": tp, "fp": fp, "fn": fn, "tn": tn})

    print(f"F1={metrics['f1']:.3f}  Recall={metrics['recall']:.3f}  "
          f"Precision={metrics['precision']:.3f}  ROC-AUC={metrics['roc_auc']:.3f}  "
          f"PR-AUC={metrics['pr_auc']:.3f}")
    print(f"Confusion matrix -> TP={tp} FP={fp} FN={fn} TN={tn}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / f"lgbm_spike_t{horizon}.joblib")

    return model, X_test, y_test, metrics


def run_shap(model, X_test, horizon, max_display=15, sample_size=2000):
    """SHAP summary + bar plot for this horizon's model."""
    X_sample = X_test.sample(min(sample_size, len(X_test)), random_state=42).copy()
    # Keep 'station' as the same pandas category dtype the model was trained
    # with - LightGBM's booster checks categorical_feature consistency
    # between train and predict data, so converting to codes here breaks it.

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):  # binary clf sometimes returns [neg, pos]
        shap_values = shap_values[1]

    plt.figure()
    shap.summary_plot(shap_values, X_sample, max_display=max_display, show=False)
    plt.title(f"SHAP summary - spike t+{horizon}h")
    plt.tight_layout()
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(SHAP_DIR / f"shap_summary_t{horizon}.png", dpi=120, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar",
                       max_display=max_display, show=False)
    plt.title(f"SHAP feature importance - spike t+{horizon}h")
    plt.tight_layout()
    plt.savefig(SHAP_DIR / f"shap_bar_t{horizon}.png", dpi=120, bbox_inches="tight")
    plt.close()

    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0), index=X_sample.columns
    ).sort_values(ascending=False)
    print(f"\nTop 10 SHAP features for t+{horizon}h:\n{mean_abs_shap.head(10)}")
    return mean_abs_shap


if __name__ == "__main__":
    train, test = load_data()

    all_metrics = []
    for h in HORIZONS:
        model, X_test, y_test, metrics = train_one_horizon(train, test, h)
        all_metrics.append(metrics)
        run_shap(model, X_test, h)

    results_df = pd.DataFrame(all_metrics)
    results_df.to_csv(DATA_DIR / "model_results.csv", index=False)
    print(f"\n{'='*60}\nAll results saved to model_results.csv:")
    print(results_df.to_string(index=False))