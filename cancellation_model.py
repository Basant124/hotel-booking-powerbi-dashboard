"""
Hotel booking cancellation model
--------------------------------
Predicts `is_canceled` and compares three models:
Logistic Regression, Random Forest and XGBoost.

Usage (from the repository root):
    python python/cancellation_model.py
    python python/cancellation_model.py --data data/hotel_bookings.csv --out outputs

Outputs (written to --out):
    model_comparison.csv                 metrics of the three models on the held-out test set
    feature_importance.csv               feature importance of the best model
    hotel_bookings_with_predictions.csv  the cleaned dataset + cancellation_risk_score + risk_segment
    best_model_summary.txt               best model name, AUC and number of high-risk bookings
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

RANDOM_STATE = 42
TARGET = "is_canceled"
# These two columns reveal the outcome (reservation_status == "Canceled"), so they must not be used as features.
LEAKAGE_COLS = ["reservation_status", "reservation_status_date"]
CATEGORICAL_COLS = [
    "hotel", "arrival_date_month", "meal", "country", "market_segment", "distribution_channel",
    "reserved_room_type", "assigned_room_type", "deposit_type", "customer_type",
]


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    df["children"] = df["children"].fillna(0)
    df["country"] = df["country"].fillna("Unknown")
    df["agent"] = df["agent"].fillna(0)
    df["company"] = df["company"].fillna(0)
    # bookings with zero guests are data-quality errors
    df = df[(df["adults"] + df["children"] + df["babies"]) > 0].copy()
    print(f"After removing zero-guest rows: {len(df):,}")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["is_family"] = ((df["children"] > 0) | (df["babies"] > 0)).astype(int)
    df["room_mismatch"] = (df["reserved_room_type"] != df["assigned_room_type"]).astype(int)
    return df


def main(data_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = add_features(load_and_clean(data_path))

    X = df.drop(columns=LEAKAGE_COLS + [TARGET])
    y = df[TARGET]
    for col in CATEGORICAL_COLS:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,} | Cancellation rate: {y.mean():.1%}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=14, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }

    rows, fitted = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "ROC_AUC": roc_auc_score(y_test, proba),
        })
        fitted[name] = model
        print(f"{name:20s} AUC = {rows[-1]['ROC_AUC']:.4f}")

    results = pd.DataFrame(rows).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
    results.to_csv(out_dir / "model_comparison.csv", index=False)
    print(results.round(4).to_string(index=False))

    best_name = results.loc[0, "Model"]
    best = fitted[best_name]
    if hasattr(best, "feature_importances_"):
        imp = best.feature_importances_
    else:
        imp = np.abs(best.coef_[0])
    importance = pd.DataFrame({"Feature": X.columns, "Importance": imp}).sort_values("Importance", ascending=False)
    importance.to_csv(out_dir / "feature_importance.csv", index=False)

    # Scores for every booking. NOTE: rows used for training are scored too (in-sample),
    # so the held-out metrics above are the honest estimate of model quality.
    score = best.predict_proba(X)[:, 1]
    out = df.copy()
    out["cancellation_risk_score"] = score
    out["risk_segment"] = pd.cut(score, bins=[-0.01, 0.33, 0.66, 1.0], labels=["Low", "Medium", "High"])
    out.to_csv(out_dir / "hotel_bookings_with_predictions.csv", index=False)

    (out_dir / "best_model_summary.txt").write_text(
        f"Best Model Name: {best_name}\n"
        f"Best Model AUC: {results.loc[0, 'ROC_AUC']:.3f}\n"
        f"High-risk bookings (risk >= 0.66): {(score >= 0.66).sum():,}\n"
    )
    print(f"\nBest model: {best_name}. Files saved to {out_dir.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/hotel_bookings.csv"))
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    a = ap.parse_args()
    main(a.data, a.out)
