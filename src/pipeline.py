import json
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config


def load_and_clean(data_path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(data_path)
    raw_rows = len(df)

    df = df.dropna(subset=["video_id", "views", "likes", "dislikes", "comment_count", "publish_time"])
    df = df[df["views"] > 0].copy()

    peak_views = df.groupby("video_id")["views"].max().rename("peak_views")
    df = df.sort_values("trending_date").drop_duplicates(subset=["video_id"], keep="first")
    df = df.merge(peak_views, on="video_id", how="left")

    df["title"] = df["title"].fillna("")
    df["tags"] = df["tags"].fillna("")
    df["description"] = df["description"].fillna("")

    stats = {
        "raw_rows": int(raw_rows),
        "cleaned_unique_videos": int(len(df)),
    }
    return df, stats


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, float]:
    threshold = float(df["peak_views"].quantile(config.AMPLIFIED_QUANTILE))
    y = (df["peak_views"] >= threshold).astype(int)

    pub_dt = pd.to_datetime(df["publish_time"], errors="coerce", utc=True)

    X = pd.DataFrame(index=df.index)
    X["like_rate"] = df["likes"] / df["views"]
    X["dislike_rate"] = df["dislikes"] / df["views"]
    X["comment_rate"] = df["comment_count"] / df["views"]
    X["like_to_dislike"] = df["likes"] / (df["dislikes"] + 1.0)
    X["title_len"] = df["title"].str.len()
    X["title_caps_ratio"] = df["title"].apply(
        lambda s: sum(1 for c in s if c.isupper()) / max(len(s), 1)
    )
    X["tag_count"] = df["tags"].apply(
        lambda s: 0 if s == "[none]" or s == "" else len(s.split("|"))
    )
    X["desc_len"] = df["description"].str.len()
    X["publish_hour"] = pub_dt.dt.hour.fillna(12).astype(int)
    X["publish_dayofweek"] = pub_dt.dt.dayofweek.fillna(0).astype(int)
    X["comments_disabled"] = df["comments_disabled"].astype(int)
    X["ratings_disabled"] = df["ratings_disabled"].astype(int)
    X["category_id"] = df["category_id"].astype(str)

    return X, y, threshold


def run_baseline(data_path: Path, results_dir: Path, mode: str) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    df, data_stats = load_and_clean(data_path)
    X, y, threshold = engineer_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=y,
    )

    num_cols = [c for c in X.columns if c != "category_id"]
    cat_cols = ["category_id"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", LogisticRegression(**config.LOGREG_PARAMS)),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "mode": mode,
        "dataset": str(data_path),
        "raw_rows": data_stats["raw_rows"],
        "cleaned_unique_videos": data_stats["cleaned_unique_videos"],
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "amplified_view_threshold": round(threshold, 2),
        "positive_class_rate": round(float(y.mean()), 4),
        "model": "LogisticRegression (baseline)",
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    output_file = results_dir / "baseline_metrics.json"
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n=== Baseline Results ({mode}) ===")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"{k}: {v}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Standard (0)", "Amplified (1)"]))
    print(f"Saved metrics to {output_file}")

    return metrics
