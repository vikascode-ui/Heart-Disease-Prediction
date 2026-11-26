"""Train, compare and evaluate Heart Disease classification models."""
from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset" / "heart.csv"
OUTPUT = ROOT / "outputs"
MODEL_PATH = ROOT / "model.pkl"
TARGET = "target"
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
FEATURES = NUMERIC + CATEGORICAL


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Dataset missing. Run: python download_dataset.py")
    df = pd.read_csv(DATA_PATH, na_values=["?", "", "NA", "nan"])
    df.columns = [c.strip().lower() for c in df.columns]
    if TARGET not in df.columns:
        # Support the common Kaggle UCI file whose diagnosis column is named 'num'
        for possible in ["num", "condition", "diagnosis", "heart_disease"]:
            if possible in df.columns:
                df = df.rename(columns={possible: TARGET})
                break
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}. See README for format.")
    df = df[FEATURES + [TARGET]].copy()
    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicate rows")
    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[TARGET])
    # UCI diagnosis: 0 = no disease; 1..4 = disease present
    df[TARGET] = (df[TARGET] > 0).astype(int)
    return df


def build_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("categorical", categorical_pipe, CATEGORICAL),
    ])


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    OUTPUT.mkdir(exist_ok=True)
    df = load_data()
    print(f"Dataset shape after cleaning: {df.shape}")
    print("Class counts:\n", df[TARGET].value_counts().rename(index={0: "No disease", 1: "Disease"}))

    # EDA: target distribution and numeric correlation (raw data only for exploration)
    plt.figure(figsize=(6, 4))
    df[TARGET].value_counts().sort_index().plot(kind="bar")
    plt.xticks([0, 1], ["No disease", "Disease"], rotation=0)
    plt.title("Target Class Distribution")
    plt.ylabel("Number of records")
    plt.tight_layout()
    plt.savefig(OUTPUT / "target_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 7))
    corr = df[NUMERIC + CATEGORICAL + [TARGET]].corr(numeric_only=True)
    plt.imshow(corr, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT / "correlation_matrix.png", dpi=160)
    plt.close()

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                                class_weight="balanced", random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2,
                                            class_weight="balanced", random_state=42),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    comparison = []
    fitted = {}
    for name, estimator in models.items():
        pipe = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        comparison.append({"model": name, "cv_f1_mean": float(scores.mean()),
                           "cv_f1_std": float(scores.std())})
        print(f"{name}: CV F1={scores.mean():.3f} ± {scores.std():.3f}")

    comparison_df = pd.DataFrame(comparison).sort_values("cv_f1_mean", ascending=False)
    comparison_df.to_csv(OUTPUT / "model_comparison.csv", index=False)
    best_name = comparison_df.iloc[0]["model"]
    best_model = fitted[best_name]
    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test)[:, 1]
    metrics = {
        "selected_model": best_name,
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_precision": float(precision_score(y_test, predictions, zero_division=0)),
        "test_recall": float(recall_score(y_test, predictions, zero_division=0)),
        "test_f1": float(f1_score(y_test, predictions, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "random_state": 42,
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUTPUT / "classification_report.txt").write_text(
        classification_report(y_test, predictions, target_names=["No disease", "Disease"], zero_division=0),
        encoding="utf-8")
    print("\nSelected:", best_name)
    print(json.dumps(metrics, indent=2))

    ConfusionMatrixDisplay.from_predictions(y_test, predictions,
        display_labels=["No disease", "Disease"], cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - Test Set")
    plt.tight_layout()
    plt.savefig(OUTPUT / "confusion_matrix.png", dpi=160)
    plt.close()
    RocCurveDisplay.from_predictions(y_test, probabilities)
    plt.title("ROC Curve - Test Set")
    plt.tight_layout()
    plt.savefig(OUTPUT / "roc_curve.png", dpi=160)
    plt.close()

    # Permutation importance on the untouched test set; useful for project discussion.
    perm = permutation_importance(best_model, X_test, y_test, n_repeats=10,
                                 random_state=42, scoring="f1", n_jobs=-1)
    importance = pd.DataFrame({"feature": FEATURES, "importance_mean": perm.importances_mean,
                              "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False)
    importance.to_csv(OUTPUT / "feature_importance.csv", index=False)
    plt.figure(figsize=(8, 5))
    top = importance.head(10).sort_values("importance_mean")
    plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    plt.title("Permutation Feature Importance (Test Set)")
    plt.xlabel("Decrease in F1 after permutation")
    plt.tight_layout()
    plt.savefig(OUTPUT / "feature_importance.png", dpi=160)
    plt.close()

    joblib.dump({"model": best_model, "features": FEATURES, "model_name": best_name}, MODEL_PATH)
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved outputs: {OUTPUT}")

if __name__ == "__main__":
    main()
