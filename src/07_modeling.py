"""
STEP 7 - MODELING STUNTING PREDICTION
Trains and evaluates multiple ML classifiers for stunting risk prediction.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

PROCESSED_DIR = Path("Data/processed")
MODEL_DIR = Path("models")
INPUT_FILE = PROCESSED_DIR / "dataset_final.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data():
    df = pd.read_csv(INPUT_FILE)
    X = df.drop(columns=["is_stunted"])
    y = df["is_stunted"]
    return X, y


def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            scale_pos_weight=None,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            use_label_encoder=False,
        ),
    }


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_stunted": precision_score(y_test, y_pred, pos_label=1),
        "recall_stunted": recall_score(y_test, y_pred, pos_label=1),
        "f1_stunted": f1_score(y_test, y_pred, pos_label=1),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def print_evaluation(name, metrics, y_test):
    print(f"\n{'='*60}")
    print(f"MODEL: {name}")
    print(f"{'='*60}")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Precision (stunted): {metrics['precision_stunted']:.4f}")
    print(f"Recall (stunted):    {metrics['recall_stunted']:.4f}")
    print(f"F1-score (stunted):  {metrics['f1_stunted']:.4f}")
    print(f"ROC-AUC:           {metrics['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, metrics["y_pred"])
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, metrics["y_pred"], target_names=["Normal", "Stunted"]))


def main():
    # 1. Load dataset
    X, y = load_data()
    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution:")
    print(f"  Normal (0): {(y == 0).sum()} ({(y == 0).mean() * 100:.1f}%)")
    print(f"  Stunted (1): {(y == 1).sum()} ({(y == 1).mean() * 100:.1f}%)")
    print(f"Feature columns: {X.columns.tolist()}")

    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # 3. Compute scale_pos_weight for XGBoost
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count

    models = get_models()
    models["XGBoost"].set_params(scale_pos_weight=scale_pos_weight)

    # 4. Train and evaluate
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        print_evaluation(name, metrics, y_test)
        results[name] = {
            "model": model,
            "accuracy": metrics["accuracy"],
            "precision_stunted": metrics["precision_stunted"],
            "recall_stunted": metrics["recall_stunted"],
            "f1_stunted": metrics["f1_stunted"],
            "roc_auc": metrics["roc_auc"],
        }

    # 5. Comparison table
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    comparison = pd.DataFrame([
        {
            "model_name": name,
            "accuracy": r["accuracy"],
            "precision_stunted": r["precision_stunted"],
            "recall_stunted": r["recall_stunted"],
            "f1_stunted": r["f1_stunted"],
            "roc_auc": r["roc_auc"],
        }
        for name, r in results.items()
    ])
    print(comparison.to_string(index=False))

    # 6. Select best model (primary: f1_stunted, secondary: roc_auc)
    best_name = max(results, key=lambda n: (results[n]["f1_stunted"], results[n]["roc_auc"]))
    best_model = results[best_name]["model"]
    best_metrics = results[best_name]

    print(f"\nBest model: {best_name}")
    print(f"  F1 (stunted): {best_metrics['f1_stunted']:.4f}")
    print(f"  Recall (stunted): {best_metrics['recall_stunted']:.4f}")
    print(f"  ROC-AUC: {best_metrics['roc_auc']:.4f}")

    # 7. Save best model and feature columns
    model_path = MODEL_DIR / "stunting_model.pkl"
    features_path = MODEL_DIR / "feature_columns.pkl"

    joblib.dump(best_model, model_path)
    joblib.dump(list(X.columns), features_path)

    print(f"\nModel saved: {model_path}")
    print(f"Features saved: {features_path}")

    # 8. Simulation: predict one sample
    sample = X_test.iloc[[0]]
    pred_class = best_model.predict(sample)[0]
    pred_proba = best_model.predict_proba(sample)[0]

    print(f"\n{'='*60}")
    print("PREDICTION SIMULATION")
    print(f"{'='*60}")
    print(f"Input features:")
    for col, val in sample.iloc[0].items():
        print(f"  {col}: {val}")
    print(f"\nPredicted class: {'Stunted' if pred_class == 1 else 'Normal'}")
    print(f"Probability Normal:  {pred_proba[0]:.4f}")
    print(f"Probability Stunted: {pred_proba[1]:.4f}")
    print(f"Actual label: {'Stunted' if y_test.iloc[0] == 1 else 'Normal'}")

    # 9. Save comparison table
    comparison.to_csv(PROCESSED_DIR / "07_model_comparison.csv", index=False)
    print(f"\nComparison saved: {PROCESSED_DIR / '07_model_comparison.csv'}")


if __name__ == "__main__":
    main()
