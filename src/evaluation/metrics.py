"""
Evaluation Metrics, Confusion Matrix Plotting, and Error Analysis.
"""
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from src import config


def evaluate_predictions(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str] = None
) -> Dict[str, Any]:
    """
    Computes overall and per-class classification metrics.
    """
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report_str = classification_report(
        y_true, y_pred, labels=labels, zero_division=0
    )

    return {
        "accuracy": float(acc),
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(prec_weighted),
        "weighted_recall": float(rec_weighted),
        "weighted_f1": float(f1_weighted),
        "classification_report": report_str
    }


def plot_and_save_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
    output_filename: str = "confusion_matrix.png"
) -> str:
    """
    Generates and saves a publication-quality Confusion Matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        square=True
    )
    plt.title("Normalized Confusion Matrix - Customer Intent Recognition", fontsize=14, pad=15)
    plt.xlabel("Predicted Intent", fontsize=12)
    plt.ylabel("True Intent", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    output_path = config.REPORTS_DIR / output_filename
    plt.savefig(output_path, dpi=300)
    plt.close()
    return str(output_path)


def generate_error_analysis(
    queries: List[str],
    y_true: List[str],
    predictions: List[Dict[str, Any]],
    output_filename: str = "error_analysis.csv"
) -> pd.DataFrame:
    """
    Identifies misclassifications and saves detailed diagnostic table.
    """
    errors = []
    for q, true_intent, pred in zip(queries, y_true, predictions):
        predicted_intent = pred["raw_intent"]
        confidence = pred["confidence"]
        if true_intent != predicted_intent:
            errors.append({
                "query": q,
                "cleaned_query": pred["cleaned_query"],
                "true_intent": true_intent,
                "predicted_intent": predicted_intent,
                "confidence": confidence,
                "top_runner_up": pred["top_intents"][1]["intent"] if len(pred["top_intents"]) > 1 else None,
                "key_features": ", ".join(pred.get("explanation", []))
            })

    error_df = pd.DataFrame(errors)
    output_path = config.REPORTS_DIR / output_filename
    error_df.to_csv(output_path, index=False)
    return error_df
