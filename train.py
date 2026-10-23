"""
Main Model Training and Evaluation Pipeline.
Usage:
    python train.py
"""
import sys
from pathlib import Path
import pandas as pd

from src import config
from src.data.dataset_generator import load_data, prepare_and_save_data
from src.models.intent_classifier import CustomerIntentClassifier
from src.evaluation.metrics import (
    evaluate_predictions,
    plot_and_save_confusion_matrix,
    generate_error_analysis
)


def run_training_pipeline():
    print("=" * 70)
    print("     CUSTOMER INTENT RECOGNITION - TRAINING PIPELINE")
    print("=" * 70)

    # 1. Load Data
    print("\n[Step 1/4] Loading train, validation, and test datasets...")
    if not config.TRAIN_DATA_FILE.exists():
        print("Datasets not found. Generating fresh dataset splits...")
        train_df, val_df, test_df = prepare_and_save_data()
    else:
        train_df = load_data("train")
        val_df = load_data("val")
        test_df = load_data("test")

    print(f"Loaded: {len(train_df)} train samples, {len(val_df)} val samples, {len(test_df)} test samples.")
    print(f"Total Intents: {train_df['intent'].nunique()} categories")

    # 2. Train Model
    print("\n[Step 2/4] Training TF-IDF + Calibrated Linear Intent Classifier...")
    classifier = CustomerIntentClassifier(confidence_threshold=config.CONFIDENCE_THRESHOLD)
    classifier.fit(train_df["query"].tolist(), train_df["intent"].tolist(), save=True)
    print("Model training completed and serialized successfully.")

    # 3. Evaluate on Test Set
    print("\n[Step 3/4] Evaluating classifier on held-out test set...")
    test_queries = test_df["query"].tolist()
    test_labels = test_df["intent"].tolist()

    predictions = classifier.predict_batch(test_queries)
    y_pred_raw = [p["raw_intent"] for p in predictions]

    eval_results = evaluate_predictions(
        y_true=test_labels,
        y_pred=y_pred_raw,
        labels=config.INTENT_LABELS
    )

    print("\n" + "-" * 45)
    print(f"  Test Accuracy:    {eval_results['accuracy']:.4f} ({eval_results['accuracy']*100:.2f}%)")
    print(f"  Macro F1-Score:   {eval_results['macro_f1']:.4f}")
    print(f"  Weighted F1-Score:{eval_results['weighted_f1']:.4f}")
    print("-" * 45)

    print("\nDetailed Per-Class Classification Report:")
    print(eval_results["classification_report"])

    # 4. Diagnostics & Reporting
    print("[Step 4/4] Generating Confusion Matrix & Error Analysis...")
    cm_path = plot_and_save_confusion_matrix(
        y_true=test_labels,
        y_pred=y_pred_raw,
        labels=config.INTENT_LABELS,
        output_filename="confusion_matrix.png"
    )
    print(f"Confusion Matrix saved to: {cm_path}")

    error_df = generate_error_analysis(
        queries=test_queries,
        y_true=test_labels,
        predictions=predictions,
        output_filename="error_analysis.csv"
    )
    print(f"Misclassified samples: {len(error_df)} / {len(test_df)} ({len(error_df)/len(test_df)*100:.2f}%)")
    print(f"Error diagnostics saved to: {config.REPORTS_DIR / 'error_analysis.csv'}")

    print("\n" + "=" * 70)
    print("Training pipeline finished successfully!")
    print("To test the trained model on custom queries, run:")
    print("    python predict.py --text 'Where is my package?'")
    print("=" * 70)


if __name__ == "__main__":
    run_training_pipeline()
