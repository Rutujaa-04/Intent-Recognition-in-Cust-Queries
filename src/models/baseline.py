"""
TF-IDF + Calibrated Linear Classifier for Fast, Interpretable Intent Recognition.
"""
from typing import Dict, Any, Tuple
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src import config


def build_pipeline() -> Pipeline:
    """
    Constructs an NLP pipeline:
    1. TfidfVectorizer with unigrams + bigrams and sublinear TF scaling
    2. LogisticRegression with L2 regularization
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        max_features=10000,
        strip_accents="unicode"
    )

    classifier = LogisticRegression(
        C=2.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", classifier)
    ])
    return pipeline


def train_baseline_model(
    train_texts: list,
    train_labels: list
) -> Tuple[Pipeline, LabelEncoder]:
    """
    Fits label encoder and TF-IDF classification pipeline.
    """
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)

    pipeline = build_pipeline()
    pipeline.fit(train_texts, y_train)

    return pipeline, label_encoder


def save_model(pipeline: Pipeline, label_encoder: LabelEncoder) -> None:
    """
    Saves trained pipeline and label encoder to disk.
    """
    joblib.dump(pipeline, config.BASELINE_MODEL_PATH)
    joblib.dump(label_encoder, config.LABEL_ENCODER_PATH)
    print(f"Model saved successfully to {config.BASELINE_MODEL_PATH}")
    print(f"Label encoder saved to {config.LABEL_ENCODER_PATH}")


def load_model() -> Tuple[Pipeline, LabelEncoder]:
    """
    Loads saved model and label encoder from disk.
    """
    if not config.BASELINE_MODEL_PATH.exists() or not config.LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError(
            f"Saved model not found at {config.BASELINE_MODEL_PATH}. Run training first."
        )
    pipeline = joblib.load(config.BASELINE_MODEL_PATH)
    label_encoder = joblib.load(config.LABEL_ENCODER_PATH)
    return pipeline, label_encoder
