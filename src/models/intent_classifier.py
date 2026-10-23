"""
High-Level Intent Classifier with Confidence Calibration, Fallback Routing, and Explainability.
"""
from typing import Dict, Any, List, Optional
import numpy as np

from src import config
from src.data.preprocessor import TextPreprocessor
from src.models.baseline import load_model, train_baseline_model, save_model


class CustomerIntentClassifier:
    """
    Production-ready Intent Classification Engine.
    Handles text normalization, inference, confidence thresholding, and explanations.
    """

    def __init__(
        self,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
        pipeline=None,
        label_encoder=None
    ):
        self.confidence_threshold = confidence_threshold
        self.preprocessor = TextPreprocessor()

        if pipeline is not None and label_encoder is not None:
            self.pipeline = pipeline
            self.label_encoder = label_encoder
        else:
            try:
                self.pipeline, self.label_encoder = load_model()
            except FileNotFoundError:
                self.pipeline = None
                self.label_encoder = None

    def fit(self, queries: List[str], labels: List[str], save: bool = True):
        """Trains the underlying classifier on raw queries and labels."""
        cleaned_queries = self.preprocessor.transform(queries)
        self.pipeline, self.label_encoder = train_baseline_model(cleaned_queries, labels)
        if save:
            save_model(self.pipeline, self.label_encoder)

    def predict(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Classifies a single incoming query with confidence score,
        fallback check, and ranked alternate intents.
        """
        if self.pipeline is None or self.label_encoder is None:
            raise RuntimeError("Model is not loaded or trained yet.")

        cleaned = self.preprocessor.clean_text(query)

        # Edge case: empty query
        if not cleaned:
            return {
                "query": query,
                "cleaned_query": cleaned,
                "intent": config.FALLBACK_INTENT,
                "confidence": 0.0,
                "requires_human_escalation": True,
                "top_intents": [],
                "reason": "Empty or non-text query"
            }

        # Predict probabilities
        probabilities = self.pipeline.predict_proba([cleaned])[0]
        top_indices = np.argsort(probabilities)[::-1][:top_k]

        ranked_intents = [
            {
                "intent": self.label_encoder.inverse_transform([idx])[0],
                "probability": round(float(probabilities[idx]), 4)
            }
            for idx in top_indices
        ]

        best_intent = ranked_intents[0]["intent"]
        best_confidence = ranked_intents[0]["probability"]

        # Confidence Thresholding / Out-of-Scope Fallback
        requires_escalation = best_confidence < self.confidence_threshold
        final_intent = best_intent if not requires_escalation else config.FALLBACK_INTENT

        # Explainability: Extract key n-grams
        explanation = self._explain_prediction(cleaned, top_indices[0])

        return {
            "query": query,
            "cleaned_query": cleaned,
            "intent": final_intent,
            "raw_intent": best_intent,
            "confidence": best_confidence,
            "requires_human_escalation": requires_escalation,
            "top_intents": ranked_intents,
            "explanation": explanation
        }

    def _explain_prediction(self, cleaned_query: str, class_idx: int) -> List[str]:
        """
        Identifies the strongest vocabulary tokens in the query contributing to the predicted intent.
        """
        try:
            vectorizer = self.pipeline.named_steps["tfidf"]
            classifier = self.pipeline.named_steps["clf"]

            feature_names = np.array(vectorizer.get_feature_names_out())
            tfidf_vec = vectorizer.transform([cleaned_query]).toarray()[0]

            active_indices = np.where(tfidf_vec > 0)[0]
            if len(active_indices) == 0:
                return []

            # Feature weights for the predicted class
            weights = classifier.coef_[class_idx][active_indices] * tfidf_vec[active_indices]
            top_active = active_indices[np.argsort(weights)[::-1][:3]]

            return [feature_names[idx] for idx in top_active if weights[np.where(active_indices == idx)[0][0]] > 0]
        except Exception:
            return []

    def predict_batch(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Batch prediction utility."""
        return [self.predict(q) for q in queries]
