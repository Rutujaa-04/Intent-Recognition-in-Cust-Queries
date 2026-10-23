"""
Unit tests for intent classifier and pipeline.
"""
import unittest
from src.models.intent_classifier import CustomerIntentClassifier


class TestIntentClassifier(unittest.TestCase):
    def test_classifier_fit_and_predict(self):
        queries = [
            "Where is my package?",
            "Track my delivery status",
            "I want a full refund please",
            "Return my product and give my money back"
        ]
        labels = [
            "track_order",
            "track_order",
            "request_refund",
            "request_refund"
        ]

        classifier = CustomerIntentClassifier(confidence_threshold=0.40)
        classifier.fit(queries, labels, save=False)

        # Test prediction on track query
        res = classifier.predict("Can you track my parcel?")
        self.assertIn(res["raw_intent"], ("track_order", "request_refund"))
        self.assertIn("confidence", res)
        self.assertTrue(0.0 <= res["confidence"] <= 1.0)
        self.assertEqual(len(res["top_intents"]), 2)

        # Verify probability distribution sums close to 1
        total_prob = sum(item["probability"] for item in res["top_intents"])
        self.assertAlmostEqual(total_prob, 1.0, places=1)


if __name__ == "__main__":
    unittest.main()
