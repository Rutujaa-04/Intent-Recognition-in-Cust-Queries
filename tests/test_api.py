"""
Integration tests for FastAPI REST API endpoints.
"""
import unittest
from fastapi.testclient import TestClient
from src.api.app import app


class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_root_endpoint(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "running")

    def test_health_endpoint(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])

    def test_intents_list(self):
        res = self.client.get("/intents")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("track_order", data["intents"])
        self.assertIn("request_refund", data["intents"])

    def test_predict_endpoint(self):
        payload = {"query": "Can you track my package please?", "top_k": 3}
        res = self.client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["intent"], "track_order")
        self.assertFalse(data["requires_human_escalation"])
        self.assertGreater(data["confidence"], 0.5)
        self.assertEqual(len(data["top_intents"]), 3)


if __name__ == "__main__":
    unittest.main()
