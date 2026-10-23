"""
Unit tests for text preprocessing pipeline.
"""
import unittest
from src.data.preprocessor import TextPreprocessor


class TestTextPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = TextPreprocessor()

    def test_contraction_expansion(self):
        text = "I can't find my order and won't wait any longer"
        cleaned = self.preprocessor.clean_text(text)
        self.assertIn("cannot", cleaned)
        self.assertIn("will not", cleaned)

    def test_entity_normalization(self):
        text = "Please check order #123456 and refund $49.99 to test@example.com"
        cleaned = self.preprocessor.clean_text(text)
        self.assertIn("<ENTITY_ID>", cleaned)
        self.assertIn("<AMOUNT>", cleaned)
        self.assertIn("<EMAIL>", cleaned)
        self.assertIn("refund", cleaned)

    def test_empty_string_handling(self):
        self.assertEqual(self.preprocessor.clean_text(""), "")
        self.assertEqual(self.preprocessor.clean_text("   "), "")
        self.assertEqual(self.preprocessor.clean_text(None), "")


if __name__ == "__main__":
    unittest.main()
