"""
Unit tests for CineAnalytics movie review sentiment predictor
"""
import os
import unittest
import joblib
from model_trainer import analyze_review

class TestSentimentModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model.joblib")
        cls.package = joblib.load(model_path)

    def test_positive_review(self):
        text = "This movie is an incredible masterpiece with brilliant acting, stunning cinematography and deep emotion."
        res = analyze_review(text, self.package)
        self.assertTrue(res['is_positive'])
        self.assertGreaterEqual(res['score_10'], 7.0)
        self.assertIn("masterpiece", res['top_positive_words'] + [w for w in ["brilliant", "incredible", "stunning"]])

    def test_negative_review(self):
        text = "Terrible script, horrible acting, totally boring and a complete waste of time."
        res = analyze_review(text, self.package)
        self.assertFalse(res['is_positive'])
        self.assertLessEqual(res['score_10'], 4.0)

    def test_empty_review(self):
        res = analyze_review("", self.package)
        self.assertIsNone(res)

if __name__ == "__main__":
    unittest.main()
