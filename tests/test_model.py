"""
Unit tests for CineAnalytics movie review sentiment predictor
"""
import os
import sys
import json
import unittest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from api.index import predict_sentiment_exact

class TestSentimentModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        weights_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_weights.json")
        with open(weights_path, 'r', encoding='utf-8') as f:
            cls.model_data = json.load(f)

    def test_positive_review(self):
        text = "This movie is an incredible masterpiece with brilliant acting, stunning cinematography and deep emotion."
        res = predict_sentiment_exact(text, self.model_data)
        self.assertTrue(res['is_positive'])
        self.assertGreaterEqual(res['score_10'], 7.0)
        self.assertIn(res['sentiment'], ["Ijobiy (Positive)"])

    def test_negative_review(self):
        text = "Terrible script, horrible acting, totally boring and a complete waste of time."
        res = predict_sentiment_exact(text, self.model_data)
        self.assertFalse(res['is_positive'])
        self.assertLessEqual(res['score_10'], 4.0)

    def test_empty_review(self):
        res = predict_sentiment_exact("", self.model_data)
        self.assertEqual(res['score_10'], 5.5)

if __name__ == "__main__":
    unittest.main()
