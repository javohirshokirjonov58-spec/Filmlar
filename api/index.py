import json
import os
import sys
import re
import math
import collections
from http.server import BaseHTTPRequestHandler

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

WEIGHTS_FILE = os.path.join(PARENT_DIR, "model_weights.json")
if not os.path.exists(WEIGHTS_FILE):
    WEIGHTS_FILE = os.path.join(CURRENT_DIR, "model_weights.json")

_model_data = None

def get_model_data():
    global _model_data
    if _model_data is None and os.path.exists(WEIGHTS_FILE):
        try:
            with open(WEIGHTS_FILE, 'r', encoding='utf-8') as f:
                _model_data = json.load(f)
        except Exception as e:
            print("Error loading weights:", e)
    return _model_data

def predict_sentiment_exact(text, model_data):
    features = model_data.get('features', {})
    intercept = model_data.get('intercept', 0.0)
    
    clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
    words = clean.split()
    tokens = list(words)
    for i in range(len(words) - 1):
        tokens.append(f"{words[i]} {words[i+1]}")
        
    counts = collections.Counter(tokens)
    vec_dict = {}
    sum_sq = 0.0
    
    for t, count in counts.items():
        if t in features:
            idx, idf_val, coef = features[t]
            val = (1.0 + math.log(count)) * idf_val
            vec_dict[t] = (val, coef)
            sum_sq += val * val
            
    if sum_sq == 0.0:
        prob = 0.5
        words_impact = []
    else:
        norm = math.sqrt(sum_sq)
        dot = intercept
        words_impact = []
        for t, (val, coef) in vec_dict.items():
            normalized_val = val / norm
            impact = normalized_val * coef
            dot += impact
            words_impact.append((t, impact))
            
        prob = 1.0 / (1.0 + math.exp(-dot))
        
    is_positive = prob >= 0.5
    score_10 = round(1.0 + prob * 9.0, 1)
    star_count = max(1, min(5, int(round(score_10 / 2.0))))
    stars_visual = "⭐" * star_count + "☆" * (5 - star_count)
    
    words_impact.sort(key=lambda x: x[1], reverse=True)
    top_pos = [w for w, score in words_impact if score > 0][:5]
    top_neg = [w for w, score in reversed(words_impact) if score < 0][:5]
    
    return {
        'sentiment': "Ijobiy (Positive)" if is_positive else "Salbiy (Negative)",
        'is_positive': is_positive,
        'probability_positive': prob,
        'probability_negative': 1.0 - prob,
        'confidence': prob if is_positive else (1.0 - prob),
        'score_10': score_10,
        'stars': stars_visual,
        'star_count': star_count,
        'top_positive_words': top_pos,
        'top_negative_words': top_neg
    }

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response_data = {
            "status": "online",
            "message": "CineAnalytics AI API is running on Vercel (Optimized Pure Python Runtime)",
            "dataset": "yasserh/imdb-movie-ratings-sentiment-analysis",
            "metrics": {
                "accuracy": 0.8953,
                "precision": 0.8901,
                "recall": 0.9017,
                "f1": 0.8958
            }
        }
        self.wfile.write(json.dumps(response_data).encode('utf-8'))

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        
        try:
            data = json.loads(body.decode('utf-8'))
            text = data.get('text', '')
            
            m_data = get_model_data()
            if not text or not text.strip():
                result = {"error": "Text is empty"}
            elif not m_data:
                result = {"error": "Model weights file not found"}
            else:
                result = predict_sentiment_exact(text, m_data)
                
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
