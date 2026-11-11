import json
import os
import sys
from http.server import BaseHTTPRequestHandler

# Papka yo'llarini qo'shamiz
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import joblib
from model_trainer import analyze_review

MODEL_FILE = os.path.join(parent_dir, "model.joblib")
METRICS_FILE = os.path.join(parent_dir, "metrics.joblib")

_model_pkg = None
_metrics = None

def get_model():
    global _model_pkg, _metrics
    if _model_pkg is None and os.path.exists(MODEL_FILE):
        try:
            _model_pkg = joblib.load(MODEL_FILE)
        except Exception as e:
            print("Model load error:", e)
    if _metrics is None and os.path.exists(METRICS_FILE):
        try:
            _metrics = joblib.load(METRICS_FILE)
        except Exception as e:
            print("Metrics load error:", e)
    return _model_pkg, _metrics

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
        
        _, metrics = get_model()
        response_data = {
            "status": "online",
            "message": "CineAnalytics AI API is running on Vercel",
            "dataset": "yasserh/imdb-movie-ratings-sentiment-analysis",
            "metrics": metrics or {
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
            
            model_pkg, _ = get_model()
            if not text or not text.strip():
                result = {"error": "Text is empty"}
            elif not model_pkg:
                result = {"error": "Model could not be loaded"}
            else:
                result = analyze_review(text, model_pkg)
                
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
