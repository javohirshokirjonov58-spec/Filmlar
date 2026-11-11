"""
model_trainer.py - TF-IDF va Logistic Regression orqali matnli sharhlar tahlili modeli
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from data_loader import load_data

MODEL_FILE = os.path.join(os.path.dirname(__file__), "model.joblib")
METRICS_FILE = os.path.join(os.path.dirname(__file__), "metrics.joblib")

def train_and_save_model():
    """
    Modelni to'liq IMDb dataseti asosida o'qitadi va saqlaydi.
    """
    print("Dataset o'qilmoqda...")
    df, _ = load_data()
    
    print(f"Jami ma'lumotlar soni: {len(df)}")
    X = df['text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("TF-IDF Vectorizer va Logistic Regression o'qitilmoqda...")
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words='english'
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    classifier = LogisticRegression(C=2.5, max_iter=1000, random_state=42)
    classifier.fit(X_train_vec, y_train)
    
    y_pred = classifier.predict(X_test_vec)
    y_prob = classifier.predict_proba(X_test_vec)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm.tolist(),
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    print(f"Model natijalari:")
    print(f" - Aniqlik (Accuracy): {acc*100:.2f}%")
    print(f" - Precision: {prec*100:.2f}%")
    print(f" - Recall: {rec*100:.2f}%")
    print(f" - F1 Score: {f1*100:.2f}%")
    
    # Model va vektorizatorni birgalikda saqlaymiz
    package = {
        'vectorizer': vectorizer,
        'classifier': classifier
    }
    
    joblib.dump(package, MODEL_FILE)
    joblib.dump(metrics, METRICS_FILE)
    print(f"Model saqlandi: {MODEL_FILE}")
    print(f"Metrikalar saqlandi: {METRICS_FILE}")
    return package, metrics

def load_or_train_model():
    """
    Agar saqlangan model bo'lsa uni yuklaydi, aks holda yangisini o'qitadi.
    """
    if os.path.exists(MODEL_FILE) and os.path.exists(METRICS_FILE):
        try:
            package = joblib.load(MODEL_FILE)
            metrics = joblib.load(METRICS_FILE)
            return package, metrics
        except Exception:
            pass
    return train_and_save_model()

def analyze_review(text, package):
    """
    Bitta matn bo'yicha tahlil natijasini qaytaradi:
    - Ijobiy/Salbiy
    - Ehtimollik foizi
    - 10 ballik IMDb reytingi
    - Yulduzchalar (1-5)
    - Eng ta'sirli ijobiy va salbiy so'zlar
    """
    if not text or not text.strip():
        return None
        
    vec = package['vectorizer']
    clf = package['classifier']
    
    text_vec = vec.transform([text])
    prob_pos = float(clf.predict_proba(text_vec)[0, 1])
    is_positive = prob_pos >= 0.5
    
    # 1.0 dan 10.0 gacha IMDb reytingi hisoblash
    # prob_pos = 0 -> 1.0; prob_pos = 1 -> 10.0
    # Sigmoid ko'rinishida real IMDb taqsimotiga yaqinlashtirish
    score_10 = round(1.0 + prob_pos * 9.0, 1)
    
    # Yulduzchalar soni (1 dan 5 gacha)
    star_count = max(1, min(5, int(round(score_10 / 2.0))))
    stars_visual = "⭐" * star_count + "☆" * (5 - star_count)
    
    # Matndagi eng ta'sirchan so'zlarni aniqlash
    feature_names = vec.get_feature_names_out()
    coefs = clf.coef_[0]
    
    # Matnda qatnashgan so'zlar indekslari
    cx = text_vec.tocoo()
    word_impacts = []
    for idx in cx.col:
        word = feature_names[idx]
        weight = coefs[idx] * text_vec[0, idx]
        word_impacts.append((word, float(weight)))
        
    word_impacts.sort(key=lambda x: x[1], reverse=True)
    top_positive_words = [w for w, score in word_impacts if score > 0][:5]
    top_negative_words = [w for w, score in reversed(word_impacts) if score < 0][:5]
    
    return {
        'sentiment': "Ijobiy (Positive)" if is_positive else "Salbiy (Negative)",
        'is_positive': is_positive,
        'probability_positive': prob_pos,
        'probability_negative': 1.0 - prob_pos,
        'confidence': prob_pos if is_positive else (1.0 - prob_pos),
        'score_10': score_10,
        'stars': stars_visual,
        'star_count': star_count,
        'top_positive_words': top_positive_words,
        'top_negative_words': top_negative_words
    }

if __name__ == "__main__":
    train_and_save_model()

