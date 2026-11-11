# 🎬 CineAnalytics — Kino va Filmlar Reytingini Tahlil Qilish Portali

[![CineAnalytics CI](https://github.com/javohirshokirjonov58-spec/Filmlar/actions/workflows/ci.yml/badge.svg)](https://github.com/javohirshokirjonov58-spec/Filmlar/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66.1-FF4B4B.svg)](https://streamlit.io/)
[![Vercel](https://img.shields.io/badge/Vercel-Ready-black.svg?logo=vercel)](https://vercel.com/)
[![Accuracy](https://img.shields.io/badge/Model%20Accuracy-89.53%25-brightgreen.svg)]()

Ushbu platforma **IMDb** ning 40,000 ta kino va filmlar sharhlari dataseti asosida ishlaydigan zamonaviy, interaktiv tahlil va Sun'iy Intellekt (NLP) veb-portalidir.

---

## 📌 Dataset Haqida va Ulanish

Dataset **KaggleHub** orqali avtomatik tarzda yuklanadi:
- **Dataset nomi:** `yasserh/imdb-movie-ratings-sentiment-analysis`
- **Tarkibi:** 40,000 ta real IMDb sharhlari (50% ijobiy, 50% salbiy)

### Datasetga ulanish kodi:
```python
import kagglehub
import pandas as pd
import os

# 1. KaggleHub orqali yuklab olish
path = kagglehub.dataset_download("yasserh/imdb-movie-ratings-sentiment-analysis")
print("Path to dataset files:", path)

# 2. Ma'lumotlarni o'qish
csv_file = os.path.join(path, "movie.csv")
df = pd.read_csv(csv_file)
print("Dataset shakli:", df.shape)
```

---

## 🚀 Asosiy Imkoniyatlar va Funksiyalar

1. **📊 Umumiy Tahlil (EDA & Statistika):**
   - Jami sharhlar soni (40,000), ijobiy va salbiy taqsimot (Donut chart).
   - Sharhlarning uzunligi va so'zlar soni taqsimoti (Gistogramma).
   - Ijobiy va salbiy reytinglarga eng kuchli ta'sir qiluvchi kalit so'zlar vizualizatsiyasi.

2. **🤖 Sun'iy Intellekt Tahlili (Live Predictor):**
   - Istalgan film sharhini kiritish va uni 1 soniyada tahlil qilish.
   - Ijobiy / Salbiy holat, ishonchlilik foizi (%), 10 ballik IMDb reytingi hamda yulduzchalar (⭐⭐⭐⭐⭐).
   - Mashhur filmlar (*Oppenheimer*, *Inception*, *The Godfather*, *Morbius*, *The Room*) sharhlarini bir bosishda sinab ko'rish.
   - Qaror qabul qilishda qaysi so'zlar ijobiy yoki salbiy ta'sir qilganini ko'rsatuvchi indikatorlar.

3. **⚖️ Ikki Filmni Solishtirish:**
   - 2 ta film sharhini yonma-yon kiritish va solishtirish.
   - Qaysi film tomoshabinlarga ko'proq ma'qul kelganini va ballar farqini aniqlash.

4. **🔍 40,000 ta Sharhlar Eksploreri:**
   - Istalgan so'z (masalan rejissyor nomi, janr yoki sifat) bo'yicha tezkor qidiruv.
   - Filterlar va natijalarni CSV formatida yuklab olish.

5. **📈 Model Metriki:**
   - TF-IDF + Logistic Regression arxitekturasi: **89.53% aniqlik (Accuracy)**, **89.58% F1-score**.
   - Interaktiv Chalkashlik matritsasi (Confusion Matrix).

---

## 💻 Qanday Ishga Tushiriladi?

### 1-usul (Kompyuterda - Eng oson):
Loyiha papkasidagi **`run_app.bat`** faylini ikki marta bosing. Sayt avtomatik ravishda brauzeringizda ochiladi:
`http://localhost:8501`

### 2-usul (Buyruqlar qatori orqali):
```bash
streamlit run app.py
```

### 3-usul (Vercel orqali Bulutda):
1. [vercel.com/new](https://vercel.com/new) sahifasiga kiring.
2. `javohirshokirjonov58-spec/Filmlar` repozitoriyasini tanlang va **Deploy** tugmasini bosing.
3. Loyihangiz bir zumda dunyo bo'ylab jonli ishlaydi!
