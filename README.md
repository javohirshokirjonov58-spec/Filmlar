# 🎬 CineAnalytics — Kino va Filmlar Reytingini Tahlil Qilish Portali

Ushbu loyiha **IMDb** ning 40,000 ta kino va filmlar sharhlari dataseti asosida ishlaydigan zamonaviy, interaktiv tahlil va sun'iy intellekt (NLP) veb-platformasidir.

---

## 📌 Dataset Haqida va Ulanish

Dataset **KaggleHub** orqali avtomatik tarzda yuklanadi:
- **Dataset nomi:** `yasserh/imdb-movie-ratings-sentiment-analysis`
- **Tarkibi:** 40,000 ta haqiqiy IMDb sharhlari (50% ijobiy, 50% salbiy)

### Datasetga ulanish kodi:
```python
import kagglehub
import pandas as pd
import os

# Download latest version
path = kagglehub.dataset_download("yasserh/imdb-movie-ratings-sentiment-analysis")
print("Path to dataset files:", path)

# Ma'lumotlarni o'qish
df = pd.read_csv(os.path.join(path, "movie.csv"))
print(df.head())
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
   - TF-IDF + Logistic Regression arxitekturasi: **89.53% aniqlik (Accuracy)**.
   - Interaktiv Chalkashlik matritsasi (Confusion Matrix).

---

## 💻 Qanday Ishga Tushiriladi?

### 1-usul (Eng oson):
Loyiha papkasidagi **`run_app.bat`** faylini ikki marta bosing. Sayt avtomatik ravishda brauzeringizda ochiladi.

### 2-usul (Buyruqlar qatori orqali):
```bash
streamlit run app.py
```
yoki:
```bash
"C:\Users\user\AppData\Local\Programs\Python\Python314\python.exe" -m streamlit run app.py
```
Brauzer manzili: `http://localhost:8501`

