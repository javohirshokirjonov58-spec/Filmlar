@echo off
title CineAnalytics - Kino va Filmlar Reytingi Tahlil Portali
echo ===================================================================
echo     CineAnalytics: Kino va Filmlar Reytingi Tahlil Portali
echo     KaggleHub: yasserh/imdb-movie-ratings-sentiment-analysis
echo ===================================================================
echo.
echo Dastur ishga tushirilmoqda, iltimos kuting...
echo Brauzeringizda avtomatik ochiladi: http://localhost:8501
echo.
"C:\Users\user\AppData\Local\Programs\Python\Python314\python.exe" -m streamlit run app.py --server.headless=false
pause

