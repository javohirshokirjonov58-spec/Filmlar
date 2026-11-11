"""
CineAnalytics - Kino va Filmlar Reytingini Tahlil Qilish Portali
Dataset: yasserh/imdb-movie-ratings-sentiment-analysis (KaggleHub orqali)
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_loader import load_data, DATASET_NAME
from model_trainer import load_or_train_model, analyze_review

# 1. Sahifa konfiguratsiyasi
st.set_page_config(
    page_title="CineAnalytics | Kino & Film Reyting Tahlili",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Maxsus zamonaviy Cinema CSS dizayni
st.markdown("""
<style>
    /* Asosiy fon va shriftlar */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0d1117 0%, #161b22 100%);
        color: #f0f6fc;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Sarlavha banneri */
    .hero-banner {
        background: linear-gradient(135deg, rgba(245, 197, 24, 0.15) 0%, rgba(229, 9, 20, 0.2) 50%, rgba(13, 17, 23, 0.8) 100%);
        border: 1px solid rgba(245, 197, 24, 0.3);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #f5c518, #ff6b6b, #00d2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    .hero-subtitle {
        color: #8b949e;
        font-size: 1.05rem;
    }
    
    /* Metrika kartalari */
    .metric-card {
        background: rgba(22, 27, 34, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: #f5c518;
    }
    .metric-num {
        font-size: 1.9rem;
        font-weight: 700;
        color: #f5c518;
    }
    .metric-label {
        font-size: 0.88rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 5px;
    }
    
    /* Natija kartochkalari */
    .result-card-pos {
        background: linear-gradient(145deg, rgba(35, 134, 54, 0.2) 0%, rgba(22, 27, 34, 0.9) 100%);
        border: 1px solid #2ea043;
        border-radius: 14px;
        padding: 22px;
        margin: 15px 0;
    }
    
    .result-card-neg {
        background: linear-gradient(145deg, rgba(218, 54, 51, 0.2) 0%, rgba(22, 27, 34, 0.9) 100%);
        border: 1px solid #da3633;
        border-radius: 14px;
        padding: 22px;
        margin: 15px 0;
    }
    
    .badge-pos {
        background-color: #238636;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .badge-neg {
        background-color: #da3633;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .tag-word {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 3px 10px;
        border-radius: 15px;
        margin: 3px;
        display: inline-block;
        font-size: 0.85rem;
    }
    
    /* Code block styling */
    .code-container {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
        font-family: 'Consolas', monospace;
        font-size: 0.88rem;
        color: #58a6ff;
    }
</style>
""", unsafe_allow_html=True)

# 3. Ma'lumotlar va modelni yuklash (Kesh bilan)
@st.cache_data(show_spinner=False)
def get_cached_data():
    return load_data()

@st.cache_resource(show_spinner=False)
def get_cached_model():
    return load_or_train_model()

# Header qismi
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🎬 CineAnalytics: Kino va Filmlar Reytingi Tahlil Portali</div>
    <div class="hero-subtitle">
        IMDb 40,000 ta sharhlar dataseti va Sun'iy Intellekt (NLP) asosida kino reytinglari, his-tuyg'ular (sentiment) va tomoshabinlar munosabatini chuqur tahlil qilish platformasi.
    </div>
</div>
""", unsafe_allow_html=True)

# Yuklash holatini ko'rsatish
with st.spinner("Dataset va Sun'iy intellekt modeli yuklanmoqda..."):
    df, dataset_path = get_cached_data()
    model_pkg, metrics = get_cached_model()

# Sidebar paneli
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/6/69/IMDB_Logo_2016.svg", width=120)
st.sidebar.title("📌 Boshqaruv Paneli")

# KaggleHub ulanish ma'lumoti
with st.sidebar.expander("🔗 KaggleHub Ulanish Ma'lumoti", expanded=True):
    st.markdown(f"**Dataset:** `{DATASET_NAME}`")
    st.markdown(f"**Holati:** 🟢 Yuklangan")
    st.markdown(f"**Fayl manzili:**\n`{dataset_path}`")
    st.code("""import kagglehub
path = kagglehub.dataset_download(
    "yasserh/imdb-movie-ratings-sentiment-analysis"
)
print("Path to dataset files:", path)""", language="python")

# Sahifani tanlash
menu = st.sidebar.radio(
    "Bo'limni tanlang:",
    [
        "📊 Umumiy Tahlil (EDA & Statistika)",
        "🤖 Sun'iy Intellekt Tahlili (Live Predictor)",
        "⚖️ Ikki Filmni Solishtirish",
        "🔍 Sharhlar Qidiruvi & Eksplorer",
        "📈 Model Metriki & Texnik Arxitektura"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("""
💡 **Foydalanish bo'yicha maslahat:**
- Istalgan film sharhini ingliz tilida kiriting va sun'iy intellekt uning ijobiy/salbiy reytingini soniyalar ichida aniqlab beradi.
""")

# ==============================================================================
# SAHIFA 1: UMUMIY TAHLIL (EDA & STATISTIKA)
# ==============================================================================
if menu == "📊 Umumiy Tahlil (EDA & Statistika)":
    st.subheader("📊 Dataset Statistikasi va Umumiy Tahlil")
    
    # Karta ko'rsatkichlari
    total_reviews = len(df)
    pos_count = (df['label'] == 1).sum()
    neg_count = (df['label'] == 0).sum()
    avg_words = int(df['word_count'].mean())
    max_words = int(df['word_count'].max())
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-num">{total_reviews:,}</div>
            <div class="metric-label">Jami Sharhlar</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-num" style="color: #2ea043;">{pos_count:,} ({pos_count/total_reviews*100:.1f}%)</div>
            <div class="metric-label">Ijobiy Sharhlar</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-num" style="color: #da3633;">{neg_count:,} ({neg_count/total_reviews*100:.1f}%)</div>
            <div class="metric-label">Salbiy Sharhlar</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-num" style="color: #00d2ff;">{avg_words}</div>
            <div class="metric-label">O'rtacha So'zlar Soni</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # 2 ta grafik qatori
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("#### 🎯 1. His-tuyg'ular Taqsimoti (Sentiment Balance)")
        donut_fig = go.Figure(data=[go.Pie(
            labels=['Ijobiy (Positive)', 'Salbiy (Negative)'],
            values=[pos_count, neg_count],
            hole=.55,
            marker=dict(colors=['#2ea043', '#da3633']),
            textinfo='label+percent',
            hoverinfo='value+percent'
        )])
        donut_fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c9d1d9'),
            margin=dict(t=20, b=20, l=20, r=20),
            showlegend=False
        )
        st.plotly_chart(donut_fig, use_container_width=True)
        st.caption("Izoh: Dataset mukammal darajada muvozanatlangan (50% ijobiy va 50% salbiy sharhlar).")

    with col_chart2:
        st.markdown("#### 📏 2. Sharhlardagi So'zlar Soni Taqsimoti")
        hist_data = df[df['word_count'] <= 800] # outlierlarni qisqartirish uchun
        hist_fig = px.histogram(
            hist_data,
            x='word_count',
            color='sentiment_uz',
            color_discrete_map={'Ijobiy (Positive)': '#2ea043', 'Salbiy (Negative)': '#da3633'},
            barmode='overlay',
            opacity=0.75,
            labels={'word_count': "So'zlar soni", 'sentiment_uz': "Sentiment"}
        )
        hist_fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c9d1d9'),
            xaxis=dict(gridcolor='#30363d'),
            yaxis=dict(gridcolor='#30363d'),
            margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(hist_fig, use_container_width=True)
        st.caption("Izoh: Ko'pchilik foydalanuvchilar 100 dan 300 gacha so'zdan iborat sharh yozishgan.")

    st.markdown("---")
    
    # 2-qator grafiklar: Eng muhim so'zlar
    st.markdown("#### 🔑 3. Eng Ko'p Uchraydigan Ta'sirli So'zlar (Top Sentiment Keywords)")
    c_w1, c_w2 = st.columns(2)
    
    vec = model_pkg['vectorizer']
    clf = model_pkg['classifier']
    feat_names = np.array(vec.get_feature_names_out())
    coefs = clf.coef_[0]
    
    top_pos_idx = np.argsort(coefs)[-12:]
    top_neg_idx = np.argsort(coefs)[:12]
    
    pos_words_df = pd.DataFrame({
        'Word': feat_names[top_pos_idx],
        'Weight': coefs[top_pos_idx]
    }).sort_values('Weight', ascending=True)
    
    neg_words_df = pd.DataFrame({
        'Word': feat_names[top_neg_idx],
        'Weight': np.abs(coefs[top_neg_idx])
    }).sort_values('Weight', ascending=True)
    
    with c_w1:
        st.markdown("##### 🟢 Ijobiy reytingni belgilovchi so'zlar:")
        bar_pos = px.bar(
            pos_words_df,
            x='Weight',
            y='Word',
            orientation='h',
            color='Weight',
            color_continuous_scale=['#196c2e', '#2ea043', '#56d364']
        )
        bar_pos.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c9d1d9'),
            coloraxis_showscale=False,
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis=dict(gridcolor='#30363d'),
            yaxis=dict(gridcolor='#30363d')
        )
        st.plotly_chart(bar_pos, use_container_width=True)
        
    with c_w2:
        st.markdown("##### 🔴 Salbiy reytingni belgilovchi so'zlar:")
        bar_neg = px.bar(
            neg_words_df,
            x='Weight',
            y='Word',
            orientation='h',
            color='Weight',
            color_continuous_scale=['#a82824', '#da3633', '#f85149']
        )
        bar_neg.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c9d1d9'),
            coloraxis_showscale=False,
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis=dict(gridcolor='#30363d'),
            yaxis=dict(gridcolor='#30363d')
        )
        st.plotly_chart(bar_neg, use_container_width=True)

# ==============================================================================
# SAHIFA 2: SUN'IY INTELLEKT TAHLILI (LIVE PREDICTOR)
# ==============================================================================
elif menu == "🤖 Sun'iy Intellekt Tahlili (Live Predictor)":
    st.subheader("🤖 Sun'iy Intellekt Film Reytingi & Sentiment Bashorati")
    st.markdown("Istalgan film sharhini quyidagi maydonga yozing yoki namunaviy sharhlardan birini tanlang:")
    
    # Namunaviy mashhur filmlar sharhlari
    sample_reviews = {
        "Tanlang...": "",
        "🏆 Oppenheimer (Ajoyib asar)": "Oppenheimer is a cinematic masterpiece. Christopher Nolan delivers an intense, haunting biographical drama with breathtaking sound design and exceptional acting by Cillian Murphy. Absolutely brilliant!",
        "🔥 Inception (Yuqori daraja)": "One of the greatest mind-bending sci-fi thrillers ever made. The visual effects, concept, and Hans Zimmer's score are unforgettable. A solid 10 out of 10.",
        "👑 The Godfather (Klassika)": "An absolute monumental classic. Perfect storytelling, unmatched atmosphere, and Marlon Brando gives a performance of a lifetime. Highly recommended.",
        "👎 Morbius (Zararli / Zerikarli)": "This movie was a complete disaster. Horrible CGI, boring plot, terrible script, and zero emotional connection. A huge waste of time and money.",
        "💩 The Room (Juda yomon film)": "Painfully bad dialogue, non-existent plot, laughable acting, and completely confusing scenes. One of the worst films in history.",
        "😐 O'rtacha film": "It had some decent moments and nice cinematography, but the pacing was sluggish and the ending felt quite predictable. Just okay, not great."
    }
    
    selected_sample = st.selectbox("⚡ Tezkor namunaviy sharhni sinab ko'rish:", list(sample_reviews.keys()))
    default_text = sample_reviews[selected_sample] if selected_sample != "Tanlang..." else ""
    
    user_review = st.text_area(
        "📝 Film sharhi matni (ingliz tilida):",
        value=default_text,
        height=140,
        placeholder="Masalan: This movie had phenomenal acting, stunning visuals, and kept me on the edge of my seat from start to finish..."
    )
    
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        predict_button = st.button("🚀 Tahlil Qilish", type="primary", use_container_width=True)
    with col_btn2:
        if st.button("🔄 Tozalash"):
            user_review = ""
            st.rerun()
            
    if predict_button and user_review.strip():
        with st.spinner("Matn tahlil qilinmoqda..."):
            res = analyze_review(user_review, model_pkg)
            
        if res:
            st.markdown("### 🎯 Tahlil Natijalari:")
            
            # Asosiy natija banneri
            card_class = "result-card-pos" if res['is_positive'] else "result-card-neg"
            badge_class = "badge-pos" if res['is_positive'] else "badge-neg"
            icon = "🌟 Ijobiy Baho" if res['is_positive'] else "⚠️ Salbiy Baho"
            
            st.markdown(f"""
            <div class="{card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div>
                        <span class="{badge_class}">{res['sentiment']}</span>
                        <h2 style="margin: 10px 0 5px 0; color: #f0f6fc;">{icon}</h2>
                        <p style="color: #8b949e; margin: 0;">AI Ishonch darajasi: <b>{res['confidence']*100:.1f}%</b></p>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 2.5rem; font-weight: 800; color: #f5c518;">
                            {res['score_10']} <span style="font-size: 1.2rem; color: #8b949e;">/ 10</span>
                        </div>
                        <div style="font-size: 1.4rem;">{res['stars']}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress bar va taqsimot
            r1, r2 = st.columns(2)
            with r1:
                st.markdown("##### 📊 Ehtimollik nisbati:")
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=res['probability_positive'] * 100,
                    number={'suffix': "%", 'font': {'color': '#f0f6fc', 'size': 32}},
                    title={'text': "Ijobiylik darajasi", 'font': {'color': '#8b949e'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': '#8b949e'},
                        'bar': {'color': "#2ea043" if res['is_positive'] else "#da3633"},
                        'steps': [
                            {'range': [0, 50], 'color': "rgba(218, 54, 51, 0.2)"},
                            {'range': [50, 100], 'color': "rgba(46, 160, 67, 0.2)"}
                        ],
                        'threshold': {
                            'line': {'color': "#f5c518", 'width': 4},
                            'thickness': 0.75,
                            'value': res['probability_positive'] * 100
                        }
                    }
                ))
                gauge_fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=240,
                    margin=dict(t=40, b=20, l=30, r=30)
                )
                st.plotly_chart(gauge_fig, use_container_width=True)
                
            with r2:
                st.markdown("##### 🔍 Tahlilga ta'sir qilgan kalit so'zlar:")
                st.write("**Ijobiy taassurot uyg'otgan so'zlar:**")
                if res['top_positive_words']:
                    words_pos_html = " ".join([f"<span class='tag-word' style='border-color: #2ea043; color: #56d364;'>+{w}</span>" for w in res['top_positive_words']])
                    st.markdown(words_pos_html, unsafe_allow_html=True)
                else:
                    st.caption("Ijobiy kalit so'zlar aniqlanmadi.")
                    
                st.write("**Salbiy taassurot uyg'otgan so'zlar:**")
                if res['top_negative_words']:
                    words_neg_html = " ".join([f"<span class='tag-word' style='border-color: #da3633; color: #f85149;'>-{w}</span>" for w in res['top_negative_words']])
                    st.markdown(words_neg_html, unsafe_allow_html=True)
                else:
                    st.caption("Salbiy kalit so'zlar aniqlanmadi.")
                    
                st.markdown("---")
                if res['is_positive']:
                    st.success("💡 **Xulosa:** Ushbu film tomoshabinlarga tavsiya etiladi. Matn ichidagi ijobiy baholar filmning yuqori sifatidan dalolat beradi.")
                else:
                    st.error("💡 **Xulosa:** Ushbu film tomoshabinda salbiy taassurot qoldirgan. Matnda kamchiliklar va noroziliklar yaqqol ifodalangan.")

# ==============================================================================
# SAHIFA 3: IKKI FILMNI SOLISHTIRISH
# ==============================================================================
elif menu == "⚖️ Ikki Filmni Solishtirish":
    st.subheader("⚖️ Ikki Film Sharhini Yonma-yon Solishtirish")
    st.markdown("Ikkita turli kino yoki sharhni kiriting va ularning qaysi biri tomoshabinlar tomonidan yaxshiroq baholanganini ko'ring:")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("#### 🎬 Film A")
        name_a = st.text_input("Film A nomi:", value="Interstellar")
        review_a = st.text_area(
            "Film A sharhi:",
            value="Interstellar is a breathtaking masterpiece of science fiction with an incredible soundtrack and deep emotional storytelling.",
            height=120
        )
        
    with col_b:
        st.markdown("#### 🎬 Film B")
        name_b = st.text_input("Film B nomi:", value="Transformers 5")
        review_b = st.text_area(
            "Film B sharhi:",
            value="Disappointing and noisy film with a messy story, boring characters, and completely overused explosions.",
            height=120
        )
        
    if st.button("⚔️ Solishtirish", type="primary"):
        res_a = analyze_review(review_a, model_pkg)
        res_b = analyze_review(review_b, model_pkg)
        
        if res_a and res_b:
            st.markdown("---")
            st.markdown("### 🏆 Solishtirish Natijasi:")
            
            c_res_a, c_vs, c_res_b = st.columns([5, 2, 5])
            
            with c_res_a:
                card_a = "result-card-pos" if res_a['is_positive'] else "result-card-neg"
                st.markdown(f"""
                <div class="{card_a}">
                    <h3>{name_a}</h3>
                    <div style="font-size: 2rem; font-weight: bold; color: #f5c518;">{res_a['score_10']} / 10</div>
                    <div>{res_a['stars']}</div>
                    <p>Sentiment: <b>{res_a['sentiment']}</b></p>
                    <p>Ijobiylik darajasi: <b>{res_a['probability_positive']*100:.1f}%</b></p>
                </div>
                """, unsafe_allow_html=True)
                
            with c_vs:
                st.markdown("<div style='text-align: center; padding-top: 60px;'><h1 style='color: #f5c518;'>VS</h1></div>", unsafe_allow_html=True)
                
            with c_res_b:
                card_b = "result-card-pos" if res_b['is_positive'] else "result-card-neg"
                st.markdown(f"""
                <div class="{card_b}">
                    <h3>{name_b}</h3>
                    <div style="font-size: 2rem; font-weight: bold; color: #f5c518;">{res_b['score_10']} / 10</div>
                    <div>{res_b['stars']}</div>
                    <p>Sentiment: <b>{res_b['sentiment']}</b></p>
                    <p>Ijobiylik darajasi: <b>{res_b['probability_positive']*100:.1f}%</b></p>
                </div>
                """, unsafe_allow_html=True)
                
            # G'olibni aniqlash
            st.markdown("---")
            if res_a['score_10'] > res_b['score_10']:
                diff = res_a['score_10'] - res_b['score_10']
                st.success(f"🎉 **G'olib: {name_a}!** Ushbu film {name_b} filmiga qaraganda **+{diff:.1f}** ball yuqoriroq baholandi va tomoshabinlar olqishiga sazovor bo'ldi.")
            elif res_b['score_10'] > res_a['score_10']:
                diff = res_b['score_10'] - res_a['score_10']
                st.success(f"🎉 **G'olib: {name_b}!** Ushbu film {name_a} filmiga qaraganda **+{diff:.1f}** ball yuqoriroq baholandi va tomoshabinlar olqishiga sazovor bo'ldi.")
            else:
                st.info("🤝 **Durang!** Har ikkala film tomoshabinlar tomonidan teng darajada baholandi.")

# ==============================================================================
# SAHIFA 4: SHARHLAR QIDIRUVI & EKSPLORER
# ==============================================================================
elif menu == "🔍 Sharhlar Qidiruvi & Eksplorer":
    st.subheader("🔍 40,000 ta IMDb Sharhlar Omboridan Qidiruv")
    st.markdown("KaggleHub orqali yuklangan dataset ichidan istalgan mavzu, janr yoki aktyor/rejissyor bo'yicha sharhlarni izlang:")
    
    col_s1, col_s2, col_s3 = st.columns([3, 2, 2])
    with col_s1:
        query = st.text_input("🔎 Kalit so'z bo'yicha qidirish:", placeholder="masalan: Nolan, comedy, scary, boring, Oscar...")
    with col_s2:
        sentiment_filter = st.selectbox("His-tuyg'u bo'yicha filter:", ["Barchasi", "Faqat Ijobiy", "Faqat Salbiy"])
    with col_s3:
        max_rows = st.slider("Ko'rsatish soni:", min_value=5, max_value=50, value=10, step=5)
        
    filtered_df = df
    if query:
        filtered_df = filtered_df[filtered_df['text'].str.contains(query, case=False, na=False)]
        
    if sentiment_filter == "Faqat Ijobiy":
        filtered_df = filtered_df[filtered_df['label'] == 1]
    elif sentiment_filter == "Faqat Salbiy":
        filtered_df = filtered_df[filtered_df['label'] == 0]
        
    st.markdown(f"**Topilgan natijalar:** `{len(filtered_df):,}` ta sharh")
    
    # Natijalarni chiroyli kartochkalarda chiqarish
    for idx, row in filtered_df.head(max_rows).iterrows():
        is_pos = row['label'] == 1
        badge_style = "badge-pos" if is_pos else "badge-neg"
        badge_label = "🟢 IJOBIY" if is_pos else "🔴 SALBIY"
        card_style = "result-card-pos" if is_pos else "result-card-neg"
        
        # Matnni qisqartirish
        text_preview = row['text']
        if len(text_preview) > 350:
            text_preview = text_preview[:350] + "..."
            
        st.markdown(f"""
        <div class="{card_style}">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span class="{badge_style}">{badge_label}</span>
                <span style="color: #8b949e; font-size: 0.85rem;">So'zlar: {row['word_count']} ta | Belgilar: {row['char_count']} ta</span>
            </div>
            <div style="color: #e6edf3; font-style: italic; line-height: 1.5;">"{text_preview}"</div>
        </div>
        """, unsafe_allow_html=True)
        
    # CSV yuklab olish imkoniyati
    if len(filtered_df) > 0:
        csv_data = filtered_df.head(500).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Saralangan sharhlarni yuklab olish (CSV)",
            data=csv_data,
            file_name="imdb_filtered_reviews.csv",
            mime="text/csv"
        )

# ==============================================================================
# SAHIFA 5: MODEL METRIKI & TEXNIK ARXITEKTURA
# ==============================================================================
elif menu == "📈 Model Metriki & Texnik Arxitektura":
    st.subheader("📈 Sun'iy Intellekt Modeli Arxitekturasi va Metriklari")
    
    st.markdown("""
    Ushbu platforma matnli sharhlarni tahlil qilish uchun quyidagi texnologiyalardan foydalanadi:
    - **Dataset Manbasi:** KaggleHub orqali yuklangan `yasserh/imdb-movie-ratings-sentiment-analysis`
    - **Vektorizatsiya (Feature Extraction):** TF-IDF (Term Frequency-Inverse Document Frequency) unigram va bigramlar bilan (`max_features=15,000`, `sublinear_tf=True`)
    - **Klassifikator (Classifier):** Yuqori aniqlikdagi L2-regularizatsiyalangan Logistic Regression modeli
    """)
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-num" style="color: #58a6ff;">{metrics['accuracy']*100:.2f}%</div>
            <div class="metric-label">Aniqlik (Accuracy)</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-num" style="color: #2ea043;">{metrics['precision']*100:.2f}%</div>
            <div class="metric-label">Precision</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-num" style="color: #f5c518;">{metrics['recall']*100:.2f}%</div>
            <div class="metric-label">Recall</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-num" style="color: #bc8cff;">{metrics['f1']*100:.2f}%</div>
            <div class="metric-label">F1-Score</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    col_cm1, col_cm2 = st.columns(2)
    with col_cm1:
        st.markdown("#### 🔲 Chalkashlik Matritsasi (Confusion Matrix)")
        cm = np.array(metrics['confusion_matrix'])
        cm_fig = px.imshow(
            cm,
            labels=dict(x="Bashorat qilingan", y="Haqiqiy holat", color="Soni"),
            x=['Salbiy (0)', 'Ijobiy (1)'],
            y=['Salbiy (0)', 'Ijobiy (1)'],
            text_auto=True,
            color_continuous_scale="Viridis"
        )
        cm_fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c9d1d9'),
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(cm_fig, use_container_width=True)
        st.caption("Izoh: Test to'plamidagi 8,000 ta sharh bo'yicha bashorat aniqligi ko'rsatilgan.")
        
    with col_cm2:
        st.markdown("#### 💻 Datasetga Ulanish Kodi (Python)")
        st.code("""import kagglehub
import pandas as pd
import os

# 1. KaggleHub orqali yuklab olish
path = kagglehub.dataset_download("yasserh/imdb-movie-ratings-sentiment-analysis")
print("Path to dataset files:", path)

# 2. CSV faylni o'qish
csv_file = os.path.join(path, "movie.csv")
df = pd.read_csv(csv_file)

# 3. Natijani tekshirish
print("Dataset o'lchami:", df.shape)
print(df.head())""", language="python")
        st.info("Ushbu kod orqali istalgan Python dasturingizda datasetni avtomatik bog'lab ishlatishingiz mumkin.")

