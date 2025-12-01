import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import time

# --- Sayfa Ayarları ---
st.set_page_config(page_title="AI Finans V9 (Konuşan Asistan)", layout="wide", initial_sidebar_state="collapsed")

# --- Session State ---
if 'basladi' not in st.session_state:
    st.session_state['basladi'] = False

def baslat():
    st.session_state['basladi'] = True

# --- MOTOR (Hesaplama) ---

def hisse_kodu_duzelt(text):
    temiz_liste = []
    if not text: return []
    text = text.replace(" ", "")
    ham_kodlar = text.split(",")
    for kod in ham_kodlar:
        kod = kod.upper()
        if not kod.endswith(".IS") and len(kod) > 2: kod += ".IS"
        if len(kod) > 3: temiz_liste.append(kod)
    return temiz_liste

def veri_getir(sembol):
    try:
        # ML ve Yorumlama için veri çekiyoruz
        df = yf.Ticker(sembol).history(period="2y") 
        if len(df) < 50: return pd.DataFrame()
        return df
    except:
        return pd.DataFrame()

def indikatorler(df):
    df = df.copy()
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Ortalamalar
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Bollinger
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Std'] = df['Close'].rolling(window=20).std()
    df['BB_Up'] = df['SMA_20'] + (df['Std']*2)
    df['BB_Low'] = df['SMA_20'] - (df['Std']*2)
    
    # Target (Hedef)
    df['Target'] = df['Close'].shift(-1)
    
    df.dropna(inplace=True)
    return df

# --- BASİT ML MODELİ ---
def model_egit(df):
    features = ['RSI', 'SMA_50', 'SMA_200', 'BB_Up', 'BB_Low']
    X = df[features]
    y = df['Target']
    
    X_train = X[:-1]
    y_train = y[:-1]
    X_today = X.tail(1)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    tahmin = model.predict(X_today)[0]
    
    return tahmin

# --- YENİ: KONUŞAN AI MOTORU (NLP) ---
def akilli_yorum_yap(row, trend_yonu, seviye):
    rsi = row['RSI']
    fiyat = row['Close']
    bb_low = row['BB_Low']
    bb_up = row['BB_Up']
    
    yorum = ""
    
    # --- SEVİYE 1: ACEMİ (Halk Dili) ---
    if seviye == "Acemi / Yeni Başlayan":
        yorum += "👋 **Selam! Basitçe anlatayım:**\n\n"
        
        if trend_yonu == "YUKARI":
            yorum += "🚀 **Genel Hava:** Rüzgar arkamızdan esiyor! Hisse genel olarak yükseliş trendinde, yani işler yolunda görünüyor.\n\n"
        else:
            yorum += "☔ **Genel Hava:** Hava biraz kapalı. Hisse düşüş trendinde, yani yokuş aşağı iniyor. Dikkatli olmak lazım.\n\n"
            
        if rsi < 30:
            yorum += "🛒 **Fırsat:** Hisse şu an 'İndirim Reyonunda' gibi! Fiyatı çok ucuzlamış, buralardan tepki verip yükselebilir.\n\n"
        elif rsi > 70:
            yorum += "🔥 **Uyarı:** Fiyat çok ısınmış, motor su kaynatabilir! Herkes alıyor diye gaza gelme, biraz düşmesini bekleyebilirsin.\n\n"
        else:
            yorum += "😐 **Durum:** Ne çok ucuz, ne çok pahalı. Tam ortada. Biraz izleyelim.\n\n"
            
        if fiyat < bb_low:
            yorum += "💡 **İpucu:** Fiyat normal sınırların altına sarkmış, lastik gibi geri fırlayabilir."

    # --- SEVİYE 2: ORTA DÜZEY (Bilinçli Yatırımcı) ---
    elif seviye == "Orta Düzey / Bilgili":
        yorum += "📊 **Teknik Özet:**\n\n"
        
        if trend_yonu == "YUKARI":
            yorum += "✅ **Trend:** Fiyat 200 günlük ortalamanın üzerinde. 'Boğa Piyasası' hakimiyeti sürüyor. Düşüşler alım fırsatı olabilir.\n\n"
        else:
            yorum += "❌ **Trend:** Fiyat 200 günlük ortalamanın altında. 'Ayı Piyasası' baskısı var. Trend dönmeden işlem açmak riskli.\n\n"
            
        if rsi < 30:
            yorum += "🟢 **Osilatör:** RSI 30 seviyesinin altında (Aşırı Satım). Bu bölge genellikle dip oluşumuna işaret eder.\n\n"
        elif rsi > 70:
            yorum += "🔴 **Osilatör:** RSI 70 seviyesinin üzerinde (Aşırı Alım). Kâr realizasyonu (satış) gelme ihtimali artıyor.\n\n"
            
        if fiyat > bb_up:
            yorum += "⚠️ **Volatilite:** Bollinger üst bandı delindi. Fiyat banda geri dönmek isteyecektir."

    # --- SEVİYE 3: PROFESYONEL (Trader / Analist) ---
    else:
        yorum += "📈 **Profesyonel Analiz Raporu:**\n\n"
        
        momentum = "Bullish" if trend_yonu == "YUKARI" else "Bearish"
        yorum += f"🔹 **Market Structure:** Ana trend {momentum} yapıda devam ediyor (Price > SMA200). \n\n"
        
        if rsi < 30:
            yorum += f"🔹 **Momentum:** RSI({rsi:.2f}) Aşırı Satım bölgesinde. Potansiyel bir 'Mean Reversion' (Ortalamaya Dönüş) veya 'Trend Reversal' sinyali aranmalı.\n\n"
        elif rsi > 70:
            yorum += f"🔹 **Momentum:** RSI({rsi:.2f}) Aşırı Alım bölgesinde. Long pozisyonlarda 'Stop-Loss' seviyeleri yukarı çekilmeli veya realizasyon düşünülmeli.\n\n"
        else:
            yorum += f"🔹 **Momentum:** RSI({rsi:.2f}) nötr bölgede konsolide oluyor. Kırılım yönü izlenmeli.\n\n"
            
        if fiyat < bb_low:
            yorum += "🔹 **İstatistik:** Fiyat -2 Standart Sapma bandının dışına taştı. İstatistiksel olarak içeri dönüş (Pullback) olasılığı %95'tir."

    return yorum

# --- ARAYÜZ ---

# 1. LANDING PAGE
if not st.session_state['basladi']:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center;'>🤖 AI Finans V9</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>Sizin Dilinizden Konuşan Asistan</h3>", unsafe_allow_html=True)
        st.write("")
        st.button("🚀 ASİSTANI BAŞLAT", on_click=baslat, use_container_width=True)

# 2. ANALİZ EKRANI
else:
    st.markdown("### 🔎 Akıllı Analiz Asistanı")
    
    # Üst Bar
    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        search_query = st.text_input("Hisse Kodu (Örn: THYAO)", value="THYAO")
    with col_s2:
        st.write("") 
        st.write("")
        if st.button("Analiz Et", use_container_width=True): st.rerun()

    # --- KENAR ÇUBUĞU: AI AYARLARI ---
    st.sidebar.header("🤖 AI Kişiliği")
    st.sidebar.info("Yapay zekanın size hangi dilde hitap etmesini istersiniz?")
    
    # BURASI YENİ ÖZELLİK:
    ai_seviye = st.sidebar.radio(
        "Anlatım Dili Seçin:",
        ("Acemi / Yeni Başlayan", "Orta Düzey / Bilgili", "Profesyonel / Trader")
    )
    
    st.sidebar.markdown("---")
    if st.sidebar.button("⬅️ Çıkış"):
        st.session_state['basladi'] = False
        st.rerun()

    # --- AKIŞ ---
    hisseler = hisse_kodu_duzelt(search_query)

    if not hisseler:
        st.info("Lütfen bir hisse kodu girin...")
    else:
        tabs = st.tabs([s.replace(".IS", "") for s in hisseler])
        
        for i, sembol in enumerate(hisseler):
            with tabs[i]:
                with st.spinner('Yapay zeka verileri yorumluyor...'):
                    df = veri_getir(sembol)
                    
                    if df.empty:
                        st.error("Veri yok.")
                        continue
                        
                    df = indikatorler(df)
                    tahmin = model_egit(df)
                    
                    son_veri = df.iloc[-1]
                    guncel = son_veri['Close']
                    trend = "YUKARI" if guncel > son_veri['SMA_200'] else "AŞAĞI"
                    
                    # --- AI KONUŞUYOR ---
                    ai_yorumu = akilli_yorum_yap(son_veri, trend, ai_seviye)

                    # Görsel Düzen
                    c1, c2 = st.columns([2, 1])
                    
                    with c1:
                        # Grafik
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Fiyat'))
                        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=1), name='50 G.Ort'))
                        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='blue', width=2), name='200 G.Ort'))
                        fig.update_layout(height=400, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=30, b=0))
                        st.plotly_chart(fig, use_container_width=True)
                        
                    with c2:
                        # AI Sohbet Kutusu
                        st.subheader(f"💬 AI Asistan ({ai_seviye})")
                        st.info(ai_yorumu)
                        
                        st.metric("AI Hedef Fiyat", f"{tahmin:.2f} TL", f"%{((tahmin-guncel)/guncel)*100:.2f}")
