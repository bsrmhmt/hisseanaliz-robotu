import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import time

# --- Sayfa Ayarları ---
st.set_page_config(page_title="AI Finans Platformu V7", layout="wide", initial_sidebar_state="collapsed")

# --- Session State ---
if 'basladi' not in st.session_state:
    st.session_state['basladi'] = False

def baslat():
    st.session_state['basladi'] = True

# --- MOTOR (Hesaplama Fonksiyonları) ---

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

def veri_getir(sembol, periyot):
    try:
        p = "2y" if periyot == "1y" else periyot
        df = yf.Ticker(sembol).history(period=p)
        if len(df) < 50: return pd.DataFrame()
        return df
    except:
        return pd.DataFrame()

def temel_analiz_verisi(ticker_obj):
    try:
        info = ticker_obj.info
        fk = info.get('trailingPE', 0)
        pb = info.get('priceToBook', 0)
        return fk, pb
    except:
        return 0, 0

def indikatorler(df):
    df = df.copy()
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Ortalamalar & Bollinger
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    df['Std'] = df['Close'].rolling(window=20).std()
    df['BB_Up'] = df['SMA_20'] + (df['Std']*2)
    df['BB_Low'] = df['SMA_20'] - (df['Std']*2)
    
    # ATR (Volatilite)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()

    # YENİ: OBV (On Balance Volume - Hacim Dengesi)
    # Hacmin fiyatı destekleyip desteklemediğini ölçer
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    df.dropna(inplace=True)
    return df

def destek_direnc_bul(df):
    df['Min'] = df['Low'][(df['Low'].shift(1) > df['Low']) & (df['Low'].shift(-1) > df['Low'])]
    df['Max'] = df['High'][(df['High'].shift(1) < df['High']) & (df['High'].shift(-1) < df['High'])]
    son_donem = df.iloc[-60:]
    direncler = son_donem['Max'].dropna().unique().tolist()
    destekler = son_donem['Min'].dropna().unique().tolist()
    direncler.sort(reverse=True)
    destekler.sort()
    return destekler[:2], direncler[:2]

# --- YENİ: AI SKORLAMA MOTORU ---
def ai_skor_hesapla(row, fk, trend_yonu):
    puan = 50 # Başlangıç puanı
    
    # 1. Trend Analizi (+/- 20 Puan)
    if trend_yonu == "YUKARI": puan += 20
    else: puan -= 20
        
    # 2. RSI Analizi (+/- 15 Puan)
    rsi = row['RSI']
    if 40 < rsi < 65: puan += 15 # Sağlıklı bölge
    elif rsi > 75: puan -= 10 # Aşırı şişmiş
    elif rsi < 30: puan += 10 # Tepki potansiyeli (Ucuz)
        
    # 3. Temel Analiz (+/- 15 Puan)
    if 0 < fk < 10: puan += 15
    elif fk > 35: puan -= 10
        
    # Skor Sınırla (0-100 arası)
    if puan > 100: puan = 100
    if puan < 0: puan = 0
    
    # Renk ve Mesaj Belirle
    renk = "grey"
    if puan >= 75: renk = "green"
    elif puan <= 40: renk = "red"
    else: renk = "orange"
    
    return puan, renk

def karakter_analizi_yap(row, fk, trend_yonu):
    rsi = row['RSI']
    atr_yuzde = (row['ATR'] / row['Close']) * 100
    yorumlar = {"sabirli": [], "risk_sever": [], "temelci": []}
    
    if trend_yonu == "YUKARI": yorumlar["sabirli"].append("✅ Ana trend pozitif (Boğa piyasası).")
    else: yorumlar["sabirli"].append("⚠️ Ana trend negatif (Ayı piyasası).")
        
    if atr_yuzde > 3: yorumlar["risk_sever"].append(f"🔥 Yüksek volatilite (%{atr_yuzde:.1f}). Trade fırsatı.")
    else: yorumlar["risk_sever"].append("💤 Düşük volatilite. Yatay piyasa.")
        
    if 0 < fk < 10: yorumlar["temelci"].append(f"💎 F/K ({fk:.2f}) makul seviyede.")
    
    return yorumlar

# --- ARAYÜZ ---

# 1. LANDING PAGE
if not st.session_state['basladi']:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center;'>🧠 AI Finans V7</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>Yapay Zeka + Hacim Analizi + Skorlama</h3>", unsafe_allow_html=True)
        st.write("")
        st.button("🚀 TERMİNALİ BAŞLAT", on_click=baslat, use_container_width=True)

# 2. ANALİZ EKRANI
else:
    # Üst Bar
    st.markdown("### 🔎 Hisse Analiz Terminali")
    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        search_query = st.text_input("Hisse Kodu (Örn: THYAO, SASA)", value="THYAO")
    with col_s2:
        st.write("")
        st.write("")
        if st.button("Analiz Et", use_container_width=True): st.rerun()

    # Sidebar
    st.sidebar.header("Ayarlar")
    periyot = st.sidebar.selectbox("Geçmiş:", ["1y", "2y", "5y"], index=1)
    canli_mod = st.sidebar.checkbox("Canlı Yenile (60sn)", value=False)
    if st.sidebar.button("⬅️ Çıkış"):
        st.session_state['basladi'] = False
        st.rerun()
    st.markdown("---")

    hisseler = hisse_kodu_duzelt(search_query)

    if not hisseler:
        st.info("Hisse kodu bekleniyor...")
    else:
        tabs = st.tabs([s.replace(".IS", "") for s in hisseler])
        
        for i, sembol in enumerate(hisseler):
            with tabs[i]:
                with st.spinner(f'{sembol} analiz ediliyor...'):
                    ticker = yf.Ticker(sembol)
                    df = veri_getir(sembol, periyot)
                    
                    if df.empty:
                        st.error("Veri yok.")
                        continue
                        
                    fk, pb = temel_analiz_verisi(ticker)
                    df = indikatorler(df)
                    son_veri = df.iloc[-1]
                    guncel = son_veri['Close']
                    destekler, direncler = destek_direnc_bul(df)
                    
                    # Trend ve Skor
                    trend = "YUKARI" if guncel > son_veri['SMA_200'] else "AŞAĞI"
                    skor, skor_renk = ai_skor_hesapla(son_veri, fk, trend)
                    karakter = karakter_analizi_yap(son_veri, fk, trend)

                    # --- SKOR KARTI ---
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Fiyat", f"{guncel:.2f} TL", f"%{((guncel-df['Close'].iloc[-2])/df['Close'].iloc[-2])*100:.2f}")
                    
                    # Skor Göstergesi (Progress Bar gibi)
                    c2.metric("AI Skor (0-100)", f"{skor}/100")
                    if skor_renk == "green": c2.success("GÜÇLÜ GÖRÜNÜM")
                    elif skor_renk == "red": c2.error("ZAYIF GÖRÜNÜM")
                    else: c2.warning("NÖTR / İZLE")
                        
                    c3.metric("Trend (200G)", trend, delta_color="normal" if trend=="YUKARI" else "inverse")
                    
                    # Hacim Yorumu
                    hacim_durumu = "Normal"
                    if son_veri['OBV'] > df['OBV'].mean(): hacim_durumu = "