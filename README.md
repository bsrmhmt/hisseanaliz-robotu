import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time # Zamanlayıcı için gerekli
from sklearn.ensemble import RandomForestRegressor

# --- Sayfa Ayarları ---
st.set_page_config(page_title="AI Trader V5.0 (Canlı)", layout="wide")
st.title("⚡ AI Trader V5.0: Canlı Otomatik Terminal")

# --- Kenar Çubuğu ve Ayarlar ---
st.sidebar.header("⚙️ Kontrol Merkezi")

# Canlı Mod Kutusu (En Üstte)
st.sidebar.subheader("🔴 Canlı Takip")
canli_mod = st.sidebar.checkbox("Otomatik Yenile (60sn)", value=False)
if canli_mod:
    placeholder = st.sidebar.empty() # Geri sayım sayacı yeri

# Hisseler ve Ayarlar
default_tickers = "thyao, garan, eregl, astor, sise, kchol"
user_input = st.sidebar.text_area("Hisse Listesi:", value=default_tickers)
periyot = st.sidebar.selectbox("Veri Geçmişi:", ["1y", "2y", "5y"], index=0)
bakiye_baslangic = st.sidebar.number_input("Sanal Bakiye:", value=100000, step=5000)

# --- Fonksiyonlar ---

def hisse_kodu_duzelt(text):
    temiz_liste = []
    text = text.replace(" ", "")
    ham_kodlar = text.split(",")
    for kod in ham_kodlar:
        kod = kod.upper()
        if not kod.endswith(".IS") and len(kod) > 2: kod += ".IS"
        if len(kod) > 3: temiz_liste.append(kod)
    return temiz_liste

def veri_getir(sembol, periyot):
    try:
        # Periyot düzeltmesi (1y seçilse bile hesaplama için 2y çekiyoruz)
        p = "2y" if periyot == "1y" else periyot
        df = yf.Ticker(sembol).history(period=p)
        if len(df) < 30: return pd.DataFrame()
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
    
    # SMA & Bollinger
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Std'] = df['Close'].rolling(window=20).std()
    df['BB_Up'] = df['SMA_20'] + (df['Std']*2)
    df['BB_Low'] = df['SMA_20'] - (df['Std']*2)
    
    # ATR (Stop Loss için)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    df.dropna(inplace=True)
    return df

def backtest_motoru(df, baslangic_para):
    bakiye = baslangic_para
    lot = 0
    islemler = []
    
    for i in range(len(df)-1):
        row = df.iloc[i]
        fiyat = row['Close']
        tarih = df.index[i]
        
        # Basit Strateji: RSI < 35 AL, RSI > 65 SAT
        if lot == 0 and row['RSI'] < 35:
            lot = int(bakiye / fiyat)
            bakiye -= lot * fiyat
            alis_fiyati = fiyat
            islemler.append({'Tarih': tarih, 'Tip': 'AL', 'Fiyat': fiyat})
            
        elif lot > 0:
            # Kar Al veya Stop Loss
            if row['RSI'] > 65 or fiyat < alis_fiyati * 0.95:
                bakiye += lot * fiyat
                kar_zarar = (fiyat - alis_fiyati) * lot
                tip = 'SAT (Kâr)' if kar_zarar > 0 else 'SAT (Zarar)'
                islemler.append({'Tarih': tarih, 'Tip': tip, 'Fiyat': fiyat})
                lot = 0
                
    if lot > 0: # Son gün elimizde mal varsa satıp nakite dönelim
        bakiye += lot * df['Close'].iloc[-1]
        
    return bakiye, islemler

# --- Ana Akış ---
hisseler = hisse_kodu_duzelt(user_input)

if not hisseler:
    st.warning("Lütfen hisse kodu girin.")
else:
    # Sekmeleri Oluştur
    tabs = st.tabs([s.replace(".IS", "") for s in hisseler])
    
    for i, sembol in enumerate(hisseler):
        with tabs[i]:
            df = veri_getir(sembol, periyot)
            if df.empty:
                st.error("Veri yok.")
                continue
                
            df = indikatorler(df)
            
            # AI Tahmin
            model = RandomForestRegressor(n_estimators=100)
            X = df[['RSI', 'SMA_20', 'BB_Up', 'BB_Low', 'ATR']]
            y = df['Close'].shift(-1)
            model.fit(X[:-1], y[:-1])
            tahmin = model.predict(X.tail(1))[0]
            guncel = df['Close'].iloc[-1]
            stop_loss = guncel - (df['ATR'].iloc[-1] * 2)
            
            # Backtest
            son_bakiye, islemler = backtest_motoru(df, bakiye_baslangic)
            getiri_yuzde = ((son_bakiye - bakiye_baslangic) / bakiye_baslangic) * 100
            
            # --- GÖSTERGELER ---
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Anlık Fiyat", f"{guncel:.2f} TL")
            c2.metric("AI Hedef", f"{tahmin:.2f} TL", f"%{((tahmin-guncel)/guncel)*100:.2f}")
            c3.metric("Simülasyon", f"{getiri_yuzde:.1f}% Getiri", f"{son_bakiye:.0f} TL")
            c4.metric("Risk (Stop)", f"{stop_loss:.2f} TL")
            
            # --- GRAFİK ---
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Fiyat'))
            
            # Al-Sat İşaretleri
            al_tarih = [x['Tarih'] for x in islemler if 'AL' in x['Tip']]
            al_fiyat = [x['Fiyat'] for x in islemler if 'AL' in x['Tip']]
            sat_tarih = [x['Tarih'] for x in islemler if 'SAT' in x['Tip']]
            sat_fiyat = [x['Fiyat'] for x in islemler if 'SAT' in x['Tip']]
            
            fig.add_trace(go.Scatter(x=al_tarih, y=al_fiyat, mode='markers', marker=dict(color='green', size=12, symbol='triangle-up'), name='AL Sinyali'))
            fig.add_trace(go.Scatter(x=sat_tarih, y=sat_fiyat, mode='markers', marker=dict(color='red', size=12, symbol='triangle-down'), name='SAT Sinyali'))
            
            fig.update_layout(height=450, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

# --- CANLI DÖNGÜ (EN ALTTA OLMALI) ---
if canli_mod:
    # Geri sayım
    for s in range(60, 0, -1):
        placeholder.metric("⏳ Yenilemeye Kalan", f"{s} sn")
        time.sleep(1)
    # Süre dolunca sayfayı yenile
    st.rerun()
