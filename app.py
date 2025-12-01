import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import time

# --- Sayfa Ayarları ---
st.set_page_config(page_title="AI Finans V8 (Self-Learning)", layout="wide", initial_sidebar_state="collapsed")

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

def veri_getir(sembol, periyot):
    try:
        # ML için daha fazla veriye ihtiyacımız var, 'max' veya '5y' zorluyoruz
        df = yf.Ticker(sembol).history(period="5y") 
        if len(df) < 100: return pd.DataFrame()
        return df
    except:
        return pd.DataFrame()

def temel_analiz_verisi(ticker_obj):
    try:
        info = ticker_obj.info
        fk = info.get('trailingPE', 0)
        return fk
    except:
        return 0

def indikatorler_ve_ozellikler(df):
    """
    ML modelinin öğrenmesi için gelişmiş özellikler (Features) oluşturur.
    """
    df = df.copy()
    # 1. Klasik İndikatörler
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    # Bollinger
    df['Std'] = df['Close'].rolling(window=20).std()
    df['BB_Up'] = df['SMA_20'] + (df['Std']*2)
    df['BB_Low'] = df['SMA_20'] - (df['Std']*2)
    
    # OBV
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # 2. YENİ: Lag Features (Geçmişin Hafızası)
    # Model sadece bugüne bakmasın, dünü ve önceki günü de bilsin.
    df['Close_Lag1'] = df['Close'].shift(1) # Dünkü kapanış
    df['Close_Lag2'] = df['Close'].shift(2) # Önceki gün
    df['RSI_Lag1'] = df['RSI'].shift(1)     # Dünkü RSI
    
    # 3. YENİ: Target (Hedef) - Yarınki Fiyat
    df['Target'] = df['Close'].shift(-1)
    
    df.dropna(inplace=True)
    return df

# --- YENİ: AKILLI MODEL EĞİTİMİ (SELF-OPTIMIZING) ---
def akilli_model_egit(df):
    """
    Farklı zeka seviyelerini deneyip en az hata yapanı seçen fonksiyon.
    """
    features = ['RSI', 'SMA_20', 'SMA_50', 'ATR', 'OBV', 'Close_Lag1', 'Close_Lag2', 'RSI_Lag1']
    X = df[features]
    y = df['Target']
    
    # Son satır (Bugün) tahmin için ayrılır, gerisi eğitim için
    X_train_full = X[:-1]
    y_train_full = y[:-1]
    X_today = X.tail(1)
    
    # Eğitim ve Test seti ayırma (Modelin başarısını ölçmek için)
    X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42, shuffle=False)
    
    # HİPERPARAMETRE OPTİMİZASYONU (Grid Search Simülasyonu)
    # Model farklı 'beyin yapılarını' deniyor
    parametreler = [
        {'n_estimators': 50, 'max_depth': 5},   # Hızlı ve Basit Düşünen Model
        {'n_estimators': 100, 'max_depth': 10}, # Dengeli Model
        {'n_estimators': 200, 'max_depth': 20}  # Derinlemesine Düşünen Model
    ]
    
    en_iyi_model = None
    en_dusuk_hata = float('inf')
    secilen_param = ""
    
    for param in parametreler:
        model = RandomForestRegressor(n_estimators=param['n_estimators'], max_depth=param['max_depth'], random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        hata = mean_absolute_error(y_test, preds)
        
        if hata < en_dusuk_hata:
            en_dusuk_hata = hata
            en_iyi_model = model
            secilen_param = f"Ağaç: {param['n_estimators']} | Derinlik: {param['max_depth']}"
            
    # Kazanan model ile tüm veriyi eğit ve yarını tahmin et
    en_iyi_model.fit(X_train_full, y_train_full)
    tahmin = en_iyi_model.predict(X_today)[0]
    
    # Başarı Skoru (R2 benzeri basit doğruluk)
    # Hata payı fiyata göre yüzde kaç?
    son_fiyat = df['Close'].iloc[-2]
    hata_yuzdesi = (en_dusuk_hata / son_fiyat) * 100
    dogruluk_skoru = 100 - hata_yuzdesi
    
    return tahmin, dogruluk_skoru, secilen_param, en_dusuk_hata

# --- SKORLAMA VE ANALİZ FONKSİYONLARI ---
def destek_direnc_bul(df):
    df['Min'] = df['Low'][(df['Low'].shift(1) > df['Low']) & (df['Low'].shift(-1) > df['Low'])]
    df['Max'] = df['High'][(df['High'].shift(1) < df['High']) & (df['High'].shift(-1) < df['High'])]
    son_donem = df.iloc[-60:]
    direncler = son_donem['Max'].dropna().unique().tolist()
    destekler = son_donem['Min'].dropna().unique().tolist()
    direncler.sort(reverse=True)
    destekler.sort()
    return destekler[:2], direncler[:2]

def ai_skor_hesapla(row, fk, trend_yonu, ml_dogruluk):
    puan = 50 
    if trend_yonu == "YUKARI": puan += 15
    else: puan -= 15
    rsi = row['RSI']
    if 40 < rsi < 65: puan += 10 
    elif rsi > 75: puan -= 10 
    elif rsi < 30: puan += 15
    if 0 < fk < 10: puan += 10
    
    # ML Güveni Ekliyoruz: Eğer model kendine çok güveniyorsa (Doğruluk yüksekse) puanı etkile
    if ml_dogruluk > 98: puan += 10
    
    if puan > 100: puan = 100
    if puan < 0: puan = 0
    
    renk = "grey"
    if puan >= 75: renk = "green"
    elif puan <= 40: renk = "red"
    else: renk = "orange"
    return puan, renk

# --- ARAYÜZ ---
if not st.session_state['basladi']:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center;'>🧠 AI Finans V8</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>Self-Optimizing (Kendi Kendini Eğiten) Model</h3>", unsafe_allow_html=True)
        st.write("")
        st.info("Bu sürümde Yapay Zeka, her hisse için farklı parametreleri deneyerek en az hata yapan stratejiyi otomatik seçer.")
        st.button("🚀 SİSTEMİ BAŞLAT", on_click=baslat, use_container_width=True)

else:
    st.markdown("### 🔎 Self-Learning Analiz Terminali")
    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        search_query = st.text_input("Hisse Kodu (Örn: THYAO, ASELS)", value="THYAO")
    with col_s2:
        st.write("")
        st.write("")
        if st.button("Analiz Et", use_container_width=True): st.rerun()

    st.sidebar.header("Ayarlar")
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
                with st.spinner(f'{sembol} için en uygun model eğitiliyor ve test ediliyor...'):
                    ticker = yf.Ticker(sembol)
                    df = veri_getir(sembol, "5y") # ML için uzun veri
                    
                    if df.empty:
                        st.error("Veri yok.")
                        continue
                        
                    fk = temel_analiz_verisi(ticker)
                    df = indikatorler_ve_ozellikler(df)
                    
                    # --- MACHINE LEARNING MOTORU ÇALIŞIYOR ---
                    tahmin, dogruluk, model_params, hata_payi = akilli_model_egit(df)
                    
                    son_veri = df.iloc[-1]
                    guncel = son_veri['Close']
                    destekler, direncler = destek_direnc_bul(df)
                    trend = "YUKARI" if guncel > son_veri['SMA_200'] else "AŞAĞI"
                    skor, skor_renk = ai_skor_hesapla(son_veri, fk, trend, dogruluk)

                    # --- GÖSTERGELER ---
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Anlık Fiyat", f"{guncel:.2f} TL")
                    
                    # ML Sonuçları
                    c2.metric("AI Tahmin (T+1)", f"{tahmin:.2f} TL", f"%{((tahmin-guncel)/guncel)*100:.2f}")
                    
                    c3.metric("Model Güveni (Accuracy)", f"%{dogruluk:.2f}", f"Hata Payı: {hata_payi:.2f} TL")
                    
                    c4.metric("AI Skor", f"{skor}/100")
                    
                    # Model Detayı (Expander)
                    with st.expander(f"🧠 {sembol} İçin Seçilen En İyi Modelin Detayları"):
                        st.write(f"**Optimize Edilen Parametreler:** {model_params}")
                        st.write(f"**Eğitimdeki Ortalama Hata (MAE):** {hata_payi:.2f} TL")
                        st.write("Sistem 3 farklı algoritma karmaşıklığını test etti ve bu hissenin karakterine en uygun olanı seçti.")

                    # --- GRAFİK ---
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Fiyat'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=1), name='50 G.Ort'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='blue', width=2), name='200 G.Ort'))
                    
                    # Tahmin Noktası (Gelecek)
                    last_date = df.index[-1]
                    # Basit bir timedelta ekleme (Hafta sonu hatası olmasın diye +1 gün diyoruz ama grafikte sadece nokta göstereceğiz)
                    fig.add_trace(go.Scatter(x=[last_date], y=[tahmin], mode='markers', marker=dict(color='purple', size=15, symbol='star'), name='AI Tahmin Hedefi'))

                    for d in direncler:
                        if d > guncel * 0.95: fig.add_hline(y=d, line_dash="dash", line_color="red", annotation_text="Direnç")
                    for s in destekler:
                        if s < guncel * 1.05: fig.add_hline(y=s, line_dash="dash", line_color="green", annotation_text="Destek")

                    fig.update_layout(height=550, xaxis_rangeslider_visible=False, title=f"{sembol} AI Analiz Grafiği")
                    st.plotly_chart(fig, use_container_width=True)

    if canli_mod:
        time.sleep(60)
        st.rerun()
