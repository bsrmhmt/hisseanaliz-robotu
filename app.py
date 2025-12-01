import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import time

# --- Sayfa Ayarları (En Üstte Olmalı) ---
st.set_page_config(page_title="AI Finans Platformu V6", layout="wide", initial_sidebar_state="collapsed")

# --- Session State (Oturum Durumu) ---
# Başla butonuna basılıp basılmadığını kontrol eder
if 'basladi' not in st.session_state:
    st.session_state['basladi'] = False

def baslat():
    st.session_state['basladi'] = True

# --- YARDIMCI FONKSİYONLAR (Hesaplama Motoru) ---

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
    
    # SMA & Bollinger
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
    
    df.dropna(inplace=True)
    return df

# --- YENİ: Otomatik Destek/Direnç Tespiti ---
def destek_direnc_bul(df, window=20):
    """Yerel tepe ve dipleri bularak destek direnç belirler"""
    df['Min'] = df['Low'][(df['Low'].shift(1) > df['Low']) & (df['Low'].shift(-1) > df['Low'])]
    df['Max'] = df['High'][(df['High'].shift(1) < df['High']) & (df['High'].shift(-1) < df['High'])]
    
    # Son 60 gündeki en belirgin seviyeleri al
    son_donem = df.iloc[-60:]
    direncler = son_donem['Max'].dropna().unique().tolist()
    destekler = son_donem['Min'].dropna().unique().tolist()
    
    # Birbirine çok yakın seviyeleri temizle (Basitçe)
    direncler.sort(reverse=True)
    destekler.sort()
    
    # En yakın 2 tanesini döndür
    return destekler[:2], direncler[:2]

# --- YENİ: Karakter Analiz Motoru ---
def karakter_analizi_yap(row, fk, trend_yonu):
    rsi = row['RSI']
    atr_yuzde = (row['ATR'] / row['Close']) * 100
    
    yorumlar = {
        "sabirli": [],
        "risk_sever": [],
        "temelci": []
    }
    
    # 1. Sabırlı Yatırımcı (Uzun Vadeci)
    if trend_yonu == "YUKARI":
        yorumlar["sabirli"].append("✅ Ana trend yukarı yönlü (Fiyat > 200 G.Ort). Pozisyon taşımaya uygun görünüyor.")
    else:
        yorumlar["sabirli"].append("⚠️ Ana trend henüz negatife dönmedi ama zayıflıyor. Acele etme, dönüş sinyali bekle.")
        
    if rsi < 40:
        yorumlar["sabirli"].append("✅ RSI soğumuş, kademeli alım için makul seviyeler olabilir.")
    
    # 2. Risk Sever Trader (Kısa Vadeci)
    if atr_yuzde > 3:
        yorumlar["risk_sever"].append(f"🔥 Volatilite yüksek (Günlük %{atr_yuzde:.1f} oynuyor). Tam senlik, hızlı al-sat fırsatları verebilir.")
    else:
        yorumlar["risk_sever"].append("💤 Hisse şu an çok sakin, sana göre değil. Hareketlenmesini bekle.")
        
    if rsi > 70:
        yorumlar["risk_sever"].append("⚠️ RSI aşırı şişmiş. Kısa vadeli bir 'Short' (Düşüş yönlü) işlem veya kâr satışı denenebilir.")
    elif rsi < 30:
        yorumlar["risk_sever"].append("🚀 RSI dipte. Tepki yükselişi için 'Long' (Alım yönlü) bir vur-kaç denenebilir.")

    # 3. Temel Analizci (Değer Yatırımcısı)
    if fk > 0 and fk < 8:
        yorumlar["temelci"].append(f"💎 F/K Oranı ({fk:.2f}) oldukça cazip. Şirket kârlılığına göre ucuz fiyatlanıyor.")
    elif fk > 30:
        yorumlar["temelci"].append(f"💸 F/K Oranı ({fk:.2f}) yüksek. Gelecek beklentileri çoktan satın alınmış olabilir, dikkatli ol.")
    else:
        yorumlar["temelci"].append(f"ℹ️ F/K Oranı ({fk:.2f}) sektör ortalamalarında makul görünüyor.")

    return yorumlar

# =========================================
# ARAYÜZ MİMARİSİ
# =========================================

# --- DURUM 1: BAŞLANGIÇ EKRANI (Landing Page) ---
if not st.session_state['basladi']:
    # Sayfayı ortala
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center; color: #0E1117;'>🧠 AI Finans Platformu</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; color: #262730;'>Yeni Nesil Borsa Analiz Asistanınız</h3>", unsafe_allow_html=True)
        st.write("")
        st.markdown("""
        <div style='text-align: center;'>
        Yapay zeka destekli teknik analizler, otomatik destek/direnç tespiti ve 
        kişiselleştirilmiş yatırımcı yorumları ile piyasalara profesyonel bir bakış atın.
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        st.write("")
        # BAŞLA BUTONU
        st.button("🚀 ANALİZE BAŞLA", on_click=baslat, use_container_width=True)

# --- DURUM 2: ANA ANALİZ EKRANI ---
else:
    # --- Üst Arama Çubuğu ---
    st.markdown("### 🔎 Hisse Senedi Arayın")
    col_search1, col_search2 = st.columns([3, 1])
    with col_search1:
        search_query = st.text_input("BIST Kodu Girin (Örn: THYAO, ASELS, EREGL)", value="THYAO, EREGL")
    with col_search2:
        st.write("") # Boşluk
        st.write("")
        if st.button("Analiz Et", use_container_width=True):
            st.rerun()

    # --- Kenar Çubuğu (Sadece Ayarlar Kaldı) ---
    st.sidebar.header("⚙️ Ayarlar")
    periyot = st.sidebar.selectbox("Veri Geçmişi:", ["1y", "2y", "5y"], index=1)
    canli_mod = st.sidebar.checkbox("Canlı Yenileme (60sn)", value=False)
    st.sidebar.info("Not: Trader çizgileri son 60 günün tepe/diplerine göre otomatik çizilir.")
    if st.sidebar.button("⬅️ Ana Ekrana Dön"):
        st.session_state['basladi'] = False
        st.rerun()

    st.markdown("---")

    # --- Ana Akış ---
    hisseler = hisse_kodu_duzelt(search_query)

    if not hisseler:
        st.info("Lütfen yukarıdaki arama çubuğuna bir hisse kodu yazın.")
    else:
        # Sekmeler
        tabs = st.tabs([s.replace(".IS", "") for s in hisseler])
        
        for i, sembol in enumerate(hisseler):
            with tabs[i]:
                with st.spinner(f'{sembol} verileri işleniyor ve çizgiler çiziliyor...'):
                    ticker = yf.Ticker(sembol)
                    df = veri_getir(sembol, periyot)
                    
                    if df.empty:
                        st.error("Veri bulunamadı.")
                        continue
                        
                    # Veri Hazırlığı
                    fk, pb = temel_analiz_verisi(ticker)
                    df = indikatorler(df)
                    son_veri = df.iloc[-1]
                    guncel_fiyat = son_veri['Close']
                    
                    # Destek/Direnç Hesapla
                    destekler, direncler = destek_direnc_bul(df)
                    
                    # Trend Yönü Belirle
                    trend_yonu = "NÖTR"
                    if guncel_fiyat > son_veri['SMA_200']: trend_yonu = "YUKARI"
                    elif guncel_fiyat < son_veri['SMA_200']: trend_yonu = "AŞAĞI"
                    
                    # Karakter Analizi Yap
                    karakter_yorumlari = karakter_analizi_yap(son_veri, fk, trend_yonu)

                    # --- ÜST BİLGİ KARTLARI ---
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Fiyat", f"{guncel_fiyat:.2f} TL", f"%{((guncel_fiyat - df['Close'].iloc[-2])/df['Close'].iloc[-2])*100:.2f}")
                    c2.metric("RSI (Güç)", f"{son_veri['RSI']:.1f}", "30 Altı Ucuz / 70 Üstü Pahalı")
                    c3.metric("F/K Oranı", f"{fk:.2f}" if fk>0 else "-", "Temel Değerleme")
                    c4.metric("Ana Trend (200G)", trend_yonu, delta_color="normal" if trend_yonu=="YUKARI" else "inverse")

                    # --- PROFESYONEL GRAFİK (Çizgili) ---
                    fig = go.Figure()
                    
                    # Mum Grafiği
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Fiyat'))
                    
                    # Ortalamalar
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=1), name='50 G.Ort (Orta Vade)'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='blue', width=2), name='200 G.Ort (Ana Trend)'))
                    
                    # Bollinger Bantları (Gölge)
                    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Up'], line=dict(color='gray', width=0), showlegend=False, name='BB Üst'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], line=dict(color='gray', width=0), fill='tonexty', fillcolor='rgba(128,128,128,0.1)', showlegend=False, name='BB Alt'))
                    
                    # --- OTOMATİK TRADER ÇİZGİLERİ ---
                    # Dirençler (Kırmızı Kesikli)
                    for direnc in direncler:
                        if direnc > guncel_fiyat * 0.95: # Çok alttakileri çizme
                             fig.add_hline(y=direnc, line_dash="dash", line_color="red", annotation_text=f"Direnç: {direnc:.2f}", annotation_position="top right")
                    
                    # Destekler (Yeşil Kesikli)
                    for destek in destekler:
                        if destek < guncel_fiyat * 1.05: # Çok üsttekileri çizme
                            fig.add_hline(y=destek, line_dash="dash", line_color="green", annotation_text=f"Destek: {destek:.2f}", annotation_position="bottom right")

                    fig.update_layout(height=500, xaxis_rangeslider_visible=False, title=f"{sembol} Teknik Analiz ve Trader Seviyeleri")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write("---")
                    st.subheader("🧠 Kişiselleştirilmiş Yatırımcı Analizleri")
                    st.write("Hangi profile uygunsanız, o başlığa tıklayarak size özel yorumu okuyun.")

                    # --- KARAKTER ANALİZLERİ (Expanders) ---
                    
                    with st.expander("🧘🏻‍♂️ Sabırlı / Uzun Vadeci Yatırımcı (Tıkla)"):
                        st.markdown("Bu profil; kısa vadeli dalgalanmalara takılmayan, ana trendi ve temel verileri önemseyenler içindir.")
                        for yorum in karakter_yorumlari["sabirli"]:
                            st.write(f"- {yorum}")

                    with st.expander("🎢 Risk Sever / Kısa Vadeci Trader (Tıkla)"):
                        st.markdown("Bu profil; volatiliteyi seven, hızlı al-sat yapan ve RSI gibi momentum göstergelerine bakanlar içindir.")
                        for yorum in karakter_yorumlari["risk_sever"]:
                            st.write(f"- {yorum}")
                            
                    with st.expander("💎 Temel Analizci / Değer Yatırımcısı (Tıkla)"):
                        st.markdown("Bu profil; grafikten çok şirketin kârlılığına ve ucuzluğuna (F/K, PD/DD) odaklananlar içindir.")
                        for yorum in karakter_yorumlari["temelci"]:
                            st.write(f"- {yorum}")
    
    # Canlı Döngü
    if canli_mod:
        time.sleep(60)
        st.rerun()