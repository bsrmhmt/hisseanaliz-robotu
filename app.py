import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from datetime import datetime
import time

# --- Sayfa Ayarları ---
st.set_page_config(page_title="AI Finans Pro", layout="wide", initial_sidebar_state="collapsed")

# --- Session State ---
if 'basladi' not in st.session_state: st.session_state['basladi'] = False
if 'favoriler' not in st.session_state: st.session_state['favoriler'] = []
if 'analiz_gecmisi' not in st.session_state: st.session_state['analiz_gecmisi'] = []

def navigate_to(page):
    st.session_state['page'] = page
    st.rerun()

# ==========================================
# 1. MOTOR BLOĞU (BACKEND)
# ==========================================

class AdvancedDataFetcher:
    def get_stock_data(self, sembol):
        try:
            ticker = yf.Ticker(sembol)
            # Teknik veri (Fiyatlar)
            df = ticker.history(period="2y")
            # Temel veri (Bilanço)
            info = ticker.info
            if len(df) < 50: return None
            return {'data': df, 'info': info, 'ticker': ticker}
        except: return None

# --- YENİ: TEMEL ANALİZ MOTORU (INVESTING PRO TARZI) ---
class FundamentalEngine:
    def calculate_fair_value(self, info):
        """
        Benjamin Graham Formülü ile Adil Değer Hesaplar.
        Adil Değer = Karekök(22.5 * EPS * Defter Değeri)
        """
        try:
            eps = info.get('trailingEps', 0)
            book_value = info.get('bookValue', 0)
            current_price = info.get('currentPrice', 0)
            
            if eps is None or book_value is None or eps <= 0 or book_value <= 0:
                return None, 0 # Hesaplaamzsa
            
            # Graham Formülü
            fair_value = np.sqrt(22.5 * eps * book_value)
            
            upside = ((fair_value - current_price) / current_price) * 100
            return fair_value, upside
        except:
            return None, 0

    def calculate_health_score(self, info):
        """
        Şirket Sağlık Puanı (0-5 Arası)
        Büyüme, Kârlılık ve Borç durumuna bakar.
        """
        score = 0
        try:
            # 1. Kârlılık (Profitability)
            if info.get('profitMargins', 0) > 0.10: score += 1
            if info.get('returnOnEquity', 0) > 0.15: score += 1
            
            # 2. Büyüme (Growth)
            if info.get('revenueGrowth', 0) > 0.10: score += 1
            
            # 3. Sağlamlık (Solvency)
            debt_to_equity = info.get('debtToEquity', 100)
            if debt_to_equity < 100: score += 1 # Düşük borç
            
            # 4. Nakit Durumu
            if info.get('quickRatio', 0) > 1: score += 1
            
            return score
        except:
            return 2 # Veri yoksa orta şeker

class AdvancedTechnicalAnalysis:
    def calculate_all_indicators(self, df):
        df = df.copy()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['BB_Upper'] = df['SMA_50'] + (df['Close'].rolling(window=20).std()*2)
        df['BB_Lower'] = df['SMA_50'] - (df['Close'].rolling(window=20).std()*2)
        df['Target'] = df['Close'].shift(-5)
        df.dropna(inplace=True)
        return df

class AdvancedStockPredictor:
    def predict_with_confidence(self, df, horizon=5):
        try:
            features = ['RSI', 'SMA_50', 'SMA_200', 'BB_Upper', 'BB_Lower', 'Volume']
            X = df[features]
            y = df['Target']
            X_train = X[:-horizon]; y_train = y[:-horizon]; X_today = X.tail(1)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            prediction = model.predict(X_today)[0]
            preds = model.predict(X_train[-50:])
            mae = mean_absolute_error(y_train[-50:], preds)
            return prediction, mae, {}
        except: return 0, 0, {}

class AdvancedAIAssistant:
    def generate_analysis(self, df, trend, rsi):
        analysis_text = f"Trend {trend} yönünde. "
        if rsi < 30: analysis_text += "Fiyatlar ucuz (Aşırı Satım)."
        elif rsi > 70: analysis_text += "Fiyatlar pahalı (Aşırı Alım)."
        return analysis_text

class StateManager:
    def add_to_history(self, sembol, data):
        if 'analiz_gecmisi' not in st.session_state: st.session_state['analiz_gecmisi'] = []
        st.session_state['analiz_gecmisi'].insert(0, {'sembol': sembol, **data})

state = StateManager()

# ==========================================
# 2. ARAYÜZ (FRONTEND)
# ==========================================

def show_landing_page():
    st.markdown("""
        <style>
        .stApp {background-color: #fff;}
        .hero {text-align: center; padding: 50px 20px;}
        .main-title {
            font-size: 4rem; font-weight: 800;
            background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .card {
            padding: 20px; border-radius: 15px; background: white;
            box-shadow: 0 10px 30px rgba(0,0,0,0.05); text-align: center; border: 1px solid #eee;
        }
        .start-btn button {
            background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
            color: white; padding: 15px 40px; font-size: 1.2rem; border-radius: 30px; border:none; width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="hero"><div class="main-title">AI Finans Pro</div><p>Temel ve Teknik Analiz Bir Arada</p></div>', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="card"><h3>📊 Adil Değer</h3><p>Hissenin gerçek ederi nedir?</p></div>', unsafe_allow_html=True)
    c2.markdown('<div class="card"><h3>🏥 Şirket Sağlığı</h3><p>Bilanço ne kadar sağlam?</p></div>', unsafe_allow_html=True)
    c3.markdown('<div class="card"><h3>🚀 AI Tahmin</h3><p>Gelecek fiyat hareketleri.</p></div>', unsafe_allow_html=True)
    
    st.write("")
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.markdown('<div class="start-btn">', unsafe_allow_html=True)
        if st.button("TERMİNALİ BAŞLAT"):
            st.session_state['basladi'] = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

def create_gauge_chart(value, title):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'red'},
                {'range': [30, 70], 'color': 'yellow'},
                {'range': [70, 100], 'color': 'green'}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': value}
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
    return fig

def analyze_stocks(hisseler):
    sembol = hisseler[0]
    fetcher = AdvancedDataFetcher()
    data = fetcher.get_stock_data(sembol)
    
    if not data:
        st.error("Veri bulunamadı.")
        return
    
    df = data['data']
    info = data['info']
    
    # Motorları Çalıştır
    ta_engine = AdvancedTechnicalAnalysis()
    fund_engine = FundamentalEngine()
    
    df_tech = ta_engine.calculate_all_indicators(df)
    fair_value, upside = fund_engine.calculate_fair_value(info)
    health_score = fund_engine.calculate_health_score(info)
    
    current_price = df_tech['Close'].iloc[-1]
    rsi = df_tech['RSI'].iloc[-1]
    
    # --- YENİ: INVESTING PRO TARZI DASHBOARD ---
    st.markdown(f"## 🏆 {info.get('longName', sembol)} Analizi")
    
    # 1. TEMEL ANALİZ KARTLARI (Fair Value & Health)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Adil Değer Kartı
        st.markdown("### 💰 Adil Değer (Fair Value)")
        if fair_value:
            delta_color = "normal" if upside > 0 else "inverse"
            st.metric("Hesaplanan Gerçek Değer", f"{fair_value:.2f} TL", f"{upside:+.2f}% Potansiyel", delta_color=delta_color)
            if upside > 20: st.success("✅ HİSSE ÇOK UCUZ (Undervalued)")
            elif upside < -20: st.error("❌ HİSSE PAHALI (Overvalued)")
            else: st.warning("⚖️ HİSSE EDERİNDE (Fair)")
        else:
            st.info("Bu şirket için Adil Değer hesaplanamıyor (Zarar ediyor olabilir).")

    with col2:
        # Şirket Sağlığı Kartı
        st.markdown("### 🏥 Şirket Sağlığı")
        
        # Sağlık Progress Bar'ları
        health_labels = ["Zayıf", "Orta", "İyi", "Çok İyi", "Mükemmel"]
        health_text = health_labels[min(health_score, 4)]
        
        # Renkli Bar
        bar_color = "red" if health_score < 2 else "orange" if health_score < 4 else "green"
        st.markdown(f"""
            <div style="margin-top:10px;">
                <div style="display:flex; justify-content:space-between;">
                    <b>Sağlık Puanı: {health_score}/5</b>
                    <b style="color:{bar_color}">{health_text}</b>
                </div>
                <div style="width:100%; background:#eee; height:15px; border-radius:10px;">
                    <div style="width:{health_score*20}%; background:{bar_color}; height:100%; border-radius:10px;"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Detaylar
        with st.expander("Detaylı Karne"):
            st.write(f"• Kâr Marjı: %{info.get('profitMargins', 0)*100:.1f}")
            st.write(f"• Özsermaye Kârlılığı (ROE): %{info.get('returnOnEquity', 0)*100:.1f}")
            st.write(f"• Borç/Özsermaye: {info.get('debtToEquity', 'N/A')}")

    with col3:
        # Teknik Gösterge (İbre)
        st.markdown("### ⚡ Teknik Durum")
        # Basit teknik skor (RSI ve Ortalamalara göre)
        tech_score = 50
        if rsi < 30: tech_score += 20
        elif rsi > 70: tech_score -= 20
        if current_price > df_tech['SMA_200'].iloc[-1]: tech_score += 20
        else: tech_score -= 20
        
        fig_gauge = create_gauge_chart(tech_score, "Alım/Satım Gücü")
        st.plotly_chart(fig_gauge, use_container_width=True)
        if tech_score > 60: st.success("**GÜÇLÜ AL** Sinyali")
        elif tech_score < 40: st.error("**GÜÇLÜ SAT** Sinyali")
        else: st.warning("**NÖTR**")

    st.markdown("---")

    # 2. GRAFİK VE AI
    st.subheader("📈 Fiyat Grafiği")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df_tech.index, open=df_tech['Open'], high=df_tech['High'], low=df_tech['Low'], close=df_tech['Close'], name='Fiyat'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['SMA_50'], name='SMA 50', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['SMA_200'], name='SMA 200', line=dict(color='blue')), row=1, col=1)
    
    # Fair Value Çizgisi (Varsa)
    if fair_value:
        fig.add_hline(y=fair_value, line_dash="dash", line_color="purple", annotation_text="Adil Değer", row=1, col=1)

    fig.add_trace(go.Bar(x=df_tech.index, y=df_tech['Volume'], name='Hacim'), row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

def show_main_interface():
    with st.sidebar:
        st.title("Ayarlar")
        if st.button("Çıkış"):
            st.session_state['basladi'] = False
            st.rerun()

    st.title("📊 Piyasa Analiz Terminali")
    c1, c2 = st.columns([3, 1])
    with c1:
        search = st.text_input("Hisse Kodu", value="THYAO")
    with c2:
        st.write("")
        st.write("")
        btn = st.button("🔍 ANALİZ ET", use_container_width=True)

    if btn:
        hisseler = [s.strip().upper() + ".IS" if not s.strip().endswith(".IS") else s.strip().upper() for s in search.split(",")]
        with st.spinner("AI ve Finans Motorları Çalışıyor..."):
            analyze_stocks(hisseler)

def main():
    if not st.session_state['basladi']:
        show_landing_page()
    else:
        show_main_interface()

if __name__ == "__main__":
    main()
