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
st.set_page_config(page_title="AI Finans V10.2", layout="wide", initial_sidebar_state="collapsed")

# --- Session State ---
if 'basladi' not in st.session_state: st.session_state['basladi'] = False
if 'favoriler' not in st.session_state: st.session_state['favoriler'] = []
if 'analiz_gecmisi' not in st.session_state: st.session_state['analiz_gecmisi'] = []

# ==========================================
# 1. MOTOR BLOĞU (BACKEND)
# ==========================================

class AdvancedDataFetcher:
    def get_stock_data(self, sembol):
        try:
            ticker = yf.Ticker(sembol)
            df = ticker.history(period="2y")
            info = ticker.info
            if len(df) < 50: return None
            return {'data': df, 'info': info}
        except: return None

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
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['SMA_20'] + (df['Std']*2)
        df['BB_Lower'] = df['SMA_20'] - (df['Std']*2)
        df['Target'] = df['Close'].shift(-5)
        df.dropna(inplace=True)
        return df

class AdvancedStockPredictor:
    def predict_with_confidence(self, df, horizon=5):
        try:
            features = ['RSI', 'SMA_50', 'SMA_200', 'BB_Upper', 'BB_Lower', 'Volume']
            X = df[features]
            y = df['Target']
            X_train = X[:-horizon]
            y_train = y[:-horizon]
            X_today = X.tail(1)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            prediction = model.predict(X_today)[0]
            preds = model.predict(X_train[-50:])
            mae = mean_absolute_error(y_train[-50:], preds)
            importances = dict(zip(features, model.feature_importances_))
            return prediction, mae, importances
        except:
            return 0, 0, {}

class AdvancedAIAssistant:
    def generate_analysis(self, data_packet, preds, level):
        df = data_packet['data']
        rsi = df['RSI'].iloc[-1]
        price = df['Close'].iloc[-1]
        sma200 = df['SMA_200'].iloc[-1]
        analysis_text = ""
        recs = []
        metrics = {'risk_score': 5, 'trend': {'strength_score': 50}, 'rsi': rsi}
        trend = "YUKARI" if price > sma200 else "AŞAĞI"
        metrics['trend']['strength_score'] = 80 if trend == "YUKARI" else 30
        risk = 5
        if rsi > 70 or rsi < 30: risk += 2
        if trend == "AŞAĞI": risk += 2
        metrics['risk_score'] = min(10, risk)

        if level == "Acemi":
            analysis_text += f"👋 **Genel Durum:** Trend {trend} yönünde. "
            if rsi < 30: analysis_text += "Fiyatlar ucuzladı, fırsat olabilir."
            elif rsi > 70: analysis_text += "Fiyatlar şişti, dikkat et."
            recs.append("Acele etme, bekle.")
        elif level == "Profesyonel":
            analysis_text += f"📈 **Teknik:** Fiyat > SMA200 ({trend}). RSI:{rsi:.1f}."
            recs.append("ATR bazlı stop-loss kullan.")
        else:
            analysis_text += "📊 **Analiz:** Göstergeler incelendiğinde..."
            
        return {'analysis': analysis_text, 'recommendations': recs, 'metrics': metrics}

class DashboardComponents:
    @staticmethod
    def create_progress_bar(label, value, max_val, color):
        percentage = (value / max_val) * 100
        return f"""<div style="margin-bottom:10px;"><div style="display:flex;justify-content:space-between;"><span>{label}</span><span>{value}/{max_val}</span></div><div style="width:100%;background:#e0e0e0;border-radius:5px;"><div style="width:{percentage}%;height:10px;background:{color};border-radius:5px;"></div></div></div>"""

class StateManager:
    def add_to_history(self, sembol, data):
        if 'analiz_gecmisi' not in st.session_state: st.session_state['analiz_gecmisi'] = []
        st.session_state['analiz_gecmisi'].insert(0, {'sembol': sembol, **data})

state = StateManager()

# ==========================================
# 2. ARAYÜZ (FRONTEND)
# ==========================================

def show_landing_page():
    # --- CSS TASARIMI (MODERN GÖRÜNÜM) ---
    st.markdown("""
        <style>
        .stApp {background-color: #ffffff; color: #333;}
        
        /* Başlıklar */
        .main-title {
            font-size: 3.5rem;
            font-weight: 800;
            background: linear-gradient(120deg, #2980b9, #8e44ad);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 10px;
        }
        .sub-title {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            margin-bottom: 50px;
        }
        
        /* Kart Tasarımı */
        .feature-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            text-align: center;
            border: 1px solid #eee;
            height: 200px;
            transition: transform 0.3s;
        }
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }
        .card-icon { font-size: 2.5rem; margin-bottom: 15px; }
        .card-title { font-weight: bold; font-size: 1.1rem; margin-bottom: 10px; color: #333; }
        .card-desc { font-size: 0.9rem; color: #666; }
        
        /* Buton */
        .stButton button {
            width: 100%;
            border-radius: 50px;
            height: 50px;
            font-weight: bold;
            background: linear-gradient(90deg, #2980b9, #8e44ad);
            color: white;
            border: none;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- HTML İÇERİK (Parça Parça - Hata Yapmaz) ---
    
    st.markdown('<div class="main-title">AI Finans Premium</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Yapay Zeka Destekli Yeni Nesil Borsa Asistanı</div>', unsafe_allow_html=True)

    # Grid Yapısı (Streamlit Kolonları ile - Daha Güvenli)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="card-icon">🧠</div>
            <div class="card-title">Yapay Zeka</div>
            <div class="card-desc">Random Forest algoritması ile fiyat hareketlerini ve trendleri analiz eder.</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="card-icon">🛡️</div>
            <div class="card-title">Risk Analizi</div>
            <div class="card-desc">Otomatik stop-loss seviyeleri ve güven aralığı hesaplamaları.</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="card-icon">💬</div>
            <div class="card-title">Akıllı Asistan</div>
            <div class="card-desc">Sadece veri değil, seviyenize uygun sözlü yorumlar yapar.</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")
    st.write("")

    # Başlat Butonu
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button("🚀 TERMİNALİ BAŞLAT"):
            st.session_state['basladi'] = True
            st.rerun()

    st.markdown("<div style='text-align:center; color:#ccc; margin-top:50px;'>v10.2 Final Edition</div>", unsafe_allow_html=True)

def analyze_stocks(hisseler, ai_seviye, tahmin_periyodu):
    sembol = hisseler[0]
    
    fetcher = AdvancedDataFetcher()
    data = fetcher.get_stock_data(sembol)
    
    if not data:
        st.error(f"⚠️ {sembol} için veri çekilemedi.")
        return
    
    # HATA DÜZELTMESİ YAPILDI
    df_raw = data['data']
    ta_engine = AdvancedTechnicalAnalysis()
    df = ta_engine.calculate_all_indicators(df_raw) 
    
    predictor = AdvancedStockPredictor()
    prediction, confidence, imp = predictor.predict_with_confidence(df, horizon=tahmin_periyodu)
    
    assistant = AdvancedAIAssistant()
    res = assistant.generate_analysis({'data': df}, {}, ai_seviye)

    # --- DASHBOARD ---
    current_price = df['Close'].iloc[-1]
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fiyat", f"{current_price:.2f} TL", f"%{((current_price/df['Close'].iloc[-2])-1)*100:.2f}")
    c2.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
    c3.metric("AI Hedef", f"{prediction:.2f} TL")
    c4.metric("Güven Aralığı", f"±{confidence:.2f}")

    st.subheader("📈 Profesyonel Grafik")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Fiyat'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name='SMA 200', line=dict(color='blue', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='gray', width=1, dash='dot'), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='gray', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)', showlegend=False), row=1, col=1)
    
    colors = ['green' if df['Close'].iloc[i] > df['Open'].iloc[i] else 'red' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Hacim'), row=2, col=1)
    
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("🤖 AI Raporu")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info(f"**Yorum:** {res['analysis']}")
        if res['recommendations']:
            st.success(f"💡 **Öneri:** {res['recommendations'][0]}")
            
    with col2:
        st.write("Risk Analizi")
        st.markdown(DashboardComponents.create_progress_bar("Risk Skoru", res['metrics']['risk_score'], 10, "red"), unsafe_allow_html=True)
        st.markdown(DashboardComponents.create_progress_bar("Trend Gücü", res['metrics']['trend']['strength_score'], 100, "blue"), unsafe_allow_html=True)
    
    state.add_to_history(sembol, {'price': current_price, 'timestamp': datetime.now().strftime("%H:%M")})

def show_main_interface():
    with st.sidebar:
        st.title("⚙️ Kontrol")
        ai_seviye = st.selectbox("Analiz Dili", ["Acemi", "Orta Düzey", "Profesyonel"], index=1)
        tahmin_periyodu = st.slider("Vade (Gün)", 1, 30, 5)
        st.divider()
        if st.button("🔙 Çıkış"):
            st.session_state['basladi'] = False
            st.rerun()

    st.title("📊 Piyasa Analiz")
    c1, c2 = st.columns([3, 1])
    with c1:
        search = st.text_input("Hisse Kodu", value="THYAO")
    with c2:
        st.write("")
        st.write("")
        btn = st.button("🔍 Analiz", use_container_width=True)

    if btn:
        hisseler = [s.strip().upper() + ".IS" if not s.strip().endswith(".IS") else s.strip().upper() for s in search.split(",")]
        with st.spinner("Yapay Zeka Çalışıyor..."):
            analyze_stocks(hisseler, ai_seviye, tahmin_periyodu)

# --- ANA UYGULAMA ---
def main():
    if not st.session_state['basladi']:
        show_landing_page()
    else:
        show_main_interface()

if __name__ == "__main__":
    main()
