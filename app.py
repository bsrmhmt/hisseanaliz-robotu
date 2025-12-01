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
st.set_page_config(page_title="AI Finans V11", layout="wide", initial_sidebar_state="collapsed")

# --- Session State (Sayfa Yönlendirme Sistemi) ---
if 'page' not in st.session_state: st.session_state['page'] = 'landing'
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
            df = ticker.history(period="2y")
            info = ticker.info
            if len(df) < 50: return None
            return {'data': df, 'info': info}
        except: return None

# --- YENİ: RİSK ANALİZ MOTORU ---
class RiskEngine:
    def calculate_risk_metrics(self, df):
        # Günlük Getiriler
        df['Returns'] = df['Close'].pct_change()
        
        # 1. Volatilite (Yıllıklandırılmış Standart Sapma)
        volatility = df['Returns'].std() * np.sqrt(252) * 100
        
        # 2. VaR (Value at Risk - %95 Güvenle)
        # "En kötü günde paranızın % kaçını kaybedersiniz?"
        var_95 = np.percentile(df['Returns'].dropna(), 5) * 100
        
        # 3. Max Drawdown (Zirveden Maksimum Düşüş)
        # "Tepeden alan biri en fazla ne kadar zarar gördü?"
        cumulative_returns = (1 + df['Returns']).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns / peak) - 1
        max_drawdown = drawdown.min() * 100
        
        return {
            'volatility': volatility,
            'var_95': var_95,
            'max_drawdown': max_drawdown,
            'drawdown_series': drawdown
        }

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
        except: return 0, 0, {}

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
            analysis_text += f"👋 **Durum:** Trend {trend}. "
            if rsi < 30: analysis_text += "Fiyat ucuzladı."
            elif rsi > 70: analysis_text += "Fiyat şişti."
            recs.append("Acele etme.")
        elif level == "Profesyonel":
            analysis_text += f"📈 **Teknik:** Fiyat > SMA200 ({trend}). RSI:{rsi:.1f}."
            recs.append("ATR bazlı stop takibi.")
        else:
            analysis_text += "📊 **Analiz:** Göstergeler inceleniyor..."
            
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
# 2. SAYFA TASARIMLARI (FRONTEND)
# ==========================================

def show_landing_page():
    # --- PREMIUM CSS ---
    st.markdown("""
        <style>
        .stApp {background-color: #ffffff;}
        
        .main-title {
            font-size: 5rem;
            font-weight: 800;
            background: linear-gradient(120deg, #4facfe 0%, #00f2fe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 10px;
            letter-spacing: -2px;
        }
        .sub-title {
            font-size: 1.4rem;
            color: #555;
            text-align: center;
            font-weight: 300;
            margin-bottom: 60px;
        }
        
        /* Kartlar */
        .feature-card {
            background: #fff;
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.05);
            text-align: center;
            border: 1px solid #f0f0f0;
            height: 280px;
            transition: all 0.3s ease;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .feature-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 20px 50px rgba(0,0,0,0.1);
            border-color: #4facfe;
        }
        .icon { font-size: 3rem; margin-bottom: 10px; }
        .card-h { font-weight: 700; font-size: 1.2rem; color: #333; margin-bottom: 10px; }
        .card-p { font-size: 0.9rem; color: #666; margin-bottom: 15px; }
        
        /* Butonlar */
        .stButton button {
            border-radius: 50px;
            font-weight: 600;
            border: none;
            transition: all 0.3s;
        }
        
        /* Büyük Başlat Butonu */
        .big-btn button {
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 20px 50px;
            font-size: 1.3rem !important;
            box-shadow: 0 10px 30px rgba(79, 172, 254, 0.4);
            width: 100%;
        }
        .big-btn button:hover {
            transform: scale(1.05);
            box-shadow: 0 15px 40px rgba(79, 172, 254, 0.6);
        }
        
        /* Risk Butonu */
        .risk-btn button {
            background-color: transparent;
            color: #4facfe;
            border: 2px solid #4facfe;
        }
        .risk-btn button:hover {
            background-color: #4facfe;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">AI Finans Premium</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Yapay Zeka Destekli Finansal Mühendislik</div>', unsafe_allow_html=True)

    # --- KARTLAR ---
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="feature-card">
            <div><div class="icon">🧠</div><div class="card-h">Yapay Zeka</div><div class="card-p">Fiyat hareketlerini tahmin eden akıllı motor.</div></div>
        </div>
        """, unsafe_allow_html=True)
        # Buton hilesi: Kartın altına buton koyuyoruz ama görsel bütünlük sağlıyoruz
        st.write("") 

    with c2:
        st.markdown("""
        <div class="feature-card">
            <div><div class="icon">🛡️</div><div class="card-h">Risk Merkezi</div><div class="card-p">VaR, Drawdown ve Volatilite analizleri.</div></div>
        </div>
        """, unsafe_allow_html=True)
        # RİSK BUTONU
        st.markdown('<div class="risk-btn">', unsafe_allow_html=True)
        if st.button("🛡️ Risk Analizine Git", key="go_risk", use_container_width=True):
            navigate_to('risk')
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="feature-card">
            <div><div class="icon">💬</div><div class="card-h">Asistan</div><div class="card-p">Kişiselleştirilmiş yatırım yorumları.</div></div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")
    st.write("")

    # --- ORTALANMIŞ DEV BAŞLAT BUTONU ---
    # Ortalamak için kolon hilesi: [1, 2, 1] yerine [3, 2, 3] daha dar ve ortalı yapar
    col_l, col_m, col_r = st.columns([3, 2, 3])
    with col_m:
        st.markdown('<div class="big-btn">', unsafe_allow_html=True)
        if st.button("🚀 TERMİNALİ BAŞLAT", key="go_main"):
            navigate_to('main')
        st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------
# SAYFA: RİSK ANALİZ MERKEZİ (YENİ)
# ----------------------------------------
def show_risk_interface():
    st.markdown("""<style>.stApp{background-color:#f8f9fa;}</style>""", unsafe_allow_html=True)
    
    with st.sidebar:
        st.title("🛡️ Risk Ayarları")
        if st.button("🏠 Ana Ekrana Dön"): navigate_to('landing')
    
    st.title("🛡️ Risk Analiz Merkezi")
    st.markdown("Profesyonel portföy yöneticilerinin kullandığı risk metrikleri.")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        sembol = st.text_input("Risk Analizi İçin Hisse Kodu:", "THYAO")
    with col2:
        st.write("")
        st.write("")
        hesapla = st.button("Hesapla", use_container_width=True)
    
    if hesapla:
        with st.spinner("Risk motoru çalışıyor..."):
            fetcher = AdvancedDataFetcher()
            data = fetcher.get_stock_data(sembol + ".IS")
            
            if data:
                df = data['data']
                engine = RiskEngine()
                metrics = engine.calculate_risk_metrics(df)
                
                # Risk Kartları
                c1, c2, c3 = st.columns(3)
                c1.metric("Volatilite (Risk)", f"%{metrics['volatility']:.2f}", "Yıllık Dalgalanma")
                c2.metric("VaR (%95)", f"%{metrics['var_95']:.2f}", "En Kötü Gün Kaybı")
                c3.metric("Max Drawdown", f"%{metrics['max_drawdown']:.2f}", "Tepeden Maks. Düşüş")
                
                # Grafik: Drawdown
                st.subheader("📉 Drawdown (Su Altında Kalma) Grafiği")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=metrics['drawdown_series'], fill='tozeroy', line=dict(color='red'), name='Drawdown'))
                fig.update_layout(title="Zirveden Düşüş Oranları", yaxis_tickformat='%')
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk Metre (Gauge)
                st.subheader("⏱️ Risk Metre")
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = abs(metrics['max_drawdown']),
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Risk Seviyesi (Max DD)"},
                    gauge = {
                        'axis': {'range': [None, 50]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 10], 'color': "lightgreen"},
                            {'range': [10, 20], 'color': "yellow"},
                            {'range': [20, 50], 'color': "red"}],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': abs(metrics['max_drawdown'])}
                    }
                ))
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                st.info("ℹ️ **VaR (Value at Risk):** %95 güvenle, bir günde kaybedebileceğiniz maksimum tutarı ifade eder.\n\nℹ️ **Max Drawdown:** Hissenin tarihi zirvesinden en dibe ne kadar düştüğünü gösterir. Kriz anındaki dayanıklılığı ölçer.")

            else:
                st.error("Veri bulunamadı.")

# ----------------------------------------
# SAYFA: ANA TERMİNAL (ESKİ SİSTEM)
# ----------------------------------------
def show_main_interface():
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Kontrol")
        ai_seviye = st.selectbox("Analiz Dili", ["Acemi", "Orta Düzey", "Profesyonel"], index=1)
        tahmin_periyodu = st.slider("Vade (Gün)", 1, 30, 5)
        st.divider()
        if st.button("🏠 Ana Ekrana Dön"): navigate_to('landing')

    st.title("📊 Piyasa Analiz Terminali")
    
    # ... Eski analiz kodları aynen burada ...
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

# --- ANALİZ FONKSİYONU (Eski Koddan) ---
def analyze_stocks(hisseler, ai_seviye, tahmin_periyodu):
    sembol = hisseler[0]
    fetcher = AdvancedDataFetcher()
    data = fetcher.get_stock_data(sembol)
    if not data:
        st.error(f"⚠️ {sembol} için veri çekilemedi.")
        return
    
    df_raw = data['data']
    ta_engine = AdvancedTechnicalAnalysis()
    df = ta_engine.calculate_all_indicators(df_raw)
    predictor = AdvancedStockPredictor()
    prediction, confidence, imp = predictor.predict_with_confidence(df, horizon=tahmin_periyodu)
    assistant = AdvancedAIAssistant()
    res = assistant.generate_analysis({'data': df}, {}, ai_seviye)

    current_price = df['Close'].iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fiyat", f"{current_price:.2f} TL", f"%{((current_price/df['Close'].iloc[-2])-1)*100:.2f}")
    c2.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
    c3.metric("AI Hedef", f"{prediction:.2f} TL")
    c4.metric("Güven", f"±{confidence:.2f}")

    st.subheader("📈 Profesyonel Grafik")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Fiyat'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name='SMA 200', line=dict(color='blue', width=2)), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Hacim', marker_color='lightblue'), row=2, col=1)
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns([2, 1])
    with c1: st.info(f"**Yorum:** {res['analysis']}")
    with c2: 
        st.write("Risk Analizi")
        st.markdown(DashboardComponents.create_progress_bar("Risk Skoru", res['metrics']['risk_score'], 10, "red"), unsafe_allow_html=True)

# --- ANA UYGULAMA YÖNETİCİSİ ---
def main():
    if st.session_state['page'] == 'landing':
        show_landing_page()
    elif st.session_state['page'] == 'main':
        show_main_interface()
    elif st.session_state['page'] == 'risk':
        show_risk_interface()

if __name__ == "__main__":
    main()
