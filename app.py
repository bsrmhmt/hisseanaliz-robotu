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
st.set_page_config(page_title="AI Finans Ultimate", layout="wide", initial_sidebar_state="collapsed")

# --- Session State ---
if 'page' not in st.session_state: st.session_state['page'] = 'landing'
if 'favoriler' not in st.session_state: st.session_state['favoriler'] = []
if 'analiz_gecmisi' not in st.session_state: st.session_state['analiz_gecmisi'] = []

def navigate_to(page):
    st.session_state['page'] = page
    st.rerun()

# ==========================================
# 1. MOTOR BLOĞU (TÜM MOTORLAR BİR ARADA)
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

class FundamentalEngine: # InvestingPro Tarzı Veri Motoru
    def calculate_fair_value(self, info):
        try:
            eps = info.get('trailingEps', 0)
            book_value = info.get('bookValue', 0)
            current_price = info.get('currentPrice', 0)
            if eps is None or book_value is None or eps <= 0 or book_value <= 0: return None, 0
            fair_value = np.sqrt(22.5 * eps * book_value)
            upside = ((fair_value - current_price) / current_price) * 100
            return fair_value, upside
        except: return None, 0

    def calculate_health_score(self, info):
        score = 0
        try:
            if info.get('profitMargins', 0) > 0.10: score += 1
            if info.get('returnOnEquity', 0) > 0.15: score += 1
            if info.get('revenueGrowth', 0) > 0.10: score += 1
            if info.get('debtToEquity', 100) < 100: score += 1
            if info.get('quickRatio', 0) > 1: score += 1
            return score
        except: return 2

class RiskEngine:
    def calculate_risk_metrics(self, df):
        df = df.copy()
        df['Returns'] = df['Close'].pct_change()
        volatility = df['Returns'].std() * np.sqrt(252) * 100
        var_95 = np.percentile(df['Returns'].dropna(), 5) * 100
        cumulative = (1 + df['Returns']).cumprod()
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative / peak) - 1
        max_dd = drawdown.min() * 100
        return {'volatility': volatility, 'var_95': var_95, 'max_drawdown': max_dd, 'drawdown_series': drawdown}

class AdvancedTechnicalAnalysis:
    def calculate_all_indicators(self, df):
        df = df.copy()
        df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().where(lambda x: x>0,0).rolling(14).mean() / df['Close'].diff().where(lambda x: x<0,0).abs().rolling(14).mean())))
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['SMA_200'] = df['Close'].rolling(200).mean()
        df['BB_Upper'] = df['Close'].rolling(20).mean() + (df['Close'].rolling(20).std()*2)
        df['BB_Lower'] = df['Close'].rolling(20).mean() - (df['Close'].rolling(20).std()*2)
        df['Target'] = df['Close'].shift(-5)
        df.dropna(inplace=True)
        return df

class AdvancedStockPredictor:
    def predict_with_confidence(self, df, horizon=5):
        try:
            features = ['RSI', 'SMA_50', 'SMA_200', 'BB_Upper', 'BB_Lower', 'Volume']
            X = df[features]; y = df['Target']
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
        text = f"Trend {trend}. "
        if rsi < 30: text += "Fiyat ucuz."
        elif rsi > 70: text += "Fiyat pahalı."
        return {'analysis': text}

class DashboardComponents:
    @staticmethod
    def create_progress_bar(label, value, max_val, color):
        pct = (value / max_val) * 100
        return f"""<div style="margin-bottom:5px;"><div style="display:flex;justify-content:space-between;"><span>{label}</span><span>{value}/{max_val}</span></div><div style="width:100%;background:#eee;height:8px;border-radius:5px;"><div style="width:{pct}%;height:100%;background:{color};border-radius:5px;"></div></div></div>"""

# ==========================================
# 2. SAYFA TASARIMLARI
# ==========================================

def show_landing_page():
    # --- PREMIUM CSS (SENİN SEVDİĞİN TASARIM) ---
    st.markdown("""
        <style>
        .stApp {background-color: #ffffff;}
        .main-title {
            font-size: 5rem; font-weight: 800;
            background: linear-gradient(120deg, #4facfe 0%, #00f2fe 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            text-align: center; margin-bottom: 10px; letter-spacing: -2px;
        }
        .sub-title { font-size: 1.4rem; color: #555; text-align: center; font-weight: 300; margin-bottom: 60px; }
        .feature-card {
            background: #fff; padding: 30px; border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.05); text-align: center; border: 1px solid #f0f0f0;
            height: 280px; transition: all 0.3s ease; display: flex; flex-direction: column; justify-content: space-between;
        }
        .feature-card:hover { transform: translateY(-10px); box-shadow: 0 20px 50px rgba(0,0,0,0.1); border-color: #4facfe; }
        .icon { font-size: 3rem; margin-bottom: 10px; }
        .card-h { font-weight: 700; font-size: 1.2rem; color: #333; margin-bottom: 10px; }
        .card-p { font-size: 0.9rem; color: #666; margin-bottom: 15px; }
        .big-btn button {
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%); color: white;
            padding: 20px 50px; font-size: 1.3rem !important; box-shadow: 0 10px 30px rgba(79, 172, 254, 0.4);
            width: 100%; border-radius: 50px; border: none; font-weight: 600;
        }
        .big-btn button:hover { transform: scale(1.05); box-shadow: 0 15px 40px rgba(79, 172, 254, 0.6); }
        .risk-btn button { background-color: transparent; color: #4facfe; border: 2px solid #4facfe; border-radius: 50px; }
        .risk-btn button:hover { background-color: #4facfe; color: white; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">AI Finans Ultimate</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Yapay Zeka & Temel Analiz Bir Arada</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="feature-card"><div><div class="icon">🧠</div><div class="card-h">Yapay Zeka</div><div class="card-p">Fiyat hareketlerini tahmin eden akıllı motor.</div></div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="feature-card"><div><div class="icon">🛡️</div><div class="card-h">Risk Merkezi</div><div class="card-p">VaR, Drawdown ve Volatilite analizleri.</div></div></div>', unsafe_allow_html=True)
        st.markdown('<div class="risk-btn">', unsafe_allow_html=True)
        if st.button("🛡️ Risk Analizine Git", key="go_risk", use_container_width=True): navigate_to('risk')
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="feature-card"><div><div class="icon">📊</div><div class="card-h">Adil Değer</div><div class="card-p">InvestingPro tarzı temel analiz verileri.</div></div></div>', unsafe_allow_html=True)

    st.write(""); st.write(""); st.write("")
    
    col_l, col_m, col_r = st.columns([3, 2, 3])
    with col_m:
        st.markdown('<div class="big-btn">', unsafe_allow_html=True)
        if st.button("🚀 TERMİNALİ BAŞLAT", key="go_main"): navigate_to('main')
        st.markdown('</div>', unsafe_allow_html=True)

def show_risk_interface():
    st.markdown("""<style>.stApp{background-color:#f8f9fa;}</style>""", unsafe_allow_html=True)
    with st.sidebar:
        st.title("🛡️ Risk Ayarları"); 
        if st.button("🏠 Ana Ekrana Dön"): navigate_to('landing')
    
    st.title("🛡️ Risk Analiz Merkezi"); c1, c2 = st.columns([3, 1])
    with c1: sembol = st.text_input("Hisse Kodu:", "THYAO")
    with c2: 
        st.write(""); st.write("")
        if st.button("Hesapla", use_container_width=True):
            fetcher = AdvancedDataFetcher()
            data = fetcher.get_stock_data(sembol + ".IS")
            if data:
                metrics = RiskEngine().calculate_risk_metrics(data['data'])
                k1, k2, k3 = st.columns(3)
                k1.metric("Volatilite", f"%{metrics['volatility']:.2f}")
                k2.metric("VaR (%95)", f"%{metrics['var_95']:.2f}")
                k3.metric("Max Drawdown", f"%{metrics['max_drawdown']:.2f}")
                st.line_chart(metrics['drawdown_series'])
            else: st.error("Veri yok.")

def show_main_interface():
    with st.sidebar:
        st.title("⚙️ Kontrol"); 
        if st.button("🏠 Ana Ekrana Dön"): navigate_to('landing')

    st.title("📊 Piyasa Analiz Terminali")
    c1, c2 = st.columns([3, 1])
    with c1: search = st.text_input("Hisse Kodu", value="THYAO")
    with c2: 
        st.write(""); st.write("")
        btn = st.button("🔍 Analiz", use_container_width=True)

    if btn:
        hisseler = [s.strip().upper() + ".IS" if not s.strip().endswith(".IS") else s.strip().upper() for s in search.split(",")]
        sembol = hisseler[0]
        
        with st.spinner("AI ve Finans Motorları Çalışıyor..."):
            fetcher = AdvancedDataFetcher()
            data = fetcher.get_stock_data(sembol)
            
            if not data:
                st.error("Veri yok.")
                return
            
            # Tüm Motorları Çalıştır
            df = data['data']; info = data['info']
            fund_eng = FundamentalEngine()
            fair_val, upside = fund_eng.calculate_fair_value(info)
            health = fund_eng.calculate_health_score(info)
            
            tech_eng = AdvancedTechnicalAnalysis()
            df = tech_eng.calculate_all_indicators(df)
            
            pred_eng = AdvancedStockPredictor()
            pred, conf, _ = pred_eng.predict_with_confidence(df)
            
            current = df['Close'].iloc[-1]
            
            # --- 1. ADİL DEĞER VE SAĞLIK (INVESTING PRO TARZI) ---
            st.markdown("### 🏆 Temel Analiz Karnesi")
            col1, col2, col3 = st.columns(3)
            
            with col1: # Adil Değer
                color = "normal" if upside > 0 else "inverse"
                st.metric("Adil Değer (Fair Value)", f"{fair_val:.2f} TL" if fair_val else "N/A", f"{upside:+.2f}% Potansiyel", delta_color=color)
                if upside > 20: st.success("HİSSE ÇOK UCUZ")
                elif upside < -20: st.error("HİSSE PAHALI")
                else: st.warning("EDERİNDE")
                
            with col2: # Sağlık Skoru
                st.metric("Şirket Sağlığı", f"{health}/5 Puan")
                bar_color = "red" if health < 3 else "green"
                st.markdown(DashboardComponents.create_progress_bar("Finansal Güç", health, 5, bar_color), unsafe_allow_html=True)
                
            with col3: # AI Tahmin
                st.metric("AI Tahmini (T+5)", f"{pred:.2f} TL", f"Güven: ±{conf:.2f}")

            st.divider()

            # --- 2. GRAFİK ---
            st.subheader("📈 Teknik Grafik")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Fiyat'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='orange')), row=1, col=1)
            
            # Fair Value Çizgisi (Varsa)
            if fair_val:
                fig.add_hline(y=fair_val, line_dash="dash", line_color="purple", annotation_text="Adil Değer", row=1, col=1)

            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Hacim'), row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)

def main():
    if st.session_state['page'] == 'landing': show_landing_page()
    elif st.session_state['page'] == 'main': show_main_interface()
    elif st.session_state['page'] == 'risk': show_risk_interface()

if __name__ == "__main__":
    main()
