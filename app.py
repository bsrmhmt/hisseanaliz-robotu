import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from datetime import datetime
import time

# --- Sayfa Ayarları ---
st.set_page_config(page_title="AI Finans V10", layout="wide", initial_sidebar_state="collapsed")

# --- Session State ---
if 'basladi' not in st.session_state: st.session_state['basladi'] = False
if 'favoriler' not in st.session_state: st.session_state['favoriler'] = []
if 'analiz_gecmisi' not in st.session_state: st.session_state['analiz_gecmisi'] = []

# ==========================================
# 1. MOTOR BLOĞU (BACKEND SINIFLARI)
# ==========================================

class AdvancedDataFetcher:
    def get_stock_data(self, sembol):
        try:
            # ML için uzun veri çekiyoruz
            ticker = yf.Ticker(sembol)
            df = ticker.history(period="2y")
            info = ticker.info
            if len(df) < 50: return None
            return {'data': df, 'info': info}
        except:
            return None

class AdvancedTechnicalAnalysis:
    def calculate_all_indicators(self, df):
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
        df['BB_Upper'] = df['SMA_20'] + (df['Std']*2)
        df['BB_Lower'] = df['SMA_20'] - (df['Std']*2)
        
        # Hacim Ortalaması
        df['Vol_SMA'] = df['Volume'].rolling(window=20).mean()
        
        # Features for ML
        df['Target'] = df['Close'].shift(-tahmin_periyodu_global) # Global değişkeni kullan
        
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
            
            # Model Eğitimi
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            prediction = model.predict(X_today)[0]
            
            # Güven Aralığı (Basit Standart Sapma Bazlı)
            preds = model.predict(X_train[-50:])
            mae = mean_absolute_error(y_train[-50:], preds)
            
            # Feature Importance
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
        metrics = {
            'risk_score': 5, # Varsayılan
            'trend': {'strength_score': 50},
            'rsi': rsi
        }
        
        # Trend Analizi
        trend = "YUKARI" if price > sma200 else "AŞAĞI"
        metrics['trend']['strength_score'] = 80 if trend == "YUKARI" else 30
        
        # Risk Skoru Hesapla
        risk = 5
        if rsi > 70 or rsi < 30: risk += 2
        if trend == "AŞAĞI": risk += 2
        metrics['risk_score'] = min(10, risk)

        # Seviyeye Göre Yorum
        if level == "Acemi":
            analysis_text += f"👋 **Genel Durum:** Hisse şu an {trend} trendinde. "
            if rsi < 30: analysis_text += "Fiyatlar çok düştü, indirim fırsatı olabilir."
            elif rsi > 70: analysis_text += "Fiyatlar çok yükseldi, dikkatli ol."
            else: analysis_text += "Piyasa normal seyrinde."
            recs.append("Uzun vadeli düşünüyorsan acele etme.")
            
        elif level == "Profesyonel":
            analysis_text += f"📈 **Teknik Görünüm:** Fiyat SMA200 üzerinde, Bullish yapı korunuyor." if trend=="YUKARI" else "📉 Fiyat SMA200 altında, Bearish baskı var."
            analysis_text += f" RSI({rsi:.1f}) seviyesinde momentum takibi yapılmalı."
            recs.append("Stop-loss seviyelerini ATR bazlı takip et.")
        
        else: # Orta Düzey
            analysis_text += "📊 **Analiz:** Trend yönü ve göstergeler incelendiğinde..."
            if trend=="YUKARI": analysis_text += " Yükseliş trendi destekleniyor."
            
        return {'analysis': analysis_text, 'recommendations': recs, 'metrics': metrics}

class DashboardComponents:
    @staticmethod
    def create_progress_bar(label, value, max_val, color):
        percentage = (value / max_val) * 100
        return f"""
        <div style="margin-bottom: 10px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                <span>{label}</span>
                <span>{value}/{max_val}</span>
            </div>
            <div style="width:100%; background-color: #e0e0e0; border-radius: 5px;">
                <div style="width:{percentage}%; height: 10px; background-color: {color}; border-radius: 5px;"></div>
            </div>
        </div>
        """

class StateManager:
    def add_to_history(self, sembol, data):
        if 'analiz_gecmisi' not in st.session_state:
            st.session_state['analiz_gecmisi'] = []
        # En başa ekle
        st.session_state['analiz_gecmisi'].insert(0, {'sembol': sembol, **data})

state = StateManager()
tahmin_periyodu_global = 5 # Varsayılan

# ==========================================
# 2. ARAYÜZ FONKSİYONLARI
# ==========================================

def show_landing_page():
    """Başlangıç ekranı"""
    st.markdown("""
    <style>
    .landing-container {
        padding: 50px;
        text-align: center;
        background: linear-gradient(180deg, rgba(255,255,255,0.1) 0%, rgba(0,0,0,0) 100%);
        border-radius: 20px;
    }
    .title { font-size: 3.5rem; font-weight: bold; margin-bottom: 10px; background: -webkit-linear-gradient(45deg, #FF4B2B, #FF416C); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
    .subtitle { font-size: 1.5rem; color: #888; margin-bottom: 40px; }
    .feature-card { background: #262730; padding: 20px; border-radius: 10px; margin: 10px; border: 1px solid #444; }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="landing-container"><div class="title">AI Finans V10</div><div class="subtitle">Mühendislik Harikası Borsa Asistanı</div></div>', unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        c1.markdown('<div class="feature-card">🤖<br><b>AI Analiz</b></div>', unsafe_allow_html=True)
        c2.markdown('<div class="feature-card">📊<br><b>Teknik Veri</b></div>', unsafe_allow_html=True)
        c3.markdown('<div class="feature-card">🎯<br><b>Hedef Fiyat</b></div>', unsafe_allow_html=True)
        
        st.write("")
        st.write("")
        if st.button("🚀 ASİSTANI BAŞLAT", use_container_width=True, type="primary"):
            st.session_state['basladi'] = True
            st.rerun()

def analyze_stocks(hisseler, ai_seviye, tahmin_periyodu):
    global tahmin_periyodu_global
    tahmin_periyodu_global = tahmin_periyodu
    
    sembol = hisseler[0]
    
    fetcher = AdvancedDataFetcher()
    data = fetcher.get_stock_data(sembol)
    
    if not data or data['data'].empty:
        st.error(f"{sembol} verisi çekilemedi.")
        return
    
    df = data['data']
    info = data['info']
    
    ta_engine = AdvancedTechnicalAnalysis()
    df_with_indicators = ta_engine.calculate_all_indicators(df)
    
    predictor = AdvancedStockPredictor()
    prediction, confidence, feature_importance = predictor.predict_with_confidence(df_with_indicators, horizon=tahmin_periyodu)
    
    assistant = AdvancedAIAssistant()
    predictions_data = {'prediction': prediction, 'confidence': confidence}
    analysis_result = assistant.generate_analysis({'data': df_with_indicators}, predictions_data, ai_seviye)
    
    # --- DASHBOARD ---
    current_price = df['Close'].iloc[-1]
    
    # 1. Metrikler
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fiyat", f"{current_price:.2f} TL", f"%{((current_price/df['Close'].iloc[-2])-1)*100:.2f}")
    col2.metric("RSI", f"{df_with_indicators['RSI'].iloc[-1]:.1f}")
    col3.metric("AI Hedef", f"{prediction:.2f} TL")
    col4.metric("Güven Aralığı (Hata Payı)", f"±{confidence:.2f} TL")
    
    # 2. Grafik
    st.subheader("📈 Fiyat ve AI Tahmini")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Fiyat'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name='SMA 200', line=dict(color='blue')), row=1, col=1)
    
    # Hacim
    colors = ['green' if df['Close'].iloc[i] > df['Open'].iloc[i] else 'red' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Hacim'), row=2, col=1)
    
    fig.update_layout(height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # 3. Analiz Raporu
    st.subheader("🤖 AI Raporu")
    c1, c2 = st.columns([3, 1])
    
    with c1:
        st.info(analysis_result['analysis'])
        if analysis_result['recommendations']:
            st.success(f"💡 Öneri: {analysis_result['recommendations'][0]}")
            
    with c2:
        metrics = analysis_result['metrics']
        st.markdown(DashboardComponents.create_progress_bar("Risk Skoru", metrics['risk_score'], 10, "red"), unsafe_allow_html=True)
        st.markdown(DashboardComponents.create_progress_bar("Trend Gücü", metrics['trend']['strength_score'], 100, "blue"), unsafe_allow_html=True)
    
    # Geçmişe Ekle
    state.add_to_history(sembol, {'price': current_price, 'prediction': prediction, 'timestamp': datetime.now().strftime("%H:%M")})


def show_main_interface():
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Kontrol Paneli")
        ai_seviye = st.selectbox("Analiz Dili:", ["Acemi", "Orta Düzey", "Profesyonel"], index=1)
        tahmin_periyodu = st.slider("Tahmin Vadesi (Gün)", 1, 30, 5)
        st.divider()
        st.subheader("📜 Son Analizler")
        for item in st.session_state['analiz_gecmisi'][:5]:
            st.caption(f"{item['timestamp']} - {item['sembol']}")
        
        st.divider()
        if st.button("🔙 Ana Ekrana Dön"):
            st.session_state['basladi'] = False
            st.rerun()

    # Ana Ekran
    st.title("📊 Akıllı Hisse Analizi")
    col1, col2 = st.columns([3, 1])
    with col1:
        search_input = st.text_input("Hisse Kodu (Örn: THYAO, ASELS)", value="THYAO")
    with col2:
        st.write("") 
        st.write("") 
        analyze_btn = st.button("🔍 Analiz Et", use_container_width=True)

    if analyze_btn:
        hisseler = [s.strip().upper() + ".IS" if not s.strip().endswith(".IS") else s.strip().upper() for s in search_input.split(",")]
        with st.spinner("AI Verileri İşliyor..."):
            analyze_stocks(hisseler, ai_seviye, tahmin_periyodu)

# --- ANA UYGULAMA ---
def main():
    if not st.session_state['basladi']:
        show_landing_page()
    else:
        show_main_interface()

if __name__ == "__main__":
    main()
