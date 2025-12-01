
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# --- 1. SINIFLAR VE MOTORLAR BURAYA ---
class AdvancedDataFetcher:
    # ... kodlar ...
    pass

class AdvancedTechnicalAnalysis:
    # ... kodlar ...
    pass

class AdvancedStockPredictor:
    # ... kodlar ...
    pass

class AdvancedAIAssistant:
    # ... kodlar ...
    pass

class DashboardComponents:
    # ... kodlar ...
    pass

# State yönetimi için basit bir sınıf veya sözlük yapısı
class StateManager:
    def add_to_history(self, sembol, data):
        if 'analiz_gecmisi' not in st.session_state:
            st.session_state['analiz_gecmisi'] = []
        st.session_state['analiz_gecmisi'].insert(0, {'sembol': sembol, **data})

state = StateManager()

# --- 2. AYARLAR VE SESSION STATE ---
st.set_page_config(page_title="AI Finans V10", layout="wide")

if 'basladi' not in st.session_state: st.session_state['basladi'] = False
if 'favoriler' not in st.session_state: st.session_state['favoriler'] = []
if 'analiz_gecmisi' not in st.session_state: st.session_state['analiz_gecmisi'] = []

# --- 3. SENİN YAZDIĞIN ARAYÜZ KODLARI BURAYA ---
def main():
    # ... senin kodların ...

# ... diğer fonksiyonların (show_landing_page vb.) ...

if __name__ == "__main__":
    main()




# --- Ana Uygulama ---
def main():
    # İlk başlangıç ekranı
    if not st.session_state['basladi']:
        show_landing_page()
    else:
        show_main_interface()

def show_landing_page():
    """Başlangıç ekranı"""
    st.markdown("""
    <style>
    .landing-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        padding: 40px;
        border-radius: 0;
    }
    .title {
        color: white;
        font-size: 4rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.5rem;
        text-align: center;
        margin-bottom: 40px;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
        margin: 40px 0;
        max-width: 900px;
    }
    .feature-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 10px;
    }
    .feature-title {
        color: white;
        font-size: 1.2rem;
        margin-bottom: 10px;
    }
    .feature-desc {
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="landing-container">
            <div class="title">AI Finans V10</div>
            <div class="subtitle">Yeni Nesil Akıllı Yatırım Asistanı</div>
            
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-icon">🤖</div>
                    <div class="feature-title">AI Destekli Analiz</div>
                    <div class="feature-desc">Gelişmiş makine öğrenimi modelleri</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📊</div>
                    <div class="feature-title">Çoklu Gösterge</div>
                    <div class="feature-desc">50+ teknik gösterge</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🎯</div>
                    <div class="feature-title">Kişiselleştirme</div>
                    <div class="feature-desc">Seviyene göre analiz</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("🚀 ASİSTANI BAŞLAT", use_container_width=True, type="primary"):
            st.session_state['basladi'] = True
            st.rerun()

def show_main_interface():
    """Ana analiz ekranı"""
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.title("⚙️ AI Ayarları")
        
        # Kullanıcı seviyesi
        st.subheader("🎓 Seviye Seçin")
        ai_seviye = st.selectbox(
            "Analiz Dili:",
            ["Acemi", "Orta Düzey", "Profesyonel", "Algoritmik"],
            index=1
        )
        
        st.divider()
        
        # Analiz parametreleri
        st.subheader("📈 Analiz Parametreleri")
        tahmin_periyodu = st.slider("Tahmin Periyodu (gün)", 1, 30, 5)
        risk_toleransi = st.slider("Risk Toleransı", 1, 10, 5)
        
        st.divider()
        
        # Favoriler
        st.subheader("⭐ Favoriler")
        if st.session_state['favoriler']:
            for fav in st.session_state['favoriler'][:5]:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(fav)
                with col2:
                    if st.button("X", key=f"remove_{fav}"):
                        st.session_state['favoriler'].remove(fav)
                        st.rerun()
        else:
            st.info("Henüz favori eklenmedi")
        
        st.divider()
        
        # Çıkış butonu
        if st.button("🔙 Ana Sayfaya Dön"):
            st.session_state['basladi'] = False
            st.rerun()
    
    # --- MAIN CONTENT ---
    st.title("📊 Akıllı Hisse Analizi")
    
    # Arama ve seçim bölümü
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        search_input = st.text_input(
            "Hisse Kodu Ara (Örn: THYAO, GARAN, ASELS)",
            placeholder="Virgülle ayırarak birden fazla girebilirsiniz..."
        )
    with col2:
        st.write("")
        st.write("")
        analyze_btn = st.button("🔍 Analiz Et", use_container_width=True)
    with col3:
        st.write("")
        st.write("")
        if st.button("🔄 Yenile", use_container_width=True):
            st.rerun()
    
    # Hisseleri parse et
    if search_input:
        hisseler = [s.strip().upper() + ".IS" if not s.strip().endswith(".IS") else s.strip().upper() 
                   for s in search_input.split(",") if s.strip()]
    else:
        hisseler = ["THYAO.IS"]  # Varsayılan
    
    if not hisseler:
        st.warning("Lütfen geçerli bir hisse kodu girin.")
        return
    
    # Analiz başlat
    if analyze_btn or hisseler:
        with st.spinner("AI analiz yapıyor..."):
            analyze_stocks(hisseler, ai_seviye, tahmin_periyodu)

def analyze_stocks(hisseler, ai_seviye, tahmin_periyodu):
    """Hisseleri analiz et"""
    
    # İlk hisse için detaylı analiz
    sembol = hisseler[0]
    
    # Veri çek
    fetcher = AdvancedDataFetcher()
    data = fetcher.get_stock_data(sembol)
    
    if not data or data['data'].empty:
        st.error(f"{sembol} için veri bulunamadı.")
        return
    
    df = data['data']
    info = data['info']
    
    # Teknik analiz
    ta_engine = AdvancedTechnicalAnalysis()
    df_with_indicators = ta_engine.calculate_all_indicators(df)
    
    # ML tahminleri
    predictor = AdvancedStockPredictor()
    prediction, confidence, feature_importance = predictor.predict_with_confidence(
        df_with_indicators, 
        horizon=tahmin_periyodu
    )
    
    # AI yorumu
    assistant = AdvancedAIAssistant()
    predictions_data = {
        'prediction': prediction,
        'confidence': confidence,
        'confidence_level': 'Yüksek' if confidence < df['Close'].std() * 0.5 else 'Orta',
        'success_prob': min(95, max(50, 100 - (confidence / df['Close'].iloc[-1] * 100)))
    }
    
    analysis_result = assistant.generate_analysis(
        {'data': df_with_indicators, 'info': info},
        predictions_data,
        ai_seviye
    )
    
    # --- DASHBOARD GÖSTERİMİ ---
    
    # 1. Üst Metrikler
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        current_price = df['Close'].iloc[-1]
        price_change = ((current_price / df['Close'].iloc[-2]) - 1) * 100
        delta_color = "normal" if price_change == 0 else ("positive" if price_change > 0 else "negative")
        st.metric(
            "Güncel Fiyat", 
            f"{current_price:.2f} TL", 
            f"{price_change:+.2f}%",
            delta_color=delta_color
        )
    
    with col2:
        st.metric("RSI", f"{df_with_indicators['RSI'].iloc[-1]:.1f}")
    
    with col3:
        volume_ratio = df['Volume'].iloc[-1] / df['Volume'].rolling(20).mean().iloc[-1]
        st.metric("Hacim Oranı", f"{volume_ratio:.2f}x")
    
    with col4:
        if prediction:
            potential_return = ((prediction / current_price) - 1) * 100
            st.metric("AI Tahmini", f"{prediction:.2f} TL", f"{potential_return:+.1f}%")
    
    # 2. Ana Grafik
    st.subheader("📈 Fiyat Hareketi ve Göstergeler")
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("Fiyat Hareketi", "RSI", "Hacim")
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Fiyat',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # SMA'lar
    fig.add_trace(
        go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='orange', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['SMA_200'], name='SMA 200', line=dict(color='blue', width=1)),
        row=1, col=1
    )
    
    # Bollinger Bands
    if 'BB_Upper' in df_with_indicators.columns:
        fig.add_trace(
            go.Scatter(x=df_with_indicators.index, y=df_with_indicators['BB_Upper'], 
                      name='BB Üst', line=dict(color='gray', width=1, dash='dash'),
                      showlegend=False),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df_with_indicators.index, y=df_with_indicators['BB_Lower'], 
                      name='BB Alt', line=dict(color='gray', width=1, dash='dash'),
                      fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
                      showlegend=False),
            row=1, col=1
        )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=df_with_indicators.index, y=df_with_indicators['RSI'], name='RSI'),
        row=2, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
    
    # Volume
    colors = ['green' if df['Close'].iloc[i] > df['Open'].iloc[i] else 'red' 
              for i in range(len(df))]
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Hacim', marker_color=colors),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Volume'].rolling(20).mean(), 
                  name='Hacim Ort.', line=dict(color='orange', width=1)),
        row=3, col=1
    )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 3. AI Analiz Raporu
    st.subheader("🤖 AI Analiz Raporu")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # AI yorumu
        st.markdown("#### 📝 AI Yorumu")
        st.markdown(analysis_result['analysis'])
        
        # Öneriler
        if analysis_result['recommendations']:
            st.markdown("#### 💡 Öneriler")
            for rec in analysis_result['recommendations']:
                st.info(rec)
    
    with col2:
        # Risk Metrikleri
        st.markdown("#### ⚠️ Risk Analizi")
        
        metrics = analysis_result['metrics']
        
        # Risk skoru
        risk_score = metrics['risk_score']
        risk_color = "green" if risk_score <= 3 else "orange" if risk_score <= 7 else "red"
        st.markdown(DashboardComponents.create_progress_bar(
            "Risk Skoru", risk_score, 10, risk_color
        ), unsafe_allow_html=True)
        
        # Trend gücü
        st.markdown(DashboardComponents.create_progress_bar(
            "Trend Gücü", metrics['trend']['strength_score'], 100, "blue"
        ), unsafe_allow_html=True)
        
        # RSI durumu
        rsi = metrics['rsi']
        rsi_color = "green" if rsi < 30 else "red" if rsi > 70 else "orange"
        st.markdown(DashboardComponents.create_progress_bar(
            "RSI Seviyesi", rsi, 100, rsi_color
        ), unsafe_allow_html=True)
        
        # Favori butonu
        st.write("")
        if sembol in st.session_state['favoriler']:
            if st.button("⭐ Favoriden Çıkar", use_container_width=True):
                st.session_state['favoriler'].remove(sembol)
                st.rerun()
        else:
            if st.button("⭐ Favorilere Ekle", use_container_width=True):
                st.session_state['favoriler'].append(sembol)
                st.rerun()
    
    # 4. Teknik Göstergeler Tablosu
    st.subheader("🔧 Teknik Göstergeler")
    
    if feature_importance:
        # Feature importance grafiği
        importance_df = pd.DataFrame({
            'Özellik': list(feature_importance.keys()),
            'Önem': list(feature_importance.values())
        }).sort_values('Önem', ascending=False).head(10)
        
        fig_importance = go.Figure(
            go.Bar(
                x=importance_df['Önem'],
                y=importance_df['Özellik'],
                orientation='h',
                marker_color='lightblue'
            )
        )
        fig_importance.update_layout(
            height=400,
            title="Özellik Önem Sıralaması",
            xaxis_title="Önem Derecesi",
            yaxis_title="Teknik Gösterge"
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # 5. Çoklu Hisse Karşılaştırma (birden fazla hisse varsa)
    if len(hisseler) > 1:
        st.subheader("📊 Hisse Karşılaştırması")
        
        comparison_data = []
        for sym in hisseler[:5]:  # En fazla 5 hisse
            try:
                sym_data = fetcher.get_stock_data(sym)
                if sym_data and not sym_data['data'].empty:
                    returns = (sym_data['data']['Close'].iloc[-1] / sym_data['data']['Close'].iloc[-30] - 1) * 100
                    comparison_data.append({
                        'Hisse': sym.replace('.IS', ''),
                        'Fiyat': sym_data['data']['Close'].iloc[-1],
                        '30 Gün Getiri': returns,
                        'Hacim': sym_data['data']['Volume'].iloc[-1]
                    })
            except:
                continue
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(
                comparison_df.style.format({
                    'Fiyat': '{:.2f} TL',
                    '30 Gün Getiri': '{:.1f}%',
                    'Hacim': '{:,.0f}'
                }).background_gradient(subset=['30 Gün Getiri'], cmap='RdYlGn'),
                use_container_width=True
            )
    
    # 6. Geçmiş Analizler
    if st.session_state['analiz_gecmisi']:
        with st.expander("📜 Analiz Geçmişi"):
            for entry in st.session_state['analiz_gecmisi'][:10]:
                st.caption(f"{entry['timestamp']} - {entry['sembol']}")
    
    # Analizi geçmişe ekle
    state.add_to_history(sembol, {
        'price': current_price,
        'prediction': prediction,
        'timestamp': datetime.now().isoformat()
    })

# --- Uygulamayı çalıştır ---
if __name__ == "__main__":
    main()
