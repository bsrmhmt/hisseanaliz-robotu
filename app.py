import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import time
import os

# --- LlamaIndex (Yapay Zeka) Kütüphaneleri ---
try:
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
    from llama_index.llms.ollama import Ollama
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# --- Sayfa Ayarları ---
st.set_page_config(page_title="AI Super App V10", layout="wide", initial_sidebar_state="expanded")

# --- Session State ---
if 'messages' not in st.session_state:
    st.session_state.messages = []

# ==========================================
# MODÜL 1: HİSSE ANALİZ FONKSİYONLARI
# ==========================================
def veri_getir(sembol):
    try:
        df = yf.Ticker(sembol).history(period="2y") 
        if len(df) < 50: return pd.DataFrame()
        return df
    except: return pd.DataFrame()

def indikatorler(df):
    df = df.copy()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df.dropna(inplace=True)
    return df

# ==========================================
# MODÜL 2: DOKÜMAN AI MOTORU (LOCAL RAG)
# ==========================================
@st.cache_resource
def ai_motorunu_baslat():
    """Ollama ve Embedding modellerini hafızaya yükler"""
    if not AI_AVAILABLE: return None
    
    # 1. Beyin: Ollama (Llama3)
    llm = Ollama(model="llama3", request_timeout=300.0)
    
    # 2. Çevirmen: HuggingFace (Metni sayıya çevirir)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # Ayarları kaydet
    Settings.llm = llm
    Settings.embed_model = embed_model
    return True

def dokuman_isle(uploaded_file):
    """Yüklenen PDF'i okur ve vektör veritabanına atar"""
    if not os.path.exists("temp_data"):
        os.makedirs("temp_data")
    
    file_path = os.path.join("temp_data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    reader = SimpleDirectoryReader(input_dir="temp_data")
    documents = reader.load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index.as_query_engine()

# ==========================================
# ARAYÜZ MİMARİSİ
# ==========================================

# --- Kenar Çubuğu: Mod Seçimi ---
st.sidebar.title("🎛️ Kontrol Paneli")
uygulama_modu = st.sidebar.radio("Uygulama Modu Seç:", ["📈 Hisse Analiz Robotu", "🧠 Doküman Asistanı (AI)"])

st.sidebar.markdown("---")

# ------------------------------------------
# MOD 1: HİSSE ANALİZİ (Eski V9 Kodun)
# ------------------------------------------
if uygulama_modu == "📈 Hisse Analiz Robotu":
    st.title("📈 Borsa Analiz Terminali")
    
    hisse = st.text_input("Hisse Kodu Girin:", "THYAO")
    if st.button("Analiz Et"):
        df = veri_getir(hisse + ".IS")
        if not df.empty:
            df = indikatorler(df)
            son = df.iloc[-1]
            
            c1, c2 = st.columns(2)
            c1.metric("Fiyat", f"{son['Close']:.2f} TL")
            c1.metric("RSI", f"{son['RSI']:.2f}")
            
            fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
            st.plotly_chart(fig, use_container_width=True)
            
            if son['RSI'] < 30: st.success("AI Yorumu: Aşırı satım bölgesinde, tepki gelebilir.")
            elif son['RSI'] > 70: st.error("AI Yorumu: Aşırı alım bölgesinde, dikkatli ol.")
            else: st.info("AI Yorumu: Nötr bölgede seyrediyor.")
        else:
            st.error("Veri bulunamadı.")

# ------------------------------------------
# MOD 2: DOKÜMAN ASİSTANI (YENİ ÖZELLİK)
# ------------------------------------------
elif uygulama_modu == "🧠 Doküman Asistanı (AI)":
    st.title("🧠 Yapay Zeka ile Doküman Sohbeti")
    st.caption("Kendi bilgisayarınızda çalışan Llama3 modeli ile PDF'lerinizi analiz edin. (İnternet gerekmez)")

    if not AI_AVAILABLE:
        st.error("⚠️ Gerekli kütüphaneler eksik! Lütfen 'pip install llama-index-core llama-index-llms-ollama llama-index-embeddings-huggingface' yapın.")
    else:
        # Motoru Başlat
        ai_motorunu_baslat()
        
        # Dosya Yükleme
        uploaded_file = st.file_uploader("Bir Finansal Rapor (PDF) Yükleyin", type=['pdf', 'txt'])
        
        # Eğit Butonu
        if uploaded_file and st.button("🚀 AI'yı Bu Dosyayla Eğit"):
            with st.spinner("AI dosyayı okuyor ve öğreniyor... (Bu işlem bilgisayar hızınıza göre sürebilir)"):
                st.session_state.query_engine = dokuman_isle(uploaded_file)
                st.success("Eğitim Tamamlandı! Artık soru sorabilirsiniz.")
        
        # Chat Arayüzü
        st.markdown("---")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Bu rapor hakkında ne bilmek istersin?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            if "query_engine" in st.session_state:
                with st.chat_message("assistant"):
                    with st.spinner("Llama3 Düşünüyor..."):
                        response = st.session_state.query_engine.query(prompt)
                        st.markdown(response.response)
                        st.session_state.messages.append({"role": "assistant", "content": response.response})
            else:
                st.warning("Lütfen önce bir dosya yükleyin ve eğitin.")
