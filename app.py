import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import ta  # Technical Analysis library
import warnings
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import json
import hashlib
import pickle
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# --- Sayfa Ayarları ---
st.set_page_config(
    page_title="AI Finans V10 - Akıllı Asistan",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourrepo',
        'Report a bug': "https://github.com/yourrepo/issues",
        'About': "# AI Finans V10 - Gelişmiş Yatırım Asistanı"
    }
)

# --- Session State Yönetimi ---
class SessionStateManager:
    def __init__(self):
        self.defaults = {
            'basladi': False,
            'analiz_gecmisi': [],
            'favoriler': [],
            'kullanici_seviyesi': 'Orta Düzey',
            'dark_mode': False,
            'son_guncelleme': None,
            'cache_data': {},
            'model_cache': {}
        }
        
        for key, value in self.defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def add_to_history(self, sembol, sonuc):
        """Analiz geçmişine ekle"""
        entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'sembol': sembol,
            'sonuc': sonuc
        }
        st.session_state['analiz_gecmisi'].insert(0, entry)
        if len(st.session_state['analiz_gecmisi']) > 50:
            st.session_state['analiz_gecmisi'] = st.session_state['analiz_gecmisi'][:50]
    
    def toggle_favorite(self, sembol):
        """Favori ekle/çıkar"""
        if sembol in st.session_state['favoriler']:
            st.session_state['favoriler'].remove(sembol)
        else:
            st.session_state['favoriler'].append(sembol)

state = SessionStateManager()

# --- Cache Sistemi ---
class SmartCache:
    def __init__(self, ttl_minutes=30):
        self.ttl = ttl_minutes * 60
        self.cache_dir = Path(".cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_key(self, func_name, *args, **kwargs):
        """Cache key oluştur"""
        key_str = f"{func_name}_{str(args)}_{str(kwargs)}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, func_name, *args, **kwargs):
        """Cache'den oku"""
        key = self._get_key(func_name, *args, **kwargs)
        cache_file = self.cache_dir / f"{key}.pkl"
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                data, timestamp = pickle.load(f)
                if time.time() - timestamp < self.ttl:
                    return data
        return None
    
    def set(self, func_name, data, *args, **kwargs):
        """Cache'e yaz"""
        key = self._get_key(func_name, *args, **kwargs)
        cache_file = self.cache_dir / f"{key}.pkl"
        
        with open(cache_file, 'wb') as f:
            pickle.dump((data, time.time()), f)
        return data

cache = SmartCache(ttl_minutes=15)

# --- Gelişmiş Veri Yönetimi ---
class AdvancedDataFetcher:
    def __init__(self):
        self.base_urls = {
            'bist': 'https://www.kap.org.tr',
            'news': 'https://api.marketaux.com/v1/news/all'
        }
    
    def get_stock_data(self, sembol, period="2y", interval="1d"):
        """Gelişmiş hisse verisi çekme"""
        cache_key = f"stock_data_{sembol}_{period}"
        cached = cache.get("get_stock_data", sembol, period)
        if cached is not None:
            return cached
        
        try:
            ticker = yf.Ticker(sembol)
            df = ticker.history(period=period, interval=interval)
            
            if len(df) < 50:
                # Daha fazla veri dene
                df = ticker.history(period="5y", interval="1d")
            
            if len(df) > 0:
                # Ek bilgiler
                info = ticker.info
                df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
                df['Daily_Return'] = df['Close'].pct_change()
                df['Volatility'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
                
                result = {
                    'data': df,
                    'info': info,
                    'symbol': sembol,
                    'last_update': datetime.now()
                }
                
                return cache.set("get_stock_data", result, sembol, period)
            
        except Exception as e:
            st.error(f"Veri çekme hatası: {e}")
        
        return None
    
    def get_market_news(self, limit=5):
        """Piyasa haberlerini getir"""
        try:
            # Örnek API kullanımı - gerçek API key gereklidir
            response = requests.get(
                f"https://newsapi.org/v2/everything?q=bist&language=tr&pageSize={limit}&apiKey=YOUR_API_KEY"
            )
            if response.status_code == 200:
                return response.json().get('articles', [])
        except:
            # Fallback haberler
            return [
                {'title': 'BIST 100 Endeksi Analizi', 'source': 'Yerel Kaynak'},
                {'title': 'Dolar/TL Kuru Güncel', 'source': 'Finans Haber'}
            ]
        return []

# --- Gelişmiş Teknik Analiz ---
class AdvancedTechnicalAnalysis:
    def __init__(self):
        self.indicators = {}
    
    def calculate_all_indicators(self, df):
        """Tüm teknik göstergeleri hesapla"""
        df = df.copy()
        
        # Fiyat hareketi
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Hareketli ortalamalar
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
        df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Diff'] = macd.macd_diff()
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_Upper'] = bollinger.bollinger_hband()
        df['BB_Middle'] = bollinger.bollinger_mavg()
        df['BB_Lower'] = bollinger.bollinger_lband()
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # ATR (Volatilite)
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        
        # Volume indicators
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        
        # Support/Resistance levels
        df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['R1'] = 2 * df['Pivot'] - df['Low']
        df['S1'] = 2 * df['Pivot'] - df['High']
        
        # Trend tespiti
        df['Trend_Strength'] = self.calculate_trend_strength(df)
        
        return df.dropna()
    
    def calculate_trend_strength(self, df):
        """Trend gücünü hesapla"""
        # ADX benzeri basit trend gücü
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        
        up_move = df['High'].diff()
        down_move = -df['Low'].diff()
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=14).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=14).mean() / atr)
        
        trend_strength = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        return trend_strength.fillna(50)

# --- Gelişmiş ML Modeli ---
class AdvancedStockPredictor:
    def __init__(self):
        self.models = {
            'rf': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
            'gbr': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        self.scaler = StandardScaler()
        self.feature_importance = {}
    
    def prepare_features(self, df, horizon=5):
        """Özellik mühendisliği"""
        df = df.copy()
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        
        # Rolling statistics
        df['Close_MA_5'] = df['Close'].rolling(window=5).mean()
        df['Close_MA_10'] = df['Close'].rolling(window=10).mean()
        df['Close_Std_10'] = df['Close'].rolling(window=10).std()
        
        # Price changes
        df['Price_Change_1d'] = df['Close'].pct_change(periods=1)
        df['Price_Change_5d'] = df['Close'].pct_change(periods=5)
        
        # Target: Future price (horizon days ahead)
        df['Target'] = df['Close'].shift(-horizon)
        
        # Technical indicators
        df['RSI'] = ta.momentum.rsi(df['Close'])
        df['MACD'] = ta.trend.macd_diff(df['Close'])
        
        # Volume features
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        df.dropna(inplace=True)
        
        # Feature selection
        feature_cols = [col for col in df.columns if col not in ['Target', 'Open', 'High', 'Low', 'Close']]
        features = df[feature_cols]
        target = df['Target']
        
        return features, target, feature_cols
    
    def train_ensemble(self, X_train, y_train):
        """Ensemble model eğit"""
        from sklearn.ensemble import VotingRegressor
        
        # Bireysel modeller
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
        
        # Ensemble model
        ensemble = VotingRegressor([
            ('rf', rf),
            ('gbr', gbr)
        ])
        
        ensemble.fit(X_train, y_train)
        return ensemble
    
    def predict_with_confidence(self, df, horizon=5):
        """Güven aralıklı tahmin"""
        try:
            # Özellik hazırlama
            X, y, feature_cols = self.prepare_features(df, horizon)
            
            if len(X) < 100:
                return None, None, None
            
            # Time-series split
            tscv = TimeSeriesSplit(n_splits=5)
            
            predictions = []
            feature_importances = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Scale
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_val_scaled = self.scaler.transform(X_val)
                
                # Model eğit
                model = self.models['rf']
                model.fit(X_train_scaled, y_train)
                
                # Tahmin
                pred = model.predict(X_val_scaled)
                predictions.extend(pred)
                
                # Feature importance
                feature_importances.append(model.feature_importances_)
            
            # Ortalama feature importance
            avg_importance = np.mean(feature_importances, axis=0)
            self.feature_importance = dict(zip(feature_cols, avg_importance))
            
            # Son model ile tahmin
            X_scaled = self.scaler.fit_transform(X)
            final_model = self.train_ensemble(X_scaled, y)
            
            # Gelecek tahmini
            last_features = X.iloc[-1:].values
            last_features_scaled = self.scaler.transform(last_features)
            prediction = final_model.predict(last_features_scaled)[0]
            
            # Güven aralığı
            confidence = np.std(predictions) if predictions else 0
            
            return prediction, confidence, self.feature_importance
            
        except Exception as e:
            st.error(f"ML hatası: {e}")
            return None, None, None

# --- Gelişmiş AI Asistanı ---
class AdvancedAIAssistant:
    def __init__(self):
        self.personalities = {
            'Acemi': self._beginner_personality,
            'Orta Düzey': self._intermediate_personality,
            'Profesyonel': self._professional_personality,
            'Algoritmik': self._algorithmic_personality
        }
        
        self.sentiment_dict = {
            'positive': ['olumlu', 'yükseliş', 'güçlü', 'fırsat', 'al', 'tavsiye'],
            'negative': ['olumsuz', 'düşüş', 'zayıf', 'risk', 'sat', 'kaçın'],
            'neutral': ['nötr', 'bekle', 'izle', 'konsolide', 'dengeli']
        }
    
    def generate_analysis(self, stock_data, predictions, user_level='Orta Düzey'):
        """Kişiselleştirilmiş analiz oluştur"""
        df = stock_data['data']
        info = stock_data['info']
        
        son_fiyat = df['Close'].iloc[-1]
        son_volume = df['Volume'].iloc[-1]
        rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
        
        # Trend analizi
        trend = self._analyze_trend(df)
        momentum = self._analyze_momentum(df)
        volatility = self._analyze_volatility(df)
        volume_analysis = self._analyze_volume(df)
        
        # Risk skoru
        risk_score = self._calculate_risk_score(df, rsi, volatility)
        
        # Kişiliğe göre yorum
        personality_func = self.personalities.get(user_level, self._intermediate_personality)
        analysis = personality_func(
            son_fiyat=son_fiyat,
            trend=trend,
            momentum=momentum,
            volatility=volatility,
            volume=volume_analysis,
            rsi=rsi,
            risk_score=risk_score,
            predictions=predictions,
            info=info
        )
        
        # Öneriler ekle
        recommendations = self._generate_recommendations(
            trend, momentum, rsi, risk_score, user_level
        )
        
        return {
            'analysis': analysis,
            'recommendations': recommendations,
            'metrics': {
                'trend': trend,
                'momentum': momentum,
                'volatility': volatility,
                'risk_score': risk_score,
                'rsi': rsi
            }
        }
    
    def _beginner_personality(self, **kwargs):
        """Yeni başlayanlar için basit dil"""
        text = f"""
        🤖 **AI Asistan Diyor Ki:**
        
        📊 **Hisse Durumu:** {kwargs['son_fiyat']:.2f} TL seviyesinde işlem görüyor.
        
        📈 **Trend:** {self._translate_trend(kwargs['trend'], 'basit')}
        
        💪 **Momentum:** {self._translate_momentum(kwargs['momentum'], 'basit')}
        
        📉 **RSI Göstergesi:** {kwargs['rsi']:.1f} - {self._translate_rsi(kwargs['rsi'], 'basit')}
        
        ⚠️ **Risk Seviyesi:** {kwargs['risk_score']}/10 - {self._translate_risk(kwargs['risk_score'], 'basit')}
        
        🔮 **AI Tahmini:** {kwargs['predictions'].get('prediction', 0):.2f} TL 
        ({((kwargs['predictions'].get('prediction', 0)/kwargs['son_fiyat'])-1)*100:.1f}%)
        """
        return text
    
    def _intermediate_personality(self, **kwargs):
        """Orta düzey yatırımcılar"""
        text = f"""
        📈 **Teknik Analiz Raporu:**
        
        **🎯 Temel Veriler:**
        • Fiyat: {kwargs['son_fiyat']:.2f} TL
        • Trend Yapı: {kwargs['trend']['direction']} ({kwargs['trend']['strength']})
        • Momentum: {kwargs['momentum']['status']}
        • Volatilite: {kwargs['volatility']['level']}
        
        **📊 Göstergeler:**
        • RSI(14): {kwargs['rsi']:.1f} - {self._get_rsi_signal(kwargs['rsi'])}
        • Trend Gücü: {kwargs['trend']['strength_score']}/100
        • Risk Skoru: {kwargs['risk_score']}/10
        
        **🤖 AI Öngörüsü:**
        • 5 Günlük Tahmin: {kwargs['predictions'].get('prediction', 0):.2f} TL
        • Potansiyel Getiri: {((kwargs['predictions'].get('prediction', 0)/kwargs['son_fiyat'])-1)*100:.1f}%
        • Güven Seviyesi: {kwargs['predictions'].get('confidence_level', 'Orta')}
        """
        return text
    
    def _professional_personality(self, **kwargs):
        """Profesyonel trader'lar için"""
        text = f"""
        🔬 **Derin Analiz Raporu:**
        
        **Market Structure Analysis:**
        • Price Action: {kwargs['trend']['structure']}
        • Key Levels: S1: {kwargs.get('support', 'N/A')} | R1: {kwargs.get('resistance', 'N/A')}
        • Volume Profile: {kwargs['volume']['anomaly']}
        
        **Technical Metrics:**
        • RSI(14): {kwargs['rsi']:.1f} → {self._get_rsi_zone(kwargs['rsi'])}
        • MACD Signal: {self._get_macd_signal(kwargs.get('macd', 0))}
        • ATR Ratio: {kwargs['volatility'].get('atr_ratio', 0):.3f}
        • Bollinger Position: {self._get_bb_position(kwargs.get('bb_position', 'middle'))}
        
        **Risk Assessment:**
        • Value at Risk (1-day): {kwargs['risk_score']*10:.1f}%
        • Sharpe Ratio: {kwargs.get('sharpe', 0):.2f}
        • Maximum Drawdown: {kwargs.get('max_dd', 0):.1f}%
        
        **AI Ensemble Prediction:**
        • Target Price (5D): {kwargs['predictions'].get('prediction', 0):.2f} 
        • Confidence Interval: ±{kwargs['predictions'].get('confidence', 0):.2f}
        • Probability of Success: {kwargs['predictions'].get('success_prob', 0):.1f}%
        """
        return text
    
    def _algorithmic_personality(self, **kwargs):
        """Algoritmik trading için"""
        # JSON formatında yapılandırılmış veri
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "symbol": kwargs.get('symbol', 'UNKNOWN'),
            "signals": {
                "trend": kwargs['trend']['direction'],
                "momentum": kwargs['momentum']['status'],
                "rsi_signal": self._get_rsi_signal(kwargs['rsi']),
                "volume_signal": kwargs['volume']['signal']
            },
            "predictions": kwargs['predictions'],
            "risk_metrics": {
                "score": kwargs['risk_score'],
                "var": kwargs['risk_score'] * 10,
                "sharpe": kwargs.get('sharpe', 0)
            },
            "trading_suggestions": self._generate_algo_suggestions(kwargs)
        }
        return json.dumps(analysis, indent=2, ensure_ascii=False)
    
    def _analyze_trend(self, df):
        """Trend analizi"""
        sma_50 = df['SMA_50'].iloc[-1] if 'SMA_50' in df.columns else df['Close'].iloc[-1]
        sma_200 = df['SMA_200'].iloc[-1] if 'SMA_200' in df.columns else df['Close'].iloc[-1]
        
        price = df['Close'].iloc[-1]
        trend_score = 0
        
        # Golden/Death Cross kontrolü
        if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
            if df['SMA_50'].iloc[-1] > df['SMA_200'].iloc[-1]:
                trend_score += 30
            else:
                trend_score -= 20
        
        # Price vs MA
        if price > sma_50:
            trend_score += 20
        if price > sma_200:
            trend_score += 30
        
        # Slope analizi
        if len(df) > 20:
            slope = np.polyfit(range(20), df['Close'].iloc[-20:].values, 1)[0]
            trend_score += slope * 1000
        
        direction = "YUKARI" if trend_score > 0 else "AŞAĞI" if trend_score < 0 else "YATAY"
        
        return {
            'direction': direction,
            'strength': abs(trend_score),
            'strength_score': min(100, abs(trend_score)),
            'structure': self._determine_structure(df)
        }
    
    def _analyze_momentum(self, df):
        """Momentum analizi"""
        if 'RSI' not in df.columns:
            return {'status': 'NÖTR', 'value': 50}
        
        rsi = df['RSI'].iloc[-1]
        
        if rsi > 70:
            status = "AŞIRI ALIM"
        elif rsi < 30:
            status = "AŞIRI SATIM"
        elif rsi > 55:
            status = "YUKARI"
        elif rsi < 45:
            status = "AŞAĞI"
        else:
            status = "NÖTR"
        
        return {'status': status, 'value': rsi}
    
    def _analyze_volume(self, df):
        """Hacim analizi"""
        if 'Volume' not in df.columns:
            return {'level': 'NORMAL', 'anomaly': False}
        
        volume = df['Volume'].iloc[-1]
        avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        
        if volume_ratio > 2:
            level = "ÇOK YÜKSEK"
            anomaly = True
        elif volume_ratio > 1.5:
            level = "YÜKSEK"
            anomaly = True
        elif volume_ratio < 0.5:
            level = "DÜŞÜK"
            anomaly = True
        else:
            level = "NORMAL"
            anomaly = False
        
        return {
            'level': level,
            'ratio': volume_ratio,
            'anomaly': anomaly,
            'signal': 'ALARM' if anomaly else 'NORMAL'
        }
    
    def _calculate_risk_score(self, df, rsi, volatility):
        """Risk skoru hesapla (1-10)"""
        score = 5  # Başlangıç
        
        # Volatilite
        if volatility.get('level') == 'YÜKSEK':
            score += 2
        elif volatility.get('level') == 'DÜŞÜK':
            score -= 1
        
        # RSI
        if rsi > 70 or rsi < 30:
            score += 1
        
        # Volume anomalisi
        volume_analysis = self._analyze_volume(df)
        if volume_analysis['anomaly']:
            score += 1
        
        # Trend zayıflığı
        trend = self._analyze_trend(df)
        if trend['strength_score'] < 30:
            score += 1
        
        return min(10, max(1, score))
    
    def _generate_recommendations(self, trend, momentum, rsi, risk_score, user_level):
        """Seviyeye göre öneriler"""
        recommendations = []
        
        if user_level == 'Acemi':
            if risk_score >= 7:
                recommendations.append("⚠️ Yüksek risk! Küçük pozisyonlarla dene.")
            elif rsi < 30:
                recommendations.append("🛒 İndirim bölgesi! Uzun vade düşünebilirsin.")
            elif rsi > 70:
                recommendations.append("💰 Kâr almayı düşün! Aşırı alım bölgesi.")
        else:
            # Profesyonel öneriler
            if momentum['status'] == 'AŞIRI ALIM' and trend['direction'] == 'YUKARI':
                recommendations.append("📉 Kısa vadeli düzeltme beklenebilir.")
            if momentum['status'] == 'AŞIRI SATIM' and trend['direction'] == 'AŞAĞI':
                recommendations.append("📈 Potansiyel dip alım fırsatı.")
        
        return recommendations

# --- Dashboard Bileşenleri ---
class DashboardComponents:
    @staticmethod
    def create_metric_card(title, value, delta=None, delta_type="normal"):
        """Metrik kartı oluştur"""
        colors = {
            "positive": "green",
            "negative": "red",
            "normal": "blue"
        }
        
        delta_color = colors.get(delta_type, "blue")
        
        card = f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            padding: 20px;
            color: white;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        ">
            <div style="font-size: 14px; opacity: 0.9;">{title}</div>
            <div style="font-size: 32px; font-weight: bold; margin: 10px 0;">{value}</div>
            {f'<div style="font-size: 14px; color: {delta_color};">{delta}</div>' if delta else ''}
        </div>
        """
        return card
    
    @staticmethod
    def create_alert(message, type="info"):
        """Alert mesajı oluştur"""
        icons = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌"
        }
        
        colors = {
            "info": "#2196F3",
            "success": "#4CAF50",
            "warning": "#FF9800",
            "error": "#F44336"
        }
        
        return f"""
        <div style="
            background-color: {colors.get(type, '#2196F3')}20;
            border-left: 4px solid {colors.get(type, '#2196F3')};
            padding: 12px;
            border-radius: 4px;
            margin: 10px 0;
            display: flex;
            align-items: center;
        ">
            <span style="font-size: 20px; margin-right: 10px;">{icons.get(type, 'ℹ️')}</span>
            <span>{message}</span>
        </div>
        """
    
    @staticmethod
    def create_progress_bar(label, value, max_value=100, color="blue"):
        """Progress bar oluştur"""
        percentage = (value / max_value) * 100
        colors = {
            "blue": "#2196F3",
            "green": "#4CAF50",
            "red": "#F44336",
            "orange": "#FF9800",
            "purple": "#9C27B0"
        }
        
        return f"""
        <div style="margin: 15px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span>{label}</span>
                <span>{value:.1f}/{max_value}</span>
            </div>
            <div style="
                width: 100%;
                height: 10px;
                background-color: #e0e0e0;
                border-radius: 5px;
                overflow: hidden;
            ">
                <div style="
                    width: {percentage}%;
                    height: 100%;
                    background-color: {colors.get(color, '#2196F3')};
                    border-radius: 5px;
                "></div>
            </div>
        </div>
        """

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
            
            <button style="
                background: white;
                color: #764ba2;
                border: none;
                padding: 15px 40px;
                font-size: 1.2rem;
                border-radius:
