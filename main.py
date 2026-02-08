# =====================================================
# PREDIKSI HARGA SAHAM PT BANK CENTRAL ASIA TBK (BBCA)
# MENGGUNAKAN ALGORITMA EXTREME GRADIENT BOOSTING
# ENHANCED VERSION - IMPROVED ROBUSTNESS
# =====================================================
# Nama    : Alexius Kenriko Salim
# NIM     : 22101152630094
# Program : Teknik Informatika
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px
import pickle
import json
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta
import warnings
import inspect
import html
import textwrap
warnings.filterwarnings('ignore')


def render_html(s: str):
    """Render HTML safely in Streamlit without Markdown code-block indentation issues."""
    st.markdown(textwrap.dedent(s).strip(), unsafe_allow_html=True)

# =====================================================
# TOOLTIP HELPERS (CUSTOM, COLORABLE)
# =====================================================
def tooltip_span(tooltip: str, icon: str = "ⓘ", width_px: int | None = None) -> str:
    """Return an info icon with a browser tooltip (title=...). Safe: no markdown code-block issues."""
    tip = html.escape(tooltip, quote=True)
    ico = html.escape(icon, quote=True)
    return f"<span class='cg-tipicon' title='{tip}' aria-label='{tip}'>{ico}</span>"

def metric_with_tooltip(label: str, value: str, tooltip: str, delta: str | None = None):
    """Render a metric card with a fully stylable tooltip."""
    safe_label = html.escape(label, quote=True)
    safe_value = html.escape(str(value), quote=True)
    tip_html = tooltip_span(tooltip)

    delta_html = ""
    if delta is not None:
        safe_delta = html.escape(str(delta), quote=True)
        delta_html = f"<div class='metric-delta'>{safe_delta}</div>"

    render_html(f"""
        <div class='metric-card'>
            <div class='metric-top'>
                <div class='metric-label'>{safe_label}</div>
                {tip_html}
            </div>
            <div class='metric-value'>{safe_value}</div>
            {delta_html}
        </div>
        """)

# =====================================================
# KONFIGURASI HORIZON PREDIKSI
# =====================================================
HORIZON_DAYS = 7  # jumlah hari prediksi (Direct Multi-Step)


# =====================================================
# KONFIGURASI HALAMAN
# =====================================================
st.set_page_config(
    page_title="Prediksi Saham BBCA",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - New Color Scheme
st.markdown("""
<style>

/* Browser-tooltip icon (title=...) */
.cg-tipicon{
  color: #00d4ff;
  font-weight: 800;
  cursor: help;
  user-select: none;
  padding: 2px 6px;
  border-radius: 8px;
  border: 1px solid rgba(0, 212, 255, 0.55);
  background: rgba(0, 212, 255, 0.10);
  line-height: 1;
  font-size: 0.9rem;
}
    /* =========================
       GOOGLE FONT
    ========================= */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    /* =========================
      MAIN APP BACKGROUND (FINAL FIX)
    ========================= */
    [data-testid="stAppViewContainer"] {
        background-color: #0e123f !important;
    }

    [data-testid="stAppViewContainer"] > div:first-child {
        background-color: #0e123f !important;
    }

    .block-container {
        background-color: #0e123f !important;
    }

    /* =========================
      TOP BAR
    ========================= */
    [data-testid="stHeader"],
    [data-testid="stToolbar"] {
        background-color: #1e2657 !important;
    }

    /* =========================
      SIDEBAR
    ========================= */
    [data-testid="stSidebar"] {
        background-color: #283061 !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        background-color: #283061 !important;
    }

    [data-testid="stSidebar"] * {
        color: #e8e8e8 !important;
    }

    /* Sidebar Radio */
    [data-testid="stSidebar"] .stRadio > label {
        color: #ffffff !important;
        font-weight: 500;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        color: #e8e8e8 !important;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        transition: all 0.25s ease;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background-color: #3a4578;
    }

    /* =========================
      HEADERS
    ========================= */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #60a5fa;
        text-align: center;
        padding: 1.5rem 0;
        border-bottom: 3px solid #60a5fa;
        margin-bottom: 1rem;
        text-shadow: 0 0 20px rgba(96, 165, 250, 0.3);
    }

    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #60a5fa;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-left: 4px solid #60a5fa;
        padding-left: 1rem;
    }

    /* =========================
      TEXT
    ========================= */
    h1, h2, h3, h4, h5, h6 {
        color: #60a5fa !important;
    }

    p, span, div, label {
        color: #e8e8e8;
    }

    /* =========================
      INFO / CARD BOX
    ========================= */
    .info-box,
    .prediction-card,
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2657 0%, #283061 100%);
        border-radius: 1rem;
        border: 1px solid #3a4578;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        padding: 1rem; 
    }

    /* Prediction card hover */
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(96, 165, 250, 0.4);
    }

    .prediction-value {
        font-size: 2rem;
        font-weight: 700;
        color: #60a5fa;
    }

    /* =========================
      METRICS
    ========================= */
    [data-testid="stMetricValue"] {
        color: #60a5fa !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #a8a8a8 !important;
    }

    [data-testid="stMetricDelta"] {
        color: #4ade80 !important;
    }

    /* =========================
      TABLES
    ========================= */
    .dataframe {
        background-color: #1e2657 !important;
        color: #e8e8e8 !important;
        border: 1px solid #3a4578 !important;
    }

    .dataframe th {
        background-color: #283061 !important;
        color: #60a5fa !important;
        border: 1px solid #3a4578 !important;
    }

    .dataframe td {
        background-color: #1e2657 !important;
        color: #e8e8e8 !important;
        border: 1px solid #3a4578 !important;
    }

    /* =========================
      BUTTONS
    ========================= */
    .stButton > button {
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
        color: #ffffff;
        font-weight: 600;
        border-radius: 0.5rem;
        padding: 0.75rem 2rem;
        border: none;
        box-shadow: 0 4px 6px rgba(96, 165, 250, 0.3);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        transform: translateY(-2px);
    }

    /* === Form Submit Button === */
    div[data-testid="stFormSubmitButton"] > button {
        background-color: #111985 !important;
        color: #1a1a2e !important;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        border: none;
    }

    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #212885 !important;
        color: #1a1a2e !important;
    }

    /* =========================
      INPUTS
    ========================= */
    .stNumberInput input,
    .stTextInput input {
        background-color: #1e2657 !important;
        color: #e8e8e8 !important;
        border: 1px solid #3a4578 !important;
    }

    /* =========================
      SCROLLBAR
    ========================= */
    ::-webkit-scrollbar {
        width: 10px;
    }

    ::-webkit-scrollbar-track {
        background: #1e2657;
    }

    ::-webkit-scrollbar-thumb {
        background: #3a4578;
        border-radius: 5px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #60a5fa;
    }

    /* =========================
      DIVIDER
    ========================= */
    hr {
        border-color: #3a4578 !important;
        margin: 2rem 0 !important;
    }
    
    /* Sidebar Divider */
    [data-testid="stSidebar"] hr {
            margin-top: 0.1rem !important;
            margin-bottom: 0.3rem !important;
            border-color: #3a4578;
    }
            
    [data-testid="stLogoSpacer"],
    [data-testid="stLogoSpacer"]:hover {
            height: 4px !important;
            min-height: 4px !important;
            padding: 1rem !important;
    }


    [data-testid="stSidebarHeader"],
    [data-testid="stSidebarHeader"]:hover,
    [data-testid="stSidebarHeader"]:focus {
            height: 28px !important;
            min-height: 28px !important;
            max-height: 28px !important;
            padding: 0 !important;
    }        

    /* =========================
      CUSTOM TOOLTIP (COLORABLE)
    ========================= */
    .cg-tt {
        position: relative;
        display: inline-block;
        line-height: 1;
    }
    .cg-tt-icon {
        color: #00d4ff;
        font-weight: 800;
        cursor: help;
        user-select: none;
        font-size: 0.95rem;
        padding: 2px 4px;
        border-radius: 6px;
        background: rgba(0, 212, 255, 0.10);
        border: 1px solid rgba(0, 212, 255, 0.25);
    }
    .cg-tt-bubble {
        visibility: hidden;
        opacity: 0;
        transition: opacity .15s ease-in-out;
        position: absolute;
        z-index: 99999;
        width: var(--tt-width, 340px);
        background: #1e2657;
        color: #e8e8e8;
        border: 1px solid #00d4ff;
        border-radius: 12px;
        padding: 10px 12px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.40);
        top: 140%;
        right: 0;
        text-align: left;
        font-size: 0.85rem;
        line-height: 1.35;
        pointer-events: none;
    }
    .cg-tt-bubble::after {
        content: "";
        position: absolute;
        top: -8px;
        right: 14px;
        border-width: 0 8px 8px 8px;
        border-style: solid;
        border-color: transparent transparent #1e2657 transparent;
    }
    .cg-tt:hover .cg-tt-bubble {
        visibility: visible;
        opacity: 1;
    }

    /* =========================
      METRIC CARDS (CUSTOM)
    ========================= */
    .metric-card {
        background: #0f3460;
        border: 1px solid rgba(0, 212, 255, 0.35);
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    .metric-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        margin-bottom: 6px;
    }
    .metric-label {
        color: #e8e8e8;
        font-size: 0.9rem;
        font-weight: 600;
        opacity: 0.95;
    }
    .metric-value {
        color: #00d4ff;
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: 0.2px;
    }
    .metric-delta {
        margin-top: 6px;
        font-size: 0.85rem;
        color: #a8a8a8;
    }

    </style>
    """, unsafe_allow_html=True)


# =====================================================
# FUNGSI UNTUK LOAD DAN PROSES DATA
# =====================================================
@st.cache_data
def load_and_process_data():
    """Load dan preprocessing data dengan enhanced features"""
    
    # Load data
    df = pd.read_csv(
        'data/data_historis_BBCA.csv',
        thousands='.',
        decimal=','
    )
    
    # Rename kolom
    df = df.rename(columns={
        'Tanggal': 'Date',
        'Pembukaan': 'Open',
        'Tertinggi': 'High',
        'Terendah': 'Low',
        'Terakhir': 'Close',
        'Vol.': 'Volume',
        'Perubahan%': 'ChangePercent'
    })
    
    # Konversi volume
    def convert_volume(vol):
        if isinstance(vol, str):
            if 'M' in vol:
                return float(vol.replace('M','').replace(',','.')) * 1_000_000
            elif 'K' in vol:
                return float(vol.replace('K','').replace(',','.')) * 1_000
            else:
                return float(vol.replace(',','.'))
        return vol
    
    df['Volume'] = df['Volume'].apply(convert_volume)
    
    # Parsing tanggal dan pengurutan
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.drop(columns=['ChangePercent'])
    df = df.sort_values('Date')
    df = df.set_index('Date')
    
    # ==========================================
    # FEATURE ENGINEERING - BASIC
    # ==========================================
    df['return_1'] = df['Close'].pct_change()
    df['hl_range'] = (df['High'] - df['Low']) / df['Close']
    df['oc_change'] = (df['Close'] - df['Open']) / df['Open']
    
    for i in range(1, 11):
        df[f'Close_lag{i}'] = df['Close'].shift(i)
    
    for i in range(1, 6):
        df[f'return_lag{i}'] = df['return_1'].shift(i)
    
    df['rolling_mean_5'] = df['Close'].rolling(5).mean()
    df['rolling_std_5'] = df['Close'].rolling(5).std()
    
    # ==========================================
    # FEATURE ENGINEERING - ENHANCED
    # ==========================================
    
    # 1. Volatility Features
    df['rolling_std_10'] = df['Close'].rolling(10).std()
    df['rolling_std_20'] = df['Close'].rolling(20).std()
    df['volatility_ratio'] = df['rolling_std_5'] / (df['rolling_std_20'] + 1e-8)
    df['volatility_change'] = df['rolling_std_5'].pct_change()
    
    # 2. Momentum Features
    df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
    df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
    df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
    df['return_acceleration'] = df['return_1'] - df['return_lag1']
    df['momentum_change'] = df['momentum_5'] - df['momentum_5'].shift(1)
    
    # 3. Volume Features
    df['volume_change'] = df['Volume'].pct_change()
    df['volume_ma_5'] = df['Volume'].rolling(5).mean()
    df['volume_ma_20'] = df['Volume'].rolling(20).mean()
    df['volume_ma_ratio'] = df['Volume'] / (df['volume_ma_5'] + 1e-8)
    df['volume_surge'] = (df['Volume'] > df['volume_ma_20'] * 1.5).astype(int)
    
    # 4. Range Features
    df['range_pct'] = (df['High'] - df['Low']) / df['Open']
    df['range_change'] = df['hl_range'] / (df['hl_range'].shift(1) + 1e-8) - 1
    df['range_ma_5'] = df['hl_range'].rolling(5).mean()
    df['range_expansion'] = df['hl_range'] / (df['range_ma_5'] + 1e-8)
    
    # 5. Price Position Features
    df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-8)
    df['price_to_ma5'] = df['Close'] / (df['rolling_mean_5'] + 1e-8) - 1
    df['price_to_ma10'] = df['Close'] / (df['Close'].rolling(10).mean() + 1e-8) - 1
    
    # 6. Trend Consistency Features
    df['consecutive_up'] = (df['Close'] > df['Close'].shift(1)).astype(int)
    for i in range(2, 6):
        df['consecutive_up'] += (df['Close'].shift(i-1) > df['Close'].shift(i)).astype(int)
    
    # 7. Gap Features
    df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    df['gap_filled'] = ((df['gap'] > 0) & (df['Low'] <= df['Close'].shift(1))).astype(int)
    
    # Target
    df['Target_1hari'] = df['Close'].shift(-1)
    df['Target_1minggu'] = df['Close'].shift(-5)
    
    df = df.dropna()
    
    return df


@st.cache_resource
def load_trained_models():
    """Load model yang sudah dilatih dari file pickle"""
    
    model_path = 'models/trained_models.pkl'
    params_path = 'models/best_parameters.json'
    eval_path = 'models/evaluation_results.json'
    
    if not os.path.exists(model_path):
        st.error("❌ Model belum dilatih! Silakan jalankan 'model.py' terlebih dahulu.")
        st.stop()
    
    # Load model
    with open(model_path, 'rb') as f:
        results = pickle.load(f)
    
    # Load best parameters
    best_params = None
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            best_params = json.load(f)
    
    # Load evaluation results
    eval_results = None
    if os.path.exists(eval_path):
        with open(eval_path, 'r') as f:
            eval_results = json.load(f)
    
    return results, best_params, eval_results


def create_features_from_manual_input(open_price, high_price, low_price, close_price, volume, df, fitur):
    """
    Membuat fitur lengkap dari input manual user dengan enhanced features
    """
    # Ambil data historis (30 baris terakhir untuk konteks yang lebih panjang)
    historical_data = df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(30).copy()
    
    # Tambahkan data input manual sebagai data terakhir
    new_row = pd.DataFrame({
        'Open': [open_price],
        'High': [high_price],
        'Low': [low_price],
        'Close': [close_price],
        'Volume': [volume]
    }, index=[historical_data.index[-1] + timedelta(days=1)])
    
    combined_data = pd.concat([historical_data, new_row])
    
    # Hitung fitur dasar
    combined_data['return_1'] = combined_data['Close'].pct_change()
    combined_data['hl_range'] = (combined_data['High'] - combined_data['Low']) / combined_data['Close']
    combined_data['oc_change'] = (combined_data['Close'] - combined_data['Open']) / combined_data['Open']
    
    for i in range(1, 11):
        combined_data[f'Close_lag{i}'] = combined_data['Close'].shift(i)
    
    for i in range(1, 6):
        combined_data[f'return_lag{i}'] = combined_data['return_1'].shift(i)
    
    combined_data['rolling_mean_5'] = combined_data['Close'].rolling(5).mean()
    combined_data['rolling_std_5'] = combined_data['Close'].rolling(5).std()
    
    # Hitung fitur enhanced
    combined_data['rolling_std_10'] = combined_data['Close'].rolling(10).std()
    combined_data['rolling_std_20'] = combined_data['Close'].rolling(20).std()
    combined_data['volatility_ratio'] = combined_data['rolling_std_5'] / (combined_data['rolling_std_20'] + 1e-8)
    combined_data['volatility_change'] = combined_data['rolling_std_5'].pct_change()
    
    combined_data['momentum_5'] = combined_data['Close'] / combined_data['Close'].shift(5) - 1
    combined_data['momentum_10'] = combined_data['Close'] / combined_data['Close'].shift(10) - 1
    combined_data['momentum_20'] = combined_data['Close'] / combined_data['Close'].shift(20) - 1
    combined_data['return_acceleration'] = combined_data['return_1'] - combined_data['return_lag1']
    combined_data['momentum_change'] = combined_data['momentum_5'] - combined_data['momentum_5'].shift(1)
    
    combined_data['volume_change'] = combined_data['Volume'].pct_change()
    combined_data['volume_ma_5'] = combined_data['Volume'].rolling(5).mean()
    combined_data['volume_ma_20'] = combined_data['Volume'].rolling(20).mean()
    combined_data['volume_ma_ratio'] = combined_data['Volume'] / (combined_data['volume_ma_5'] + 1e-8)
    combined_data['volume_surge'] = (combined_data['Volume'] > combined_data['volume_ma_20'] * 1.5).astype(int)
    
    combined_data['range_pct'] = (combined_data['High'] - combined_data['Low']) / combined_data['Open']
    combined_data['range_change'] = combined_data['hl_range'] / (combined_data['hl_range'].shift(1) + 1e-8) - 1
    combined_data['range_ma_5'] = combined_data['hl_range'].rolling(5).mean()
    combined_data['range_expansion'] = combined_data['hl_range'] / (combined_data['range_ma_5'] + 1e-8)
    
    combined_data['close_position'] = (combined_data['Close'] - combined_data['Low']) / (combined_data['High'] - combined_data['Low'] + 1e-8)
    combined_data['price_to_ma5'] = combined_data['Close'] / (combined_data['rolling_mean_5'] + 1e-8) - 1
    combined_data['price_to_ma10'] = combined_data['Close'] / (combined_data['Close'].rolling(10).mean() + 1e-8) - 1
    
    combined_data['consecutive_up'] = (combined_data['Close'] > combined_data['Close'].shift(1)).astype(int)
    for i in range(2, 6):
        combined_data['consecutive_up'] += (combined_data['Close'].shift(i-1) > combined_data['Close'].shift(i)).astype(int)
    
    combined_data['gap'] = (combined_data['Open'] - combined_data['Close'].shift(1)) / combined_data['Close'].shift(1)
    combined_data['gap_filled'] = ((combined_data['gap'] > 0) & (combined_data['Low'] <= combined_data['Close'].shift(1))).astype(int)
    
    # Ambil baris terakhir (data input manual dengan fitur lengkap)
    return combined_data[fitur].iloc[-1:].values


def create_features_for_prediction(df, fitur):
    """Membuat fitur untuk prediksi dari data terbaru dengan enhanced features"""
    # Ambil data terakhir (lebih banyak untuk fitur yang membutuhkan window lebih panjang)
    latest_data = df.tail(30).copy()
    
    # Fitur sudah ada di df, tinggal ambil yang terakhir
    return latest_data[fitur].iloc[-1:].values


def get_next_trading_dates(df, start_date, n_days=HORIZON_DAYS):
    """
    Mendapatkan tanggal trading berikutnya berdasarkan pola historis data
    """
    all_dates = df.index.tolist()
    date_diffs = [(all_dates[i+1] - all_dates[i]).days for i in range(len(all_dates)-1)]
    avg_diff = np.median(date_diffs)
    
    future_dates = []
    current_date = start_date
    
    for i in range(n_days):
        current_date = current_date + timedelta(days=1)
        while current_date.weekday() >= 5:
            current_date = current_date + timedelta(days=1)
        future_dates.append(current_date)
    
    return future_dates


def get_actual_prices_for_dates(df_original, future_dates):
    """
    Mendapatkan harga aktual jika tersedia di data historis
    """
    actual_prices = []
    for date in future_dates:
        try:
            if date in df_original.index:
                actual_prices.append(df_original.loc[date, 'Close'])
            else:
                actual_prices.append(None)
        except:
            actual_prices.append(None)
    
    return actual_prices


def predict_next_days_direct(models_multistep, df, fitur, days=HORIZON_DAYS):
    """
    Prediksi harga menggunakan Direct Multi-Step Forecasting Strategy.

    Catatan:
    - Model yang dilatih pada versi ini memprediksi RETURN (bukan harga absolut),
      yaitu (Close_{t+day} / Close_t) - 1.
    - Karena itu, hasil prediksi dikonversi kembali menjadi harga:
      pred_price = last_close * (1 + pred_return)
    """
    predictions = []
    features = create_features_for_prediction(df, fitur)

    last_close = float(df['Close'].iloc[-1])

    for day in range(1, days + 1):
        pred_return = float(models_multistep[day].predict(features)[0])
        pred_price = last_close * (1.0 + pred_return)
        predictions.append(pred_price)

    return predictions


def predict_from_manual_input(models_multistep, feature_array, base_close):
    """
    Prediksi harga menggunakan input manual dari user.

    Catatan:
    - Model memprediksi RETURN terhadap harga penutupan (Close) yang diberikan user.
    - Konversi: pred_price = base_close * (1 + pred_return)
    """
    predictions = []
    base_close = float(base_close)

    for day in range(1, HORIZON_DAYS + 1):
        pred_return = float(models_multistep[day].predict(feature_array)[0])
        pred_price = base_close * (1.0 + pred_return)
        predictions.append(pred_price)

    return predictions


# =====================================================
# LOAD DATA DAN MODEL TERLATIH
# =====================================================
with st.spinner('Memuat dan memproses data...'):
    df = load_and_process_data()

# Simpan dataframe asli sebelum dropna untuk referensi tanggal
df_original = pd.read_csv('data/data_historis_BBCA.csv', thousands='.', decimal=',')
df_original = df_original.rename(columns={
    'Tanggal': 'Date',
    'Pembukaan': 'Open',
    'Tertinggi': 'High',
    'Terendah': 'Low',
    'Terakhir': 'Close',
    'Vol.': 'Volume',
    'Perubahan%': 'ChangePercent'
})
df_original['Date'] = pd.to_datetime(df_original['Date'])
df_original = df_original.sort_values('Date')
df_original = df_original.set_index('Date')

with st.spinner('Memuat model yang sudah dilatih...'):
    results, best_params, eval_results = load_trained_models()

# Extract hasil dari model
model_harian = results['model_harian']
model_mingguan = results['model_mingguan']
models_multistep = results['models_multistep']
X_train = results['X_train']
X_test = results['X_test']
y_harian_test = results['y_harian_test']
y_mingguan_test = results['y_mingguan_test']
y_pred_harian = results['y_pred_harian']
y_pred_mingguan = results['y_pred_mingguan']
hasil_harian = results['hasil_harian']
hasil_mingguan = results['hasil_mingguan']
hasil_multistep = results['hasil_multistep']
split_idx = results['split_idx']
fitur = results['fitur']


# =====================================================
# HEADER
# =====================================================
st.markdown('<div class="main-header">SISTEM PREDIKSI HARGA SAHAM BBCA</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h4 style='color: #00d4ff;'>Menggunakan Algoritma Extreme Gradient Boosting (XGBoost)</h4>
        <p style='color: #e8e8e8;'><b>Studi Kasus:</b> PT Bank Central Asia Tbk</p>
        <p style='color: #a8a8a8; font-size: 0.9rem;'>
            Alexius Kenriko Salim (22101152630094)<br>
            Program Studi Teknik Informatika
        </p>
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.image("logo_BCA2.png", width=230)
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Menu Navigasi",
    ["Dashboard", "Analisis Data", "Prediksi Manual", "Prediksi 1 Hari", "Prediksi 1 Minggu", "Parameter Model", "Feature Importance", "Info Penelitian"]
)

st.sidebar.markdown("---")
# Tooltip untuk Informasi Data di sidebar
_sidebar_tip = (
    "Periode Data dan Total Data dihitung dari df (data yang sudah melalui feature engineering dan dropna). "
    "Data Training dan Data Testing dihitung berdasarkan split_idx. Total Fitur adalah jumlah kolom fitur pada variabel 'fitur'."
)
tooltip_safe_data = html.escape(_sidebar_tip, quote=True)

st.sidebar.markdown(textwrap.dedent(f"""
<div style='background-color: #0f3460; padding: 1rem; border-radius: 0.5rem; border: 1px solid #00d4ff;'>
    <h4 style='margin: 0; color: #00d4ff;'>Informasi Data</h4>
    <p style='margin: 0.5rem 0; color: #e8e8e8;'><b>Periode Data:</b></p>
    <p style='margin: 0; font-size: 0.9rem; color: #e8e8e8;'>
        {df.index[0].strftime('%d %B %Y')} s/d {df.index[-1].strftime('%d %B %Y')}
    </p>
    <p style='margin: 0.5rem 0; color: #e8e8e8;'><b>Total Data:</b> {len(df)} hari</p>
    <p style='margin: 0; color: #e8e8e8;'><b>Data Training:</b> {split_idx} hari</p>
    <p style='margin: 0; color: #e8e8e8;'><b>Data Testing:</b> {len(df) - split_idx} hari</p>
    <p style='margin: 0.5rem 0; color: #4ade80;'><b>Total Fitur:</b> {len(fitur)}</p>
</div>
""").strip(), unsafe_allow_html=True)




# =====================================================
# HALAMAN: DASHBOARD (DENGAN PREDIKSI OTOMATIS)
# =====================================================
if menu == "Dashboard":
    st.markdown('<div class="sub-header">Ringkasan Kinerja Model</div>', unsafe_allow_html=True)
    
    # Tampilkan performa overall
    # Tampilkan performa overall
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Prediksi 1 Hari ke Depan")
        metric_cols = st.columns(2)
        with metric_cols[0]:
            metric_with_tooltip("RMSE", f"Rp {hasil_harian['RMSE']:,.0f}", "Nilai RMSE ditampilkan dari hasil evaluasi model yang sudah disimpan saat proses pelatihan (file hasil pelatihan yang kemudian dimuat kembali di aplikasi). RMSE menunjukkan rata-rata besar selisih antara harga prediksi dan harga aktual pada data uji, dalam satuan Rupiah (semakin kecil berarti prediksi makin mendekati harga aktual).")
            metric_with_tooltip("MAE", f"Rp {hasil_harian['MAE']:,.0f}", "Nilai MAE ditampilkan dari hasil evaluasi model yang disimpan dari proses pelatihan dan dimuat pada aplikasi. MAE menunjukkan rata-rata selisih absolut antara harga prediksi dan harga aktual pada data uji, dalam Rupiah. Metrik ini mudah dipahami karena langsung menggambarkan ‘rata-rata melesetnya’ prediksi (semakin kecil semakin baik).")
        with metric_cols[1]:
            metric_with_tooltip("MAPE", f"{hasil_harian['MAPE']:.2f}%", "Nilai MAPE ditampilkan dari hasil evaluasi model yang disimpan saat pelatihan dan dimuat kembali pada aplikasi. MAPE menunjukkan rata-rata kesalahan prediksi dalam bentuk persentase terhadap harga aktual pada data uji. Metrik ini membantu melihat kesalahan secara relatif (misalnya 2% berarti rata-rata meleset sekitar 2% dari harga aktual; semakin kecil semakin baik).")
            metric_with_tooltip("R²", f"{hasil_harian['R2']:.4f}", "Nilai R² ditampilkan dari hasil evaluasi model yang disimpan saat pelatihan dan dimuat pada aplikasi. R² menunjukkan seberapa baik prediksi model mengikuti pola perubahan harga aktual pada data uji. Nilai yang mendekati 1 berarti model semakin mampu menjelaskan variasi pergerakan harga, sedangkan nilai mendekati 0 berarti kemampuan penjelasannya rendah.")
    
    with col2:
        st.markdown("#### Prediksi 1 Minggu ke Depan")
        metric_cols = st.columns(2)
        with metric_cols[0]:
            metric_with_tooltip("RMSE", f"Rp {hasil_mingguan['RMSE']:,.0f}", "Nilai RMSE ditampilkan dari hasil evaluasi model yang sudah disimpan saat proses pelatihan (file hasil pelatihan yang kemudian dimuat kembali di aplikasi). RMSE menunjukkan rata-rata besar selisih antara harga prediksi dan harga aktual pada data uji, dalam satuan Rupiah (semakin kecil berarti prediksi makin mendekati harga aktual).")
            metric_with_tooltip("MAE", f"Rp {hasil_mingguan['MAE']:,.0f}", "Nilai MAE ditampilkan dari hasil evaluasi model yang disimpan dari proses pelatihan dan dimuat pada aplikasi. MAE menunjukkan rata-rata selisih absolut antara harga prediksi dan harga aktual pada data uji, dalam Rupiah. Metrik ini mudah dipahami karena langsung menggambarkan ‘rata-rata melesetnya’ prediksi (semakin kecil semakin baik).")
        with metric_cols[1]:
            metric_with_tooltip("MAPE", f"{hasil_mingguan['MAPE']:.2f}%", "Nilai MAPE ditampilkan dari hasil evaluasi model yang disimpan saat pelatihan dan dimuat kembali pada aplikasi. MAPE menunjukkan rata-rata kesalahan prediksi dalam bentuk persentase terhadap harga aktual pada data uji. Metrik ini membantu melihat kesalahan secara relatif (misalnya 2% berarti rata-rata meleset sekitar 2% dari harga aktual; semakin kecil semakin baik).")
            metric_with_tooltip("R²", f"{hasil_mingguan['R2']:.4f}", "Nilai R² ditampilkan dari hasil evaluasi model yang disimpan saat pelatihan dan dimuat pada aplikasi. R² menunjukkan seberapa baik prediksi model mengikuti pola perubahan harga aktual pada data uji. Nilai yang mendekati 1 berarti model semakin mampu menjelaskan variasi pergerakan harga, sedangkan nilai mendekati 0 berarti kemampuan penjelasannya rendah.")
    
    # Tampilkan performa pada high volatility jika tersedia
    if eval_results and eval_results.get('high_volatility_performance'):
        st.markdown("---")
        st.markdown("#### Performa Model pada Kondisi Berbeda")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Periode High Volatility**")
            high_vol = eval_results['high_volatility_performance']
            metric_cols = st.columns(2)
            with metric_cols[0]:
                st.metric("RMSE", f"Rp {high_vol['RMSE']:,.0f}")
                st.metric("MSE", f"{high_vol['MSE']:,.0f}")
                st.metric("MAE", f"Rp {high_vol['MAE']:,.0f}")
            with metric_cols[1]:
                st.metric("MAPE", f"{high_vol['MAPE']:.2f}%")
                st.metric("R²", f"{high_vol['R2']:.4f}")
        
        with col2:
            st.markdown("**Periode Normal Volatility**")
            normal_vol = eval_results['normal_volatility_performance']
            metric_cols = st.columns(2)
            with metric_cols[0]:
                st.metric("RMSE", f"Rp {normal_vol['RMSE']:,.0f}")
                st.metric("MSE", f"{normal_vol['MSE']:,.0f}")
                st.metric("MAE", f"Rp {normal_vol['MAE']:,.0f}")
            with metric_cols[1]:
                st.metric("MAPE", f"{normal_vol['MAPE']:.2f}%")
                st.metric("R²", f"{normal_vol['R2']:.4f}")
    
    st.markdown("---")
    
    st.markdown('<div class="sub-header">Gambaran Umum Harga Saham BBCA</div>', unsafe_allow_html=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Harga Penutupan',
        line=dict(color='#00d4ff', width=2)
    ))
    
    fig.add_vrect(
        x0=df.index[0], x1=df.index[split_idx-1],
        fillcolor="green", opacity=0.1,
        layer="below", line_width=0,
        annotation_text="Training", annotation_position="top left"
    )
    fig.add_vrect(
        x0=df.index[split_idx], x1=df.index[-1],
        fillcolor="orange", opacity=0.1,
        layer="below", line_width=0,
        annotation_text="Testing", annotation_position="top right"
    )
    
    fig.update_layout(
        title=dict(
            text="Pergerakan Harga Saham BBCA",
            font=dict(color="#ffffff")
        ),
        xaxis=dict(
            title=dict(text="Tanggal", font=dict(color="#ffffff")),
            tickfont=dict(color="#ffffff")
        ),
        yaxis=dict(
            title=dict(text="Harga (IDR)", font=dict(color="#ffffff")),
            tickfont=dict(color="#ffffff")
        ),
        hovermode='x unified',
        height=500,
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#0f3460',
        font=dict(color="#ffffff")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ===== PREDIKSI OTOMATIS 5 HARI KE DEPAN =====
    st.markdown('<div class="sub-header">Prediksi Harga Saham 7 Hari ke Depan (Out-of-Sample)</div>', unsafe_allow_html=True)
    
    # Info harga terakhir
    last_price = df['Close'].iloc[-1]
    last_date = df.index[-1]
    
    render_html(f"""
    <div class="info-box">
        <h4>Data Terakhir</h4>
        <p><b>Tanggal:</b> {last_date.strftime('%d %B %Y')}</p>
        <p><b>Harga Penutupan:</b> Rp {last_price:,.0f}</p>
    </div>
    """)
    
    # Prediksi 7 hari ke depan menggunakan Direct Multi-Step Strategy
    with st.spinner('Melakukan prediksi menggunakan Direct Multi-Step Forecasting...'):
        predictions = predict_next_days_direct(models_multistep, df, fitur, days=HORIZON_DAYS)
        
        # Dapatkan tanggal trading berikutnya berdasarkan pola historis
        future_dates = get_next_trading_dates(df_original, last_date, n_days=HORIZON_DAYS)
        
        # Dapatkan harga aktual jika tersedia
        actual_prices = get_actual_prices_for_dates(df_original, future_dates)
    
    # Tampilkan prediksi dalam cards
    pred_heading_tip = (
        "Prediksi harga penutupan untuk Hari +1 sampai Hari +7 dihitung menggunakan strategi Direct Multi-Step "
        "(setiap horizon memakai model tersendiri pada models_multistep). "
        "Tanggal Hari +n adalah hari bursa berikutnya (berdasarkan pola hari perdagangan historis). "
        "Persentase menunjukkan perubahan dibanding harga penutupan terakhir."
    )
    st.markdown(
        f"<div style='display:flex; align-items:center; gap:8px; margin-bottom:0.25rem;'>"
        f"<h4 style='margin:0; color:#e8e8e8;'>Hasil Prediksi</h4>"
        f"{tooltip_span(pred_heading_tip)}"
        f"</div>",
        unsafe_allow_html=True
    )

    
    cols = st.columns(HORIZON_DAYS)
    colors = ['#00d4ff', '#00a8cc', '#007c99', '#005566', '#003344', '#2b2d42', '#1f2233']
    
    for i, (col, pred, color, future_date) in enumerate(zip(cols, predictions, colors, future_dates)):
        with col:
            change = ((pred - last_price) / last_price) * 100
            arrow = "↑" if change > 0 else "↓"
            tooltip_text = (
                f"Prediksi harga penutupan untuk Hari +{i+1} dihitung oleh model horizon {i+1} hari (Direct Multi-Step). "
                f"Tanggal merupakan hari bursa berikutnya (berdasarkan pola historis). "
                f"Perubahan {abs(change):.2f}% dihitung dibanding harga penutupan terakhir (Rp {last_price:,.0f})."
            )
            tip_html = tooltip_span(tooltip_text, width_px=360)

            render_html(f"""
            <div style='background-color: #0f3460; color: white; padding: 1rem; border-radius: 0.5rem; text-align: center;
                        border: 2px solid {color}; position: relative;'>
                <div style='position:absolute; top:10px; right:10px;'>{tip_html}</div>
                <h5 style='margin: 0; color: {color};'>Hari +{i+1}</h5>
                <p style='margin: 0.25rem 0; font-size: 0.8rem; color: #a8a8a8;'>{future_date.strftime('%d %b %Y')}</p>
                <h3 style='margin: 0.5rem 0; color: #00d4ff;'>Rp {pred:,.0f}</h3>
                <p style='margin: 0; font-size: 0.9rem; color: {"#4ade80" if change > 0 else "#f87171"};'>
                    {arrow} {abs(change):.2f}%
                </p>
            </div>
            """)


    
    st.markdown("---")
    
    # Grafik Prediksi
    st.markdown("#### Visualisasi Prediksi 7 Hari ke Depan")
    
    # Data historis (30 hari terakhir)
    historical_dates = df.index[-30:].tolist()
    historical_prices = df['Close'].iloc[-30:].tolist()
    
    # Gabungkan dengan prediksi
    all_dates = historical_dates + future_dates
    all_prices = historical_prices + [None] * len(predictions)
    predicted_prices = [None] * len(historical_prices) + predictions
    
    # Tambahkan harga aktual jika tersedia
    actual_extended = [None] * len(historical_prices) + actual_prices
    
    fig_forecast = go.Figure()
    
    # Historical data
    fig_forecast.add_trace(go.Scatter(
        x=historical_dates,
        y=historical_prices,
        mode='lines+markers',
        name='Data Historis',
        line=dict(color='#00d4ff', width=2),
        marker=dict(size=6)
    ))
    
    # Predicted data
    fig_forecast.add_trace(go.Scatter(
        x=[historical_dates[-1]] + future_dates,
        y=[historical_prices[-1]] + predictions,
        mode='lines+markers',
        name='Prediksi',
        line=dict(color='#ff6b6b', width=2, dash='dash'),
        marker=dict(size=8, symbol='diamond')
    ))
    
    # Actual prices (if available)
    if any(actual_prices):
        valid_actuals = [(future_dates[i], actual_prices[i]) for i in range(len(actual_prices)) if actual_prices[i] is not None]
        if valid_actuals:
            actual_dates, actual_vals = zip(*valid_actuals)
            fig_forecast.add_trace(go.Scatter(
                x=actual_dates,
                y=actual_vals,
                mode='markers',
                name='Harga Aktual',
                marker=dict(size=10, color='#4ade80', symbol='star')
            ))
    
    fig_forecast.update_layout(
        title="Prediksi Harga Saham BBCA - 7 Hari ke Depan",
        xaxis_title="Tanggal",
        yaxis_title="Harga (IDR)",
        hovermode='x unified',
        height=500,
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#0f3460',
        font=dict(color="#ffffff"),
        showlegend=True,
        legend=dict(bgcolor='#0f3460', bordercolor='#00d4ff', borderwidth=1)
    )
    
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Tabel prediksi detail
    st.markdown("#### Detail Prediksi")
    
    prediction_df = pd.DataFrame({
        'Hari': [f'Hari +{i+1}' for i in range(HORIZON_DAYS)],
        'Tanggal': [d.strftime('%d %B %Y') for d in future_dates],
        'Harga Prediksi (IDR)': [f'Rp {p:,.0f}' for p in predictions],
        'Perubahan dari Hari Ini': [f'{((p - last_price) / last_price) * 100:+.2f}%' for p in predictions],
        'Harga Aktual (IDR)': [f'Rp {a:,.0f}' if a else 'Belum Tersedia' for a in actual_prices]
    })
    
    st.dataframe(prediction_df, use_container_width=True, hide_index=True)
    
    # Metrik evaluasi jika ada harga aktual
    if any(actual_prices):
        st.markdown("#### Evaluasi Prediksi (Data Tersedia)")
        valid_pairs = [(predictions[i], actual_prices[i]) for i in range(len(actual_prices)) if actual_prices[i] is not None]
        
        if valid_pairs:
            preds, actuals = zip(*valid_pairs)
            mse = mean_squared_error(actuals, preds)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actuals, preds)
            mape = np.mean(np.abs((np.array(actuals) - np.array(preds)) / np.array(actuals))) * 100
            
            eval_cols = st.columns(4)
            with eval_cols[0]:
                st.metric("RMSE", f"Rp {rmse:,.0f}")
            with eval_cols[1]:
                st.metric("MSE", f"{mse:,.0f}")
            with eval_cols[2]:
                st.metric("MAE", f"Rp {mae:,.0f}")
            with eval_cols[3]:
                st.metric("MAPE", f"{mape:.2f}%")


# =====================================================
# HALAMAN: ANALISIS DATA
# =====================================================
elif menu == "Analisis Data":
    st.markdown('<div class="sub-header">Analisis Data Saham BBCA</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Statistik Deskriptif", "Distribusi Data", "Korelasi Fitur", "Data Mentah"])
    
    with tab1:
        st.markdown("### Statistik Deskriptif")
        
        stats_df = df[['Open', 'High', 'Low', 'Close', 'Volume']].describe()
        st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Harga Tertinggi & Terendah")
            highest = df['Close'].max()
            lowest = df['Close'].min()
            highest_date = df['Close'].idxmax()
            lowest_date = df['Close'].idxmin()
            
            st.metric("Harga Tertinggi", f"Rp {highest:,.0f}", f"{highest_date.strftime('%d %B %Y')}")
            st.metric("Harga Terendah", f"Rp {lowest:,.0f}", f"{lowest_date.strftime('%d %B %Y')}")
        
        with col2:
            st.markdown("#### Volatilitas")
            volatility = df['Close'].std()
            avg_return = df['return_1'].mean() * 100
            
            st.metric("Standar Deviasi", f"Rp {volatility:,.0f}")
            st.metric("Rata-rata Return Harian", f"{avg_return:.3f}%")
    
    with tab2:
        st.markdown("### Distribusi Harga Penutupan")
        
        fig_hist = px.histogram(
            df, 
            x='Close',
            nbins=50,
            title='Distribusi Harga Penutupan',
            labels={'Close': 'Harga Penutupan (IDR)'}
        )
        
        fig_hist.update_layout(
            paper_bgcolor='#1a1a2e',
            plot_bgcolor='#0f3460',
            font=dict(color="#ffffff"),
            height=500
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### Distribusi Return Harian")
        
        fig_return = px.histogram(
            df.dropna(), 
            x='return_1',
            nbins=50,
            title='Distribusi Return Harian',
            labels={'return_1': 'Return'}
        )
        
        fig_return.update_layout(
            paper_bgcolor='#1a1a2e',
            plot_bgcolor='#0f3460',
            font=dict(color="#ffffff"),
            height=500
        )
        
        st.plotly_chart(fig_return, use_container_width=True)
    
    with tab3:
        st.markdown("### Korelasi Antar Fitur")
        
        # Pilih fitur numerik untuk korelasi
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'return_1', 'hl_range', 'oc_change',
                       'momentum_5', 'volatility_ratio', 'volume_ma_ratio']
        corr_matrix = df[numeric_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect='auto',
            color_continuous_scale='RdBu_r',
            title='Matriks Korelasi Fitur'
        )
        
        fig_corr.update_layout(
            paper_bgcolor='#1a1a2e',
            font=dict(color="#ffffff"),
            height=600
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab4:
        st.markdown("### Data Mentah")
        
        st.markdown(f"Menampilkan {len(df)} baris data")
        
        # Filter data
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Tanggal Mulai", df.index[0])
        with col2:
            end_date = st.date_input("Tanggal Akhir", df.index[-1])
        
        # Filter berdasarkan tanggal
        mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
        filtered_df = df.loc[mask]
        
        st.dataframe(
            filtered_df[['Open', 'High', 'Low', 'Close', 'Volume']].style.format({
                'Open': 'Rp {:,.0f}',
                'High': 'Rp {:,.0f}',
                'Low': 'Rp {:,.0f}',
                'Close': 'Rp {:,.0f}',
                'Volume': '{:,.0f}'
            }),
            use_container_width=True
        )
        st.markdown("""
        <style>
        div.stDownloadButton > button {
            background-color: #120b8f;
            color: #ffffff;
            border-radius: 8px;
            padding: 0.55em 1.3em;
            font-weight: 600;
            border: none;
        }

        div.stDownloadButton > button:hover {
            background-color: #110b78;
            color: #ffffff;
        }
        </style>
        """, unsafe_allow_html=True)

        # Download button
        csv = filtered_df.to_csv()
        st.download_button(
            label="Download Data sebagai CSV",
            data=csv,
            file_name=f'BBCA_data_{start_date}_{end_date}.csv',
            mime='text/csv'
        )


# =====================================================
# HALAMAN: PREDIKSI MANUAL
# =====================================================
elif menu == "Prediksi Manual":
    st.markdown('<div class="sub-header">Prediksi dengan Input Manual</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>Cara Penggunaan</h4>
        <p>Masukkan data OHLC (Open, High, Low, Close) dan Volume untuk mendapatkan prediksi harga 7 hari ke depan.</p>
        <p><b>Catatan:</b> Fitur turunan (termasuk enhanced features) akan dihitung otomatis oleh sistem berdasarkan input Anda.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Ambil data terakhir sebagai default
    last_row = df.iloc[-1]
    
    with st.form("manual_prediction_form"):
        st.markdown("### Input Data OHLCV")
        
        col1, col2 = st.columns(2)
        
        with col1:
            open_price = st.number_input(
                "Harga Pembukaan (Open)",
                min_value=0.0,
                value=float(last_row['Open']),
                step=50.0,
                format="%.2f"
            )
            
            high_price = st.number_input(
                "Harga Tertinggi (High)",
                min_value=0.0,
                value=float(last_row['High']),
                step=50.0,
                format="%.2f"
            )
            
            low_price = st.number_input(
                "Harga Terendah (Low)",
                min_value=0.0,
                value=float(last_row['Low']),
                step=50.0,
                format="%.2f"
            )
        
        with col2:
            close_price = st.number_input(
                "Harga Penutupan (Close)",
                min_value=0.0,
                value=float(last_row['Close']),
                step=50.0,
                format="%.2f"
            )
            
            volume = st.number_input(
                "Volume",
                min_value=0.0,
                value=float(last_row['Volume']),
                step=1000.0,
                format="%.0f"
            )
        
        submitted = st.form_submit_button("🔮 Prediksi Harga", use_container_width=True)
    
    if submitted:
        # Validasi input
        if high_price < max(open_price, close_price) or low_price > min(open_price, close_price):
            st.error("⚠️ Data tidak valid! Pastikan High >= max(Open, Close) dan Low <= min(Open, Close)")
        else:
            with st.spinner('Melakukan prediksi dengan enhanced model...'):
                # Buat fitur dari input manual
                features = create_features_from_manual_input(
                    open_price, high_price, low_price, close_price, volume,
                    df, fitur
                )
                
                # Prediksi
                predictions = predict_from_manual_input(models_multistep, features, base_close=close_price)
                
                # Tanggal prediksi
                base_date = datetime.now()
                future_dates = get_next_trading_dates(df_original, pd.Timestamp(base_date), n_days=HORIZON_DAYS)
            
            st.success("✅ Prediksi berhasil!")
            
            st.markdown("---")
            st.markdown("### Hasil Prediksi 7 Hari ke Depan")
            
            # Tampilkan dalam cards
            cols = st.columns(HORIZON_DAYS)
            colors = ['#00d4ff', '#00a8cc', '#007c99', '#005566', '#003344', '#2b2d42', '#1f2233']
            
            for i, (col, pred, color, future_date) in enumerate(zip(cols, predictions, colors, future_dates)):
                with col:
                    change = ((pred - close_price) / close_price) * 100
                    arrow = "↑" if change > 0 else "↓"
                    
                    render_html(f"""
                    <div style='background-color: #0f3460; 
                                color: white; padding: 1rem; border-radius: 0.5rem; text-align: center;
                                border: 2px solid {color};'>
                        <h5 style='margin: 0; color: {color};'>Hari +{i+1}</h5>
                        <p style='margin: 0.25rem 0; font-size: 0.8rem; color: #a8a8a8;'>{future_date.strftime('%d %b %Y')}</p>
                        <h3 style='margin: 0.5rem 0; color: #00d4ff;'>Rp {pred:,.0f}</h3>
                        <p style='margin: 0; font-size: 0.9rem; color: {"#4ade80" if change > 0 else "#f87171"};'>
                            {arrow} {abs(change):.2f}%
                        </p>
                    </div>
                    """)
            
            st.markdown("---")
            
            # Grafik
            st.markdown("### Visualisasi Prediksi")
            
            fig_manual = go.Figure()
            
            # Input price as starting point
            fig_manual.add_trace(go.Scatter(
                x=[base_date],
                y=[close_price],
                mode='markers',
                name='Input Harga',
                marker=dict(size=12, color='#00d4ff', symbol='circle')
            ))
            
            # Predictions
            fig_manual.add_trace(go.Scatter(
                x=[base_date] + future_dates,
                y=[close_price] + predictions,
                mode='lines+markers',
                name='Prediksi',
                line=dict(color='#ff6b6b', width=2, dash='dash'),
                marker=dict(size=8, symbol='diamond')
            ))
            
            fig_manual.update_layout(
                title="Prediksi Berdasarkan Input Manual",
                xaxis_title="Tanggal",
                yaxis_title="Harga (IDR)",
                hovermode='x unified',
                height=500,
                paper_bgcolor='#1a1a2e',
                plot_bgcolor='#0f3460',
                font=dict(color="#ffffff"),
                showlegend=True,
                legend=dict(bgcolor='#0f3460', bordercolor='#00d4ff', borderwidth=1)
            )
            
            st.plotly_chart(fig_manual, use_container_width=True)
            
            # Tabel detail
            st.markdown("### Detail Prediksi")
            
            prediction_df = pd.DataFrame({
                'Hari': [f'Hari +{i+1}' for i in range(HORIZON_DAYS)],
                'Tanggal': [d.strftime('%d %B %Y') for d in future_dates],
                'Harga Prediksi (IDR)': [f'Rp {p:,.0f}' for p in predictions],
                'Perubahan': [f'{((p - close_price) / close_price) * 100:+.2f}%' for p in predictions]
            })
            
            st.dataframe(prediction_df, use_container_width=True, hide_index=True)


# =====================================================
# HALAMAN: PREDIKSI 1 HARI
# =====================================================
elif menu == "Prediksi 1 Hari":
    st.markdown('<div class="sub-header">Evaluasi Model Prediksi 1 Hari</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        metric_with_tooltip("RMSE", f"Rp {hasil_harian['RMSE']:,.0f}", "Nilai RMSE ditampilkan dari hasil evaluasi model yang sudah disimpan saat proses pelatihan (file hasil pelatihan yang kemudian dimuat kembali di aplikasi). RMSE menunjukkan rata-rata besar selisih antara harga prediksi dan harga aktual pada data uji, dalam satuan Rupiah (semakin kecil berarti prediksi makin mendekati harga aktual).")
    with col2:
        metric_with_tooltip("MSE", f"{hasil_harian['MSE']:,.0f}", "Nilai MSE ditampilkan dari hasil evaluasi model yang sudah disimpan saat proses pelatihan dan dimuat oleh aplikasi. MSE menghitung rata-rata kuadrat selisih antara harga prediksi dan harga aktual pada data uji. Karena menggunakan kuadrat, kesalahan yang besar akan dihukum lebih berat (semakin kecil berarti performa lebih baik).")
    with col3:
        metric_with_tooltip("MAE", f"Rp {hasil_harian['MAE']:,.0f}", "Nilai MAE ditampilkan dari hasil evaluasi model yang disimpan dari proses pelatihan dan dimuat pada aplikasi. MAE menunjukkan rata-rata selisih absolut antara harga prediksi dan harga aktual pada data uji, dalam Rupiah. Metrik ini mudah dipahami karena langsung menggambarkan ‘rata-rata melesetnya’ prediksi (semakin kecil semakin baik).")
    with col4:
        metric_with_tooltip("MAPE", f"{hasil_harian['MAPE']:.2f}%", "Nilai MAPE ditampilkan dari hasil evaluasi model yang disimpan saat pelatihan dan dimuat kembali pada aplikasi. MAPE menunjukkan rata-rata kesalahan prediksi dalam bentuk persentase terhadap harga aktual pada data uji. Metrik ini membantu melihat kesalahan secara relatif (misalnya 2% berarti rata-rata meleset sekitar 2% dari harga aktual; semakin kecil semakin baik).")
    with col5:
        metric_with_tooltip("R²", f"{hasil_harian['R2']:.4f}", "Nilai R² ditampilkan dari hasil evaluasi model yang disimpan saat pelatihan dan dimuat pada aplikasi. R² menunjukkan seberapa baik prediksi model mengikuti pola perubahan harga aktual pada data uji. Nilai yang mendekati 1 berarti model semakin mampu menjelaskan variasi pergerakan harga, sedangkan nilai mendekati 0 berarti kemampuan penjelasannya rendah.")
    
    st.markdown("---")
    
    # Grafik Actual vs Predicted
    st.markdown("### Perbandingan Harga Aktual vs Prediksi (Data Testing)")
    
    # Ambil tanggal untuk data testing
    test_dates = df.index[split_idx:]
    
    fig_comp = go.Figure()
    
    fig_comp.add_trace(go.Scatter(
        x=test_dates,
        y=y_harian_test,
        mode='lines',
        name='Aktual',
        line=dict(color='#00d4ff', width=2)
    ))
    
    fig_comp.add_trace(go.Scatter(
        x=test_dates,
        y=y_pred_harian,
        mode='lines',
        name='Prediksi',
        line=dict(color='#ff6b6b', width=2, dash='dash')
    ))
    
    fig_comp.update_layout(
        xaxis_title="Tanggal",
        yaxis_title="Harga (IDR)",
        hovermode='x unified',
        height=500,
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#0f3460',
        font=dict(color="#ffffff"),
        showlegend=True,
        legend=dict(bgcolor='#0f3460', bordercolor='#00d4ff', borderwidth=1)
    )
    
    st.plotly_chart(fig_comp, use_container_width=True)
    
    st.markdown("---")
    
    # Scatter plot
    st.markdown("### Scatter Plot: Aktual vs Prediksi")
    
    fig_scatter = go.Figure()
    
    fig_scatter.add_trace(go.Scatter(
        x=y_harian_test,
        y=y_pred_harian,
        mode='markers',
        marker=dict(color='#00d4ff', size=8),
        name='Data Testing'
    ))
    
    # Perfect prediction line
    min_val = min(y_harian_test.min(), y_pred_harian.min())
    max_val = max(y_harian_test.max(), y_pred_harian.max())
    
    fig_scatter.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='#ff6b6b', dash='dash'),
        name='Perfect Prediction'
    ))
    
    fig_scatter.update_layout(
        xaxis_title="Harga Aktual (IDR)",
        yaxis_title="Harga Prediksi (IDR)",
        height=500,
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#0f3460',
        font=dict(color="#ffffff"),
        showlegend=True,
        legend=dict(bgcolor='#0f3460', bordercolor='#00d4ff', borderwidth=1)
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.markdown("---")
    
    # Residual plot
    st.markdown("### Analisis Error (Residual)")
    
    residuals = y_harian_test - y_pred_harian
    
    fig_residual = go.Figure()
    
    fig_residual.add_trace(go.Scatter(
        x=test_dates,
        y=residuals,
        mode='markers',
        marker=dict(color='#00d4ff', size=6),
        name='Residual'
    ))
    
    fig_residual.add_hline(
        y=0, 
        line_dash="dash", 
        line_color="#ff6b6b",
        annotation_text="Zero Error"
    )
    
    fig_residual.update_layout(
        title="Residual Plot - Prediksi 1 Hari",
        xaxis_title="Tanggal",
        yaxis_title="Error (IDR)",
        hovermode='x unified',
        height=400,
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#0f3460',
        font=dict(color="#ffffff")
    )
    
    st.plotly_chart(fig_residual, use_container_width=True)
    
    # Statistik residual
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Error", f"Rp {residuals.mean():,.0f}")
    with col2:
        st.metric("Std Error", f"Rp {residuals.std():,.0f}")
    with col3:
        st.metric("Max Absolute Error", f"Rp {abs(residuals).max():,.0f}")


# =====================================================
# HALAMAN: PREDIKSI 1 MINGGU
# =====================================================
elif menu == "Prediksi 1 Minggu":
    st.markdown('<div class="sub-header">Evaluasi Model Prediksi 1 Minggu</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        metric_with_tooltip("RMSE", f"Rp {hasil_mingguan['RMSE']:,.0f}", "Nilai RMSE ditampilkan dari hasil evaluasi model yang sudah disimpan saat proses pelatihan (file hasil pelatihan yang kemudian dimuat kembali di aplikasi). RMSE menunjukkan rata-rata besar selisih antara harga prediksi dan harga aktual pada data uji, dalam satuan Rupiah (semakin kecil berarti prediksi makin mendekati harga aktual).")
    with col2:
        metric_with_tooltip("MSE", f"{hasil_mingguan['MSE']:,.0f}", "Nilai MSE ditampilkan dari hasil evaluasi model yang sudah disimpan saat proses pelatihan dan dimuat oleh aplikasi. MSE menghitung rata-rata kuadrat selisih antara harga prediksi dan harga aktual pada data uji. Karena menggunakan kuadrat, kesalahan yang besar akan dihukum lebih berat (semakin kecil berarti performa lebih baik).")
    with col3:
        metric_with_tooltip("MAE", f"Rp {hasil_mingguan['MAE']:,.0f}", "Nilai MAE ditampilkan dari hasil evaluasi model yang disimpan dari proses pelatihan dan dimuat pada aplikasi. MAE menunjukkan rata-rata selisih absolut antara harga prediksi dan harga aktual pada data uji, dalam Rupiah. Metrik ini mudah dipahami karena langsung menggambarkan ‘rata-rata melesetnya’ prediksi (semakin kecil semakin baik).")
    with col4:
        metric_with_tooltip("MAPE", f"{hasil_mingguan['MAPE']:.2f}%", "Nilai MAPE ditampilkan dari hasil evaluasi model yang disimpan saat pelatihan dan dimuat kembali pada aplikasi. MAPE menunjukkan rata-rata kesalahan prediksi dalam bentuk persentase terhadap harga aktual pada data uji. Metrik ini membantu melihat kesalahan secara relatif (misalnya 2% berarti rata-rata meleset sekitar 2% dari harga aktual; semakin kecil semakin baik).")
    with col5:
        metric_with_tooltip("R²", f"{hasil_mingguan['R2']:.4f}", "Nilai R² ditampilkan dari hasil evaluasi model yang disimpan saat pelatihan dan dimuat pada aplikasi. R² menunjukkan seberapa baik prediksi model mengikuti pola perubahan harga aktual pada data uji. Nilai yang mendekati 1 berarti model semakin mampu menjelaskan variasi pergerakan harga, sedangkan nilai mendekati 0 berarti kemampuan penjelasannya rendah.")
    
    st.markdown("---")
    
    # Grafik Actual vs Predicted
    st.markdown("### Perbandingan Harga Aktual vs Prediksi (Data Testing)")
    
    test_dates = df.index[split_idx:]
    
    fig_comp_weekly = go.Figure()
    
    fig_comp_weekly.add_trace(go.Scatter(
        x=test_dates,
        y=y_mingguan_test,
        mode='lines',
        name='Aktual',
        line=dict(color='#00d4ff', width=2)
    ))
    
    fig_comp_weekly.add_trace(go.Scatter(
        x=test_dates,
        y=y_pred_mingguan,
        mode='lines',
        name='Prediksi',
        line=dict(color='#ff6b6b', width=2, dash='dash')
    ))
    
    fig_comp_weekly.update_layout(
        xaxis_title="Tanggal",
        yaxis_title="Harga (IDR)",
        hovermode='x unified',
        height=500,
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#0f3460',
        font=dict(color="#ffffff"),
        showlegend=True,
        legend=dict(bgcolor='#0f3460', bordercolor='#00d4ff', borderwidth=1)
    )
    
    st.plotly_chart(fig_comp_weekly, use_container_width=True)
    
    st.markdown("---")
    
    # Scatter plot
    st.markdown("### Scatter Plot: Aktual vs Prediksi")
    
    fig_scatter_weekly = go.Figure()
    
    fig_scatter_weekly.add_trace(go.Scatter(
        x=y_mingguan_test,
        y=y_pred_mingguan,
        mode='markers',
        marker=dict(color='#00d4ff', size=8),
        name='Data Testing'
    ))
    
    # Perfect prediction line
    min_val = min(y_mingguan_test.min(), y_pred_mingguan.min())
    max_val = max(y_mingguan_test.max(), y_pred_mingguan.max())
    
    fig_scatter_weekly.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='#ff6b6b', dash='dash'),
        name='Perfect Prediction'
    ))
    
    fig_scatter_weekly.update_layout(
        xaxis_title="Harga Aktual (IDR)",
        yaxis_title="Harga Prediksi (IDR)",
        height=500,
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#0f3460',
        font=dict(color="#ffffff"),
        showlegend=True,
        legend=dict(bgcolor='#0f3460', bordercolor='#00d4ff', borderwidth=1)
    )
    
    st.plotly_chart(fig_scatter_weekly, use_container_width=True)
    
    st.markdown("---")
    
    # Residual plot
    st.markdown("### Analisis Error (Residual)")
    
    residuals_weekly = y_mingguan_test - y_pred_mingguan
    
    fig_residual_weekly = go.Figure()
    
    fig_residual_weekly.add_trace(go.Scatter(
        x=test_dates,
        y=residuals_weekly,
        mode='markers',
        marker=dict(color='#00d4ff', size=6),
        name='Residual'
    ))
    
    fig_residual_weekly.add_hline(
        y=0, 
        line_dash="dash", 
        line_color="#ff6b6b",
        annotation_text="Zero Error"
    )
    
    fig_residual_weekly.update_layout(
        title="Residual Plot - Prediksi 1 Minggu",
        xaxis_title="Tanggal",
        yaxis_title="Error (IDR)",
        hovermode='x unified',
        height=400,
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#0f3460',
        font=dict(color="#ffffff")
    )
    
    st.plotly_chart(fig_residual_weekly, use_container_width=True)
    
    # Statistik residual
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Error", f"Rp {residuals_weekly.mean():,.0f}")
    with col2:
        st.metric("Std Error", f"Rp {residuals_weekly.std():,.0f}")
    with col3:
        st.metric("Max Absolute Error", f"Rp {abs(residuals_weekly).max():,.0f}")
    
    st.markdown("---")
    
    # Perbandingan dengan prediksi 1 hari
    st.markdown("### Perbandingan: Prediksi 1 Hari vs 1 Minggu")
    
    comparison_df = pd.DataFrame({
        'Metrik': ['RMSE', 'MAE', 'MAPE', 'R²'],
        'Prediksi 1 Hari': [
            f"Rp {hasil_harian['RMSE']:,.0f}",
            f"Rp {hasil_harian['MAE']:,.0f}",
            f"{hasil_harian['MAPE']:.2f}%",
            f"{hasil_harian['R2']:.4f}"
        ],
        'Prediksi 1 Minggu': [
            f"Rp {hasil_mingguan['RMSE']:,.0f}",
            f"Rp {hasil_mingguan['MAE']:,.0f}",
            f"{hasil_mingguan['MAPE']:.2f}%",
            f"{hasil_mingguan['R2']:.4f}"
        ]
    })
    
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)


# =====================================================
# HALAMAN: PARAMETER MODEL
# =====================================================
elif menu == "Parameter Model":
    st.markdown('<div class="sub-header">Parameter Model Terbaik</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>Informasi Parameter</h4>
        <p>Parameter berikut diperoleh dari proses <b>RandomizedSearchCV</b> dengan Time Series Cross-Validation.</p>
        <p>Setiap horizon prediksi (Hari +1 sampai Hari +7) memiliki model terpisah dengan parameter optimal masing-masing.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if best_params:
        # Tabs untuk setiap model
        tabs = st.tabs([f"Model Hari +{i}" for i in range(1, HORIZON_DAYS + 1)])
        
        for i, tab in enumerate(tabs, start=1):
            with tab:
                st.markdown(f"### Parameter Optimal - Model Hari +{i}")
                
                params = best_params[f'model_day_{i}']
                
                # Kelompokkan parameter
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Boosting Parameters")
                    render_html(f"""
                    - **n_estimators**: {params.get('n_estimators', 'N/A')}
                    - **learning_rate**: {params.get('learning_rate', 'N/A')}
                    - **max_depth**: {params.get('max_depth', 'N/A')}
                    - **min_child_weight**: {params.get('min_child_weight', 'N/A')}
                    - **gamma**: {params.get('gamma', 'N/A')}
                    """)
                
                with col2:
                    st.markdown("#### Regularization & Sampling")
                    st.markdown(f"""
                    - **subsample**: {params.get('subsample', 'N/A')}
                    - **colsample_bytree**: {params.get('colsample_bytree', 'N/A')}
                    - **reg_alpha** (L1): {params.get('reg_alpha', 'N/A')}
                    - **reg_lambda** (L2): {params.get('reg_lambda', 'N/A')}
                    - **objective**: {params.get('objective', 'N/A')}
                    """)
                
                st.markdown("---")
                
                # Tampilkan performa model ini
                if hasil_multistep and i in hasil_multistep:
                    st.markdown(f"#### Performa Model Hari +{i}")
                    
                    metric_cols = st.columns(5)
                    with metric_cols[0]:
                        st.metric("RMSE", f"Rp {hasil_multistep[i]['RMSE']:,.0f}")
                    with metric_cols[1]:
                        st.metric("MSE", f"{hasil_multistep[i]['MSE']:,.0f}")
                    with metric_cols[2]:
                        st.metric("MAE", f"Rp {hasil_multistep[i]['MAE']:,.0f}")
                    with metric_cols[3]:
                        st.metric("MAPE", f"{hasil_multistep[i]['MAPE']:.2f}%")
                    with metric_cols[4]:
                        st.metric("R²", f"{hasil_multistep[i]['R2']:.4f}")
    else:
        st.warning("Parameter model tidak tersedia.")
    
    st.markdown("---")
    
    


# =====================================================
# HALAMAN: FEATURE IMPORTANCE
# =====================================================
elif menu == "Feature Importance":
    st.markdown('<div class="sub-header">Analisis Feature Importance</div>', unsafe_allow_html=True)
    
    st.markdown("""
        Feature importance menunjukkan seberapa besar kontribusi setiap fitur terhadap prediksi model.
        Fitur dengan importance tinggi memiliki pengaruh lebih besar dalam menentukan harga saham.
    """)
    
    st.markdown("---")
    
    # Load feature importance
    if os.path.exists('models/feature_importance.csv'):
        feature_importance = pd.read_csv('models/feature_importance.csv')
        
        # Tabs untuk berbagai visualisasi
        tab1, tab2, tab3 = st.tabs(["Top 20 Features", "Feature Groups", "Semua Features"])
        
        with tab1:
            st.markdown("### Top 20 Fitur Terpenting")
            
            top_20 = feature_importance.head(20)
            
            fig_importance = px.bar(
                top_20,
                x='importance',
                y='feature',
                orientation='h',
                title='Top 20 Features by Importance',
                labels={'importance': 'Importance Score', 'feature': 'Feature'},
                color='importance',
                color_continuous_scale='Blues'
            )
            
            fig_importance.update_layout(
                height=600,
                paper_bgcolor='#1a1a2e',
                plot_bgcolor='#0f3460',
                font=dict(color="#ffffff"),
                yaxis={'categoryorder': 'total ascending'}
            )
            
            st.plotly_chart(fig_importance, use_container_width=True)
            
            st.markdown("#### Tabel Top 20 Features")
            st.dataframe(
                top_20.style.format({'importance': '{:.4f}'}),
                use_container_width=True,
                hide_index=True
            )
        
        with tab2:
            st.markdown("### Analisis Feature Groups")
            
            # Kategorikan fitur
            def categorize_feature(feature_name):
                if 'lag' in feature_name.lower():
                    return 'Lag Features'
                elif 'rolling' in feature_name.lower() or 'ma' in feature_name.lower():
                    return 'Moving Average Features'
                elif 'momentum' in feature_name.lower():
                    return 'Momentum Features'
                elif 'volatility' in feature_name.lower() or 'std' in feature_name.lower():
                    return 'Volatility Features'
                elif 'volume' in feature_name.lower():
                    return 'Volume Features'
                elif 'range' in feature_name.lower():
                    return 'Range Features'
                elif 'gap' in feature_name.lower():
                    return 'Gap Features'
                elif 'price_to' in feature_name.lower() or 'position' in feature_name.lower():
                    return 'Price Position Features'
                elif 'consecutive' in feature_name.lower():
                    return 'Trend Features'
                else:
                    return 'Other Features'
            
            feature_importance['category'] = feature_importance['feature'].apply(categorize_feature)
            
            category_importance = feature_importance.groupby('category')['importance'].agg(['sum', 'mean', 'count']).reset_index()
            category_importance = category_importance.sort_values('sum', ascending=False)
            
            # Bar chart untuk kategori
            fig_category = px.bar(
                category_importance,
                x='category',
                y='sum',
                title='Total Importance by Feature Category',
                labels={'sum': 'Total Importance', 'category': 'Feature Category'},
                color='sum',
                color_continuous_scale='Viridis'
            )
            
            fig_category.update_layout(
                height=500,
                paper_bgcolor='#1a1a2e',
                plot_bgcolor='#0f3460',
                font=dict(color="#ffffff"),
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig_category, use_container_width=True)
            
            st.markdown("#### Ringkasan per Kategori")
            st.dataframe(
                category_importance.style.format({
                    'sum': '{:.4f}',
                    'mean': '{:.4f}',
                    'count': '{:.0f}'
                }),
                use_container_width=True,
                hide_index=True
            )
            
            # Pie chart
            fig_pie = px.pie(
                category_importance,
                values='sum',
                names='category',
                title='Distribution of Feature Importance by Category'
            )
            
            fig_pie.update_layout(
                paper_bgcolor='#1a1a2e',
                font=dict(color="#ffffff")
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with tab3:
            st.markdown("### Semua Features")
            
            st.dataframe(
                feature_importance.style.format({'importance': '{:.4f}'}),
                use_container_width=True,
                hide_index=True
            )
            
            st.markdown("""
            <style>
            div.stDownloadButton > button {
                background-color: #120b8f;
                color: #ffffff;
                border-radius: 8px;
                padding: 0.55em 1.3em;
                font-weight: 600;
                border: none;
            }

            div.stDownloadButton > button:hover {
                background-color: #110b78;
                color: #ffffff;
            }
            </style>
            """, unsafe_allow_html=True)

            # Download button
            csv = feature_importance.to_csv(index=False)
            st.download_button(
                label="Download Feature Importance sebagai CSV",
                data=csv,
                file_name='feature_importance.csv',
                mime='text/csv'
            )
    
    else:
        st.error("File feature importance tidak ditemukan!")


# =====================================================
# HALAMAN: INFO PENELITIAN
# =====================================================
elif menu == "Info Penelitian":
    st.markdown('<div class="sub-header">Informasi Penelitian</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Prediksi Harga Saham PT Bank Central Asia Tbk (BBCA)
    ### Menggunakan Algoritma Extreme Gradient Boosting (XGBoost)
    
    ---
    
    #### Peneliti
    - **Nama**: Alexius Kenriko Salim
    - **NIM**: 22101152630094
    - **Program Studi**: Teknik Informatika
    - **Universitas**: Universitas Putra Indonesia (YPTK) Padang
                
    ---
    
    #### Metodologi
    
    **1. Data Collection**
    - Sumber: Data historis harga saham BBCA
    - Periode: Dari data yang tersedia di CSV
    - Fitur: Open, High, Low, Close, Volume
    
    **2. Feature Engineering**
    
    *Basic Features:*
    - Lag features (1-10 hari)
    - Return features
    - Rolling statistics (mean, std)
    - OHLC-derived features
    
    ---
    
    #### Metrik Evaluasi            
    - **RMSE** (Root Mean Squared Error): Mengukur magnitude error
    - **MAE** (Mean Absolute Error): Error rata-rata absolut
    - **MAPE** (Mean Absolute Percentage Error): Error dalam persentase
    - **R²** (Coefficient of Determination): Proporsi variansi yang dijelaskan
    
    
    """)
    
    # Tampilkan hasil
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Prediksi 1 Hari:**")
        st.markdown(f"- RMSE: Rp {hasil_harian['RMSE']:,.0f}")
        st.markdown(f"- MAE: Rp {hasil_harian['MAE']:,.0f}")
        st.markdown(f"- MAPE: {hasil_harian['MAPE']:.2f}%")
        st.markdown(f"- R²: {hasil_harian['R2']:.4f}")
    
    with col2:
        st.markdown("**Prediksi 1 Minggu:**")
        st.markdown(f"- RMSE: Rp {hasil_mingguan['RMSE']:,.0f}")
        st.markdown(f"- MAE: Rp {hasil_mingguan['MAE']:,.0f}")
        st.markdown(f"- MAPE: {hasil_mingguan['MAPE']:.2f}%")
        st.markdown(f"- R²: {hasil_mingguan['R2']:.4f}")
    
    st.markdown("""
    
    """, unsafe_allow_html=True)


# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #a8a8a8; padding: 1rem;'>
    <p>© 2026 - Alexius Kenriko Salim</p>
</div>
""", unsafe_allow_html=True)