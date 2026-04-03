"""
CNH/USD Multimodal Dataset - Configuration
"""
import os
from pathlib import Path

# === Date Range ===
START_DATE = "2021-01-01"
END_DATE = "2026-04-01"

# === Paths ===
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "cnhusd_multimodal_dataset"
TIMESERIES_DIR = DATA_DIR / "timeseries"
MACRO_DIR = DATA_DIR / "macro"
TEXT_DIR = DATA_DIR / "text"
NEWS_EN_DIR = TEXT_DIR / "news_en"
NEWS_CN_DIR = TEXT_DIR / "news_cn"
CENTRAL_BANK_DIR = TEXT_DIR / "central_bank"
SENTIMENT_DIR = DATA_DIR / "sentiment"
IMAGES_DIR = DATA_DIR / "images"
CANDLESTICK_DIR = IMAGES_DIR / "candlestick"
GAF_DIR = IMAGES_DIR / "gaf"
ALIGNED_DIR = DATA_DIR / "aligned"
EXPLANATIONS_DIR = DATA_DIR / "explanations"
METADATA_DIR = DATA_DIR / "metadata"

# === API Keys (set via environment variables) ===
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# === yfinance Tickers ===
TICKER_CNH = "CNH=X"       # Offshore RMB (USD/CNH)
TICKER_CNY = "CNY=X"       # Onshore RMB (USD/CNY)
TICKER_DXY = "DX-Y.NYB"    # US Dollar Index
TICKER_VIX = "^VIX"        # VIX Fear Index

# === FRED Series IDs ===
FRED_SERIES = {
    # US Macro
    "fed_funds_rate": "FEDFUNDS",
    "us_cpi": "CPIAUCSL",
    "us_pmi_employment": "MANEMP",
    "us_trade_balance": "BOPGSTB",
    "us_10y_treasury": "DGS10",
    "us_m2": "M2SL",
    "us_unemployment": "UNRATE",
    "us_industrial_production": "INDPRO",
    "us_retail_sales": "RSAFS",
    # China Macro (available on FRED)
    "cn_cpi": "CHNCPIALLMINMEI",
    "cn_fx_reserves": "TRESEGCNM052N",
    # CNY Exchange Rate (PBOC midprice proxy)
    "dexchus": "DEXCHUS",
}

# === GDELT Search Keywords (CNH/USD specific) ===
GDELT_KEYWORDS = [
    '"yuan dollar" OR "USDCNH" OR "CNH"',
    '"PBOC" OR "People\'s Bank of China"',
    '"China trade" OR "US China tariff"',
    '"China economy" OR "Chinese GDP"',
    '"Fed rate" OR "Federal Reserve"',
    '"capital outflow China" OR "China FX reserves"',
]

# === Technical Indicator Parameters ===
TECH_PARAMS = {
    "ema_periods": [5, 20, 50],
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "atr_period": 14,
    "bb_period": 20,
    "bb_std": 2.0,
    "adx_period": 14,
}

# === Chart Generation ===
CHART_LOOKBACKS = {
    "5d": 5,
    "20d": 20,
    "60d": 60,
}
CHART_SIZE = (384, 384)  # pixels, ViT-friendly
