# CNH/USD Multimodal Explainable Prediction Dataset

A comprehensive multimodal dataset for USD/CNH (offshore RMB) exchange rate prediction with explainability.

## Overview

- **Target pair**: USD/CNH (offshore renminbi)
- **Date range**: 2021-01-01 to 2026-03-31 (~1300 trading days)
- **Frequency**: Daily
- **Modalities**: Time series, text, images
- **Reference**: Yu et al. "Temporal Data Meets LLM" (EMNLP 2023)

## Dataset Structure

Each trading day sample contains **40 features** across 3 modalities:

| Modality | Features | Source |
|----------|----------|--------|
| Price OHLCV | Open, High, Low, Close, Volume | yfinance |
| Technical Indicators | EMA, RSI, MACD, ATR, Bollinger, ADX, OBV (15 cols) | Computed |
| Macro Economic | Fed rate, CPI, PMI, trade balance, M2, etc. (9 cols) | FRED API |
| PBOC Midprice | Daily fixing rate proxy | FRED DEXCHUS |
| CNH-CNY Spread | Offshore-onshore spread | Computed |
| Auxiliary | DXY, VIX | yfinance |
| News | Headlines + count | GDELT, Finnhub |
| Chart Images | 5d/20d/60d candlestick PNGs (384x384) | mplfinance |
| Targets | t+1 return + direction | Computed |

## Quick Start

### 1. Install dependencies
```bash
pip install yfinance fredapi gdeltdoc pyarrow mplfinance
```

### 2. Set API keys
```bash
export FRED_API_KEY='your_key'        # https://fred.stlouisfed.org/docs/api/api_key.html
export FINNHUB_API_KEY='your_key'     # https://finnhub.io/register (optional)
```

### 3. Run data collection
```bash
# Run all phases
python scripts/run_all.py

# Or run individual phases
python scripts/phase1_price_technical.py   # Price + technical indicators
python scripts/phase2_macro.py             # Macroeconomic data (FRED)
python scripts/phase3_news_en.py           # English news (GDELT + Finnhub)
python scripts/phase5_central_bank.py      # Central bank communications
python scripts/phase6_sentiment.py         # Sentiment scoring (FinBERT)
python scripts/phase7_charts.py            # K-line chart generation
python scripts/phase8_alignment.py         # Final alignment & assembly
```

### 4. For testing without API access
```bash
python scripts/generate_sample_data.py     # Generate synthetic data
python scripts/phase7_charts.py            # Generate charts from synthetic data
python scripts/phase8_alignment.py         # Align everything
```

## Directory Structure

```
cnhusd_multimodal_dataset/
├── timeseries/          # Price OHLCV, PBOC midprice, technical indicators
├── macro/               # US and China macroeconomic indicators
├── text/
│   ├── news_en/         # English news (GDELT, Finnhub)
│   ├── news_cn/         # Chinese news (Sina, Eastmoney) [Phase 4]
│   └── central_bank/    # FOMC labeled data, Fed statements
├── sentiment/           # FinBERT scores, hawk/dove classification
├── images/
│   ├── candlestick/     # K-line chart PNGs
│   └── gaf/             # Gramian Angular Field encodings
├── aligned/             # Final aligned dataset (parquet)
├── explanations/        # LLM-generated explanations [Phase 9]
└── metadata/            # Data source configs, collection logs
```

## Data Sources

| Source | Content | Auth | Cost |
|--------|---------|------|------|
| yfinance | CNH/CNY/DXY/VIX prices | None | Free |
| FRED API | 12+ macro series | Free key | Free |
| GDELT DOC API | English news (3mo) | None | Free |
| Finnhub | Forex news + calendar | Free key | Free |
| HuggingFace | FOMC labeled data | None | Free |

## CNH/USD Unique Features

- **PBOC Midprice**: Daily central bank fixing rate (policy signal)
- **CNH-CNY Spread**: Offshore-onshore gap (capital control pressure indicator)
- **China-specific events**: Trade talks, PBOC RRR cuts, Two Sessions, etc.

## Pipeline Phases

| Phase | Description | Dependencies |
|-------|-------------|-------------|
| 1 | Price + Technical Indicators | yfinance |
| 2 | Macroeconomic Data | FRED API key |
| 3 | English News | GDELT (no key), Finnhub (optional) |
| 4 | Chinese News | Web scraping (Sina, Eastmoney) |
| 5 | Central Bank Communications | HuggingFace, Fed website |
| 6 | Sentiment Scoring | FinBERT model (transformers) |
| 7 | Chart Generation | mplfinance (from Phase 1 data) |
| 8 | Data Alignment | All previous phases |
| 9 | LLM Explanations | Qwen2.5 (future) |
