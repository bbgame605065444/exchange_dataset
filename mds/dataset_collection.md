# CNH/USD Multimodal Dataset - Collection Report

## 1. Overview

This document records the data collection results for the CNH/USD multimodal exchange rate dataset. The pipeline consists of 9 phases covering price data, macroeconomic indicators, news, sentiment, charts, and LLM explanations.

- **Date Range**: 2021-01-01 to 2026-04-01 (5 years, 1,369 trading days)
- **Collection Date**: 2026-04-04
- **Pipeline**: `scripts/collect_all.py`

---

## 2. Collection Results by Phase

### Phase 1 — Price Data + Technical Indicators

| Source | Status | Records | Detail |
|--------|--------|---------|--------|
| yfinance/CNH (daily) | FAILED | 0 | Proxy 403 blocked |
| yfinance/CNY (daily) | FAILED | 0 | Proxy 403 blocked |
| yfinance/DXY | FAILED | 0 | Proxy 403 blocked |
| yfinance/VIX | FAILED | 0 | Proxy 403 blocked |
| **synthetic/prices (fallback)** | **OK** | **1,369** | Mean-reversion random walk |

**Output Files**:
- `timeseries/USDCNH_daily_ohlcv.parquet` — 1,369 rows, OHLCV (75 KB)
- `timeseries/USDCNY_daily_ohlcv.parquet` — 1,369 rows (75 KB)
- `timeseries/cnh_cny_spread.parquet` — CNH-CNY spread (53 KB)
- `timeseries/auxiliary_indicators.parquet` — DXY + VIX (39 KB)
- `timeseries/technical_indicators.parquet` — 15 technical features (206 KB)

**Technical Indicators Computed (15 columns)**:
`ema_5`, `ema_20`, `ema_50`, `rsi_14`, `macd`, `macd_signal`, `macd_hist`, `atr_14`, `bb_upper`, `bb_middle`, `bb_lower`, `adx`, `plus_di`, `minus_di`, `obv`

### Phase 1b — Intraday Price Data (hourly + minute)

| Source | Status | Records | Detail |
|--------|--------|---------|--------|
| yfinance/hourly | FAILED | 0 | Proxy 403 blocked |
| yfinance/minute | FAILED | 0 | Proxy 403 blocked |
| **synthetic/hourly (fallback)** | **OK** | **6,258** | 1-year hourly bars (weekdays) |
| **synthetic/minute (fallback)** | **OK** | **7,200** | 7-day minute bars (weekdays) |

**Output Files**:
- `timeseries/hourly/USDCNH_hourly_ohlcv.parquet` — 6,258 rows (332 KB)
- `timeseries/hourly/hourly_technical_indicators.parquet` — 15 features (930 KB)
- `timeseries/minute/USDCNH_minute_ohlcv.parquet` — 7,200 rows (375 KB)
- `timeseries/minute/minute_technical_indicators.parquet` — 15 features (1.1 MB)

### Phase 2 — Macroeconomic Data (FRED)

| Source | Status | Records | Detail |
|--------|--------|---------|--------|
| FRED API | SKIPPED | 0 | `FRED_API_KEY` not set |
| **synthetic/macro (fallback)** | **OK** | **64 months** | US + CN macro indicators |

**Output Files**:
- `macro/us_macro.parquet` — 64 rows, 6 columns (8.9 KB)
  - `fed_funds_rate`, `us_cpi`, `us_10y_treasury`, `us_trade_balance`, `us_m2`, `us_unemployment`
- `macro/cn_macro_fred.parquet` — 64 rows, 3 columns (5.5 KB)
  - `cn_cpi`, `cn_fx_reserves`, `dexchus`
- `timeseries/pboc_midprice.parquet` — daily PBOC proxy (15 KB)

### Phase 3 — English News (GDELT + Finnhub)

| Source | Status | Records | Detail |
|--------|--------|---------|--------|
| GDELT DOC API | FAILED | 0 | Proxy 403 blocked |
| Finnhub News | SKIPPED | 0 | `FINNHUB_API_KEY` not set |
| **synthetic/news (fallback)** | **OK** | **1,676** | Templated headline generation |

**Output Files**:
- `text/news_en/gdelt_cnhusd.jsonl` — 1,676 articles (436 KB)
- `text/news_en/finnhub_forex.jsonl` — 335 articles (92 KB)

### Phase 4 — Chinese News (Sina + Eastmoney)

| Source | Status | Records | Detail |
|--------|--------|---------|--------|
| Sina Finance | FAILED | 0 | Proxy 403 blocked |
| Eastmoney | FAILED | 0 | Proxy 403 blocked |
| **synthetic/news_cn (fallback)** | **OK** | **1,923** | Chinese-language templates |

**Output Files**:
- `text/news_cn/sina_forex.jsonl` — 1,923 articles (253 KB)

### Phase 5 — Central Bank Communications

| Source | Status | Records | Detail |
|--------|--------|---------|--------|
| HuggingFace FOMC | FAILED | 0 | 403 Forbidden |
| Fed Statements | FAILED | 0 | Proxy 403 blocked |
| Finnhub Calendar | SKIPPED | 0 | `FINNHUB_API_KEY` not set |
| **synthetic/central_bank (fallback)** | **OK** | **7** | 5 FOMC + 2 Fed statements |

**Output Files**:
- `text/central_bank/fomc_labeled.jsonl` — 5 labeled FOMC sentences (586 B)
- `text/central_bank/fed_statements.jsonl` — 2 Fed statements (658 B)

### Phase 6 — Sentiment Scoring

| Source | Status | Records | Detail |
|--------|--------|---------|--------|
| FinBERT (English) | FAILED | 0 | PyTorch not installed |
| **Keyword/cn_sentiment** | **OK** | **1,923** | Keyword-based pos/neg/neutral |

**Output Files**:
- `sentiment/cn_sentiment.parquet` — 1,923 scored articles (27 KB)

### Phase 7 — K-line Chart Generation

| Source | Status | Records | Detail |
|--------|--------|---------|--------|
| **mplfinance charts** | **OK** | **204** | Every 20th trading day, 3 lookbacks |

**Output Files**:
- `images/candlestick/USDCNH_{YYYYMMDD}_{5d|20d|60d}.png` — 204 PNG files (~6 MB total)
- Chart size: 384x384 pixels (ViT-friendly)
- 68 dates x 3 lookbacks (5d, 20d, 60d)

### Phase 8 — Data Alignment & Assembly

| Source | Status | Detail |
|--------|--------|--------|
| **Alignment** | **OK** | 1,369 rows x 41 columns |

**Output File**:
- `aligned/USDCNH_daily_aligned.parquet` — Master dataset (397 KB)

**Alignment Statistics**:
- Trading days: 1,369
- Days with news: 1,230 (89.8%)
- Days with macro: 1,369 (100%)
- Days with tech: 1,369 (100%)
- Chart modalities: 3/3
- Target distribution: +1 (690), -1 (678), 0 (1)

**Columns (41)**:
| Category | Columns | Count |
|----------|---------|-------|
| OHLCV | `open`, `high`, `low`, `close`, `volume` | 5 |
| Spread | `cnh_cny_spread` | 1 |
| PBOC | `pboc_midprice` | 1 |
| Technical | `tech_ema_5` ... `tech_obv` | 15 |
| Auxiliary | `dxy_close`, `vix_close` | 2 |
| Macro US | `macro_us_fed_funds_rate` ... `macro_us_us_unemployment` | 6 |
| Macro CN | `macro_cn_cn_cpi`, `macro_cn_cn_fx_reserves`, `macro_cn_dexchus` | 3 |
| News | `news_count`, `news_titles`, `news_avg_tone` | 3 |
| Charts | `chart_5d_path`, `chart_20d_path`, `chart_60d_path` | 3 |
| Target | `target_return`, `target_direction` | 2 |

### Phase 9 — LLM Explanations

| Source | Status | Records | Detail |
|--------|--------|---------|--------|
| **Offline mode** | **OK** | **275** | Prompts only (Qwen model 403) |

**Output File**:
- `explanations/USDCNH_llm_explanations.jsonl` — 275 weekly prompts (387 KB)

---

## 3. Final Dataset Summary

### Three Granularity Levels

| Dataset | Path | Rows | Columns | Size |
|---------|------|------|---------|------|
| **Minute-level** | `timeseries/minute/` | 7,200 | 5 + 15 tech | 1.5 MB |
| **Hourly** | `timeseries/hourly/` | 6,258 | 5 + 15 tech | 1.3 MB |
| **Daily (full multimodal)** | `aligned/USDCNH_daily_aligned.parquet` | 1,369 | 41 | 397 KB |

### Total Disk Usage

| Directory | Size |
|-----------|------|
| `timeseries/` | 3.2 MB |
| `images/` | 6.0 MB |
| `text/` | 808 KB |
| `aligned/` | 404 KB |
| `explanations/` | 392 KB |
| `sentiment/` | 32 KB |
| `macro/` | 24 KB |
| `metadata/` | 12 KB |
| **Total** | **~11 MB** |

---

## 4. How to Run

### Prerequisites

```bash
pip install numpy pandas pyarrow yfinance fredapi gdeltdoc finnhub-python \
    beautifulsoup4 lxml requests mplfinance matplotlib datasets transformers torch
```

### Environment Variables (for real API data)

```bash
export FRED_API_KEY="your_fred_api_key"        # https://fred.stlouisfed.org/docs/api/api_key.html
export FINNHUB_API_KEY="your_finnhub_key"      # https://finnhub.io/dashboard
export HF_TOKEN="your_huggingface_token"       # https://huggingface.co/settings/tokens
```

### Run Full Pipeline

```bash
# Full run (all 9 phases, charts every 5th day)
python scripts/collect_all.py

# Fast run (skip charts and LLM explanations)
python scripts/collect_all.py --fast

# Custom chart interval (every 20th trading day)
python scripts/collect_all.py --chart-every 20
```

### Run Individual Phases

```bash
# Run specific phases only
python scripts/run_all.py --phases 1,2,3

# Or run phase scripts directly
python scripts/phase1_price_technical.py    # Price + Tech + Intraday
python scripts/phase2_macro.py              # Macro (needs FRED_API_KEY)
python scripts/phase3_news_en.py            # English News
python scripts/phase4_news_cn.py            # Chinese News
python scripts/phase5_central_bank.py       # Central Bank
python scripts/phase6_sentiment.py          # Sentiment
python scripts/phase7_charts.py             # K-line Charts
python scripts/phase8_alignment.py          # Alignment
python scripts/phase9_explanations.py       # LLM Explanations
```

### Generate Synthetic Data Only

```bash
# Generate all synthetic data (no API calls)
python scripts/generate_sample_data.py

# Generate only hourly synthetic data
python scripts/generate_sample_data.py --intraday 1h

# Generate only minute synthetic data
python scripts/generate_sample_data.py --intraday 1m
```

### Run Tests

```bash
# All tests (93 tests)
python -m pytest tests/ -v

# Specific test modules
python -m pytest tests/test_phase1_technical.py -v    # 20 tests
python -m pytest tests/test_generate_sample_data.py -v # 17 tests
python -m pytest tests/test_phase8_alignment.py -v     # 7 tests
```

---

## 5. Synthetic vs Real Data: How to Differentiate

### Current Status

In the current collection run (2026-04-04), **all data sources used synthetic fallback** due to network proxy restrictions (403 Forbidden). The pipeline automatically falls back to synthetic data when:

1. API keys are not set (`FRED_API_KEY`, `FINNHUB_API_KEY`)
2. External APIs are unreachable (proxy, network errors)
3. Required libraries are missing (`torch` for FinBERT)

### How to Separate Synthetic and Real Data

To maintain clear separation between synthetic and real data, adopt the following directory structure:

```
cnhusd_multimodal_dataset/
├── real/                          # <-- Real API data (when available)
│   ├── timeseries/
│   ├── macro/
│   ├── text/
│   └── ...
├── synthetic/                     # <-- Synthetic fallback data
│   ├── timeseries/
│   ├── macro/
│   ├── text/
│   └── ...
├── aligned/                       # <-- Final aligned dataset
│   ├── USDCNH_daily_aligned.parquet        # Uses best available source
│   └── USDCNH_daily_aligned_synthetic.parquet  # Pure synthetic version
└── metadata/
    └── collection_log.jsonl       # <-- Records which source was used
```

### Identification via Collection Log

The file `metadata/collection_log.jsonl` records each source's status:

```json
{"source": "yfinance/CNH", "status": "failed", "detail": "no data returned", "n_records": 0}
{"source": "synthetic/prices", "status": "ok", "detail": "fallback", "n_records": 1369}
```

**Key status values**:
- `"status": "ok"` + source starts with `synthetic/` → Synthetic fallback data
- `"status": "ok"` + real source name (e.g., `yfinance/CNH`) → Real API data
- `"status": "failed"` → Source attempted but failed, check `"detail"` for reason
- `"status": "skipped"` → Source skipped (missing API key or library)

### Recommended Workflow for Real Data Collection

```bash
# Step 1: Set API keys
export FRED_API_KEY="..."
export FINNHUB_API_KEY="..."

# Step 2: Run pipeline in an environment with open network access
python scripts/collect_all.py

# Step 3: Check collection log to verify which sources succeeded
python -c "
import json
with open('cnhusd_multimodal_dataset/metadata/collection_log.jsonl') as f:
    for line in f:
        rec = json.loads(line)
        flag = 'REAL' if not rec['source'].startswith('synthetic') and rec['status'] == 'ok' else 'SYNTH'
        print(f\"  [{flag}] {rec['source']}: {rec['status']} ({rec['n_records']} records)\")
"

# Step 4: If you want to save both versions side by side
cp -r cnhusd_multimodal_dataset cnhusd_multimodal_dataset_real
python scripts/generate_sample_data.py
cp -r cnhusd_multimodal_dataset cnhusd_multimodal_dataset_synthetic
```

### Programmatic Check: Is Data Synthetic?

```python
import json

def get_data_sources(log_path="cnhusd_multimodal_dataset/metadata/collection_log.jsonl"):
    """Return dict of {source: is_synthetic} for each data source."""
    sources = {}
    with open(log_path) as f:
        for line in f:
            rec = json.loads(line)
            is_synthetic = rec["source"].startswith("synthetic/") or rec["detail"] == "fallback"
            sources[rec["source"]] = {
                "status": rec["status"],
                "is_synthetic": is_synthetic,
                "n_records": rec["n_records"],
            }
    return sources
```

---

## 6. Pipeline Architecture

```
                     collect_all.py (entry point)
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                   │
    Phase 1             Phase 2            Phase 3-5
    yfinance            FRED API        GDELT/Finnhub/Sina
    ↓ fail?             ↓ fail?            ↓ fail?
    synthetic           synthetic          synthetic
         │                  │                   │
         │             Phase 6              Phase 7
         │           Sentiment             Charts (mplfinance)
         │           (FinBERT/keyword)        │
         │                  │                  │
         └──────────────────┼──────────────────┘
                            │
                       Phase 8: Alignment
                   (daily anchor, no look-ahead)
                            │
                       Phase 9: LLM Explanations
                     (Qwen2.5 or offline prompts)
                            │
                    ┌───────┼───────┐
                    │       │       │
               minute    hourly    daily
               (7d)      (1yr)    (5yr, 41 cols)
```

---

## 7. Data Quality Notes

1. **No look-ahead bias**: Macro data is forward-filled only; news is assigned via UTC 17:00 cutoff windows
2. **Synthetic data uses seed 42**: Results are reproducible across runs
3. **Price range validation**: Synthetic USD/CNH prices are constrained to 5.0-9.0 (realistic range)
4. **Target variables**: `target_return` and `target_direction` are shifted by -1 (future labels, not features)
5. **93/93 tests pass**: Full test coverage for all pipeline components
