"""
Generate sample/synthetic data for pipeline testing.
Use this when external APIs are not accessible.
The sample data mimics real USD/CNH structure for end-to-end pipeline validation.
"""
import sys
import json
import logging

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import (
    START_DATE, END_DATE,
    TIMESERIES_DIR, MACRO_DIR, NEWS_EN_DIR, CENTRAL_BANK_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

np.random.seed(42)


def generate_price_data():
    """Generate synthetic USD/CNH OHLCV data resembling real market data."""
    TIMESERIES_DIR.mkdir(parents=True, exist_ok=True)

    dates = pd.bdate_range(start=START_DATE, end=END_DATE)
    n = len(dates)

    # USD/CNH typically ranges 6.3 - 7.4 in 2021-2026
    # Random walk with mean reversion
    base_price = 6.8
    returns = np.random.normal(0.0001, 0.003, n)
    # Add mean reversion
    prices = [base_price]
    for r in returns[1:]:
        mean_rev = -0.001 * (prices[-1] - 7.0)  # Revert toward 7.0
        prices.append(prices[-1] * (1 + r + mean_rev))
    close = np.array(prices)

    # Generate OHLV from close
    daily_range = np.abs(np.random.normal(0.01, 0.005, n))
    high = close + daily_range * close * 0.5
    low = close - daily_range * close * 0.5
    open_price = close + np.random.normal(0, 0.001, n) * close
    volume = np.random.lognormal(15, 1, n).astype(int)

    cnh = pd.DataFrame({
        "Open": open_price,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=dates)
    cnh.index.name = "Date"
    cnh.to_parquet(TIMESERIES_DIR / "USDCNH_daily_ohlcv.parquet")
    logger.info(f"Generated CNH OHLCV: {len(cnh)} rows")

    # CNY (onshore) - slightly different from CNH
    cny_close = close + np.random.normal(0, 0.005, n)
    cny = pd.DataFrame({
        "Open": cny_close + np.random.normal(0, 0.001, n),
        "High": cny_close + np.abs(np.random.normal(0.005, 0.003, n)),
        "Low": cny_close - np.abs(np.random.normal(0.005, 0.003, n)),
        "Close": cny_close,
        "Volume": volume * 2,
    }, index=dates)
    cny.to_parquet(TIMESERIES_DIR / "USDCNY_daily_ohlcv.parquet")

    # CNH-CNY Spread
    spread = pd.DataFrame({
        "cnh_close": close,
        "cny_close": cny_close,
        "cnh_cny_spread": close - cny_close,
    }, index=dates)
    spread.to_parquet(TIMESERIES_DIR / "cnh_cny_spread.parquet")
    logger.info(f"Generated CNH-CNY spread: mean={spread['cnh_cny_spread'].mean():.4f}")

    # Auxiliary: DXY and VIX
    dxy = 100 + np.cumsum(np.random.normal(0, 0.3, n))
    vix = np.abs(15 + np.cumsum(np.random.normal(0, 0.5, n)))
    aux = pd.DataFrame({
        "dxy_close": dxy,
        "vix_close": vix,
    }, index=dates)
    aux.to_parquet(TIMESERIES_DIR / "auxiliary_indicators.parquet")
    logger.info("Generated DXY and VIX auxiliary indicators")

    return cnh


def generate_technical_indicators(cnh: pd.DataFrame):
    """Compute real technical indicators from synthetic price data."""
    # Import the compute function from phase1
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    from phase1_price_technical import compute_technical_indicators
    tech = compute_technical_indicators(cnh)
    tech.to_parquet(TIMESERIES_DIR / "technical_indicators.parquet")
    logger.info(f"Generated technical indicators: {len(tech.columns)} columns")


def generate_macro_data():
    """Generate synthetic macro data."""
    MACRO_DIR.mkdir(parents=True, exist_ok=True)

    dates_monthly = pd.date_range(start=START_DATE, end=END_DATE, freq="MS")
    n = len(dates_monthly)

    us_macro = pd.DataFrame({
        "fed_funds_rate": np.clip(np.cumsum(np.random.normal(0.05, 0.1, n)), 0, 6),
        "us_cpi": 260 + np.cumsum(np.random.normal(0.5, 0.3, n)),
        "us_10y_treasury": np.clip(1.5 + np.cumsum(np.random.normal(0.02, 0.1, n)), 0.5, 5),
        "us_trade_balance": -60 + np.random.normal(0, 5, n),
        "us_m2": 20000 + np.cumsum(np.random.normal(50, 30, n)),
        "us_unemployment": np.clip(4 + np.cumsum(np.random.normal(0, 0.1, n)), 3, 10),
    }, index=dates_monthly)
    us_macro.to_parquet(MACRO_DIR / "us_macro.parquet")

    cn_macro = pd.DataFrame({
        "cn_cpi": 102 + np.random.normal(0, 0.5, n),
        "cn_fx_reserves": 3100 + np.cumsum(np.random.normal(0, 10, n)),
        "dexchus": 6.8 + np.cumsum(np.random.normal(0, 0.02, n)),
    }, index=dates_monthly)
    cn_macro.to_parquet(MACRO_DIR / "cn_macro_fred.parquet")

    # PBOC midprice proxy
    daily_dates = pd.bdate_range(start=START_DATE, end=END_DATE)
    pboc = pd.DataFrame({
        "pboc_midprice_proxy": cn_macro["dexchus"].reindex(daily_dates, method="ffill").values[:len(daily_dates)],
    }, index=daily_dates)
    pboc.to_parquet(TIMESERIES_DIR / "pboc_midprice.parquet")

    logger.info(f"Generated macro data: US={us_macro.shape}, CN={cn_macro.shape}")


def generate_news_data():
    """Generate sample news data."""
    NEWS_EN_DIR.mkdir(parents=True, exist_ok=True)

    sample_headlines = [
        ("2026-01-15", "Yuan weakens against dollar amid trade tensions", "Reuters"),
        ("2026-01-16", "PBOC sets midpoint rate stronger than expected", "Bloomberg"),
        ("2026-01-17", "Fed signals potential rate cut in March meeting", "CNBC"),
        ("2026-01-20", "China GDP growth beats expectations at 5.2%", "FT"),
        ("2026-01-21", "US-China trade talks resume in Washington", "AP"),
        ("2026-01-22", "Capital outflows from China accelerate in January", "Reuters"),
        ("2026-01-23", "Dollar index hits 3-month low on dovish Fed", "Bloomberg"),
        ("2026-01-24", "PBOC cuts reserve requirement ratio by 50bps", "Xinhua"),
        ("2026-02-01", "Yuan rallies on positive trade balance data", "Reuters"),
        ("2026-02-05", "Federal Reserve holds rates steady as expected", "WSJ"),
        ("2026-02-10", "China manufacturing PMI expands for third month", "Caixin"),
        ("2026-02-15", "US CPI data comes in hotter than expected", "Bloomberg"),
        ("2026-02-20", "Offshore yuan drops to two-week low on risk aversion", "FT"),
        ("2026-03-01", "PBOC maintains stability in daily fixing signal", "Reuters"),
        ("2026-03-10", "China FX reserves rise to $3.25 trillion", "Bloomberg"),
        ("2026-03-15", "US tariff announcement shakes forex markets", "CNBC"),
        ("2026-03-20", "Yuan-dollar volatility hits 6-month high", "Reuters"),
        ("2026-03-25", "Federal Reserve cuts rate by 25bps, signals pause", "WSJ"),
        ("2026-03-28", "China trade surplus widens to record $78 billion", "Bloomberg"),
        ("2026-03-31", "Q1 review: Yuan depreciated 2.3% against dollar", "FT"),
    ]

    # GDELT format
    with open(NEWS_EN_DIR / "gdelt_cnhusd.jsonl", "w") as f:
        for date, title, source in sample_headlines:
            record = {
                "url": f"https://example.com/news/{date}",
                "title": title,
                "seendate": f"{date}T12:00:00Z",
                "source": source,
                "domain": f"{source.lower()}.com",
                "language": "English",
                "tone": np.random.uniform(-5, 5),
                "query_group": "yuan dollar",
            }
            f.write(json.dumps(record) + "\n")

    # Finnhub format
    with open(NEWS_EN_DIR / "finnhub_forex.jsonl", "w") as f:
        for date, title, source in sample_headlines[:10]:
            record = {
                "id": hash(title),
                "headline": title,
                "summary": f"Detailed analysis: {title}. Market participants are closely watching...",
                "source": source,
                "url": f"https://example.com/{date}",
                "datetime": pd.Timestamp(date).timestamp(),
                "category": "forex",
            }
            f.write(json.dumps(record) + "\n")

    logger.info(f"Generated {len(sample_headlines)} sample news articles")


def generate_central_bank_data():
    """Generate sample central bank data."""
    CENTRAL_BANK_DIR.mkdir(parents=True, exist_ok=True)

    # FOMC labeled samples
    fomc_samples = [
        {"sentence": "The Committee decided to raise the target range for the federal funds rate.", "label": "hawkish", "split": "train"},
        {"sentence": "Economic activity has been expanding at a moderate pace.", "label": "neutral", "split": "train"},
        {"sentence": "The Committee is prepared to adjust monetary policy as appropriate.", "label": "dovish", "split": "train"},
        {"sentence": "Inflation remains elevated and above the Committee's longer-run goal.", "label": "hawkish", "split": "train"},
        {"sentence": "The labor market has shown signs of softening.", "label": "dovish", "split": "train"},
    ]
    with open(CENTRAL_BANK_DIR / "fomc_labeled.jsonl", "w") as f:
        for s in fomc_samples:
            f.write(json.dumps(s) + "\n")

    # Fed statements
    fed_statements = [
        {"date": "20260128", "url": "https://federalreserve.gov/fomc/20260128", "text": "The Federal Open Market Committee decided to maintain the target range for the federal funds rate at 4.00 to 4.25 percent. The Committee judges that the risks to achieving its employment and inflation goals are roughly in balance.", "type": "fomc_statement"},
        {"date": "20260318", "url": "https://federalreserve.gov/fomc/20260318", "text": "The Committee decided to lower the target range for the federal funds rate by 25 basis points to 3.75 to 4.00 percent. Recent indicators suggest that economic activity has continued to expand at a solid pace.", "type": "fomc_statement"},
    ]
    with open(CENTRAL_BANK_DIR / "fed_statements.jsonl", "w") as f:
        for s in fed_statements:
            f.write(json.dumps(s) + "\n")

    logger.info("Generated sample central bank data")


def main():
    logger.info("=" * 60)
    logger.info("Generating sample data for pipeline testing")
    logger.info("=" * 60)

    # Price data
    logger.info("\n=== Price Data ===")
    cnh = generate_price_data()

    # Technical indicators
    logger.info("\n=== Technical Indicators ===")
    generate_technical_indicators(cnh)

    # Macro data
    logger.info("\n=== Macro Data ===")
    generate_macro_data()

    # News data
    logger.info("\n=== News Data ===")
    generate_news_data()

    # Central bank data
    logger.info("\n=== Central Bank Data ===")
    generate_central_bank_data()

    logger.info("\n" + "=" * 60)
    logger.info("Sample data generation complete!")
    logger.info("You can now run Phase 7 (charts) and Phase 8 (alignment)")
    logger.info("For real data, run individual phase scripts with proper API access.")


if __name__ == "__main__":
    main()
