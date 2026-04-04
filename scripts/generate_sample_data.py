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
    HOURLY_DIR, MINUTE_DIR,
    HOURLY_LOOKBACK_DAYS, MINUTE_LOOKBACK_DAYS,
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
    """Generate synthetic English news across the full 5-year date range."""
    NEWS_EN_DIR.mkdir(parents=True, exist_ok=True)

    templates = [
        ("Yuan {dir} against dollar amid {catalyst}", "Reuters"),
        ("PBOC sets midpoint rate {adj} than expected", "Bloomberg"),
        ("Fed {action} in {month} meeting, dollar {dxy_dir}", "CNBC"),
        ("China GDP growth {meets} expectations at {gdp}%", "FT"),
        ("US-China trade talks {progress}", "AP"),
        ("Capital {flow} China {verb} in {month}", "Reuters"),
        ("Dollar index hits {period} {extreme} on {reason}", "Bloomberg"),
        ("PBOC {pboc_action}", "Xinhua"),
        ("Yuan {dir2} on {data_type} data", "Reuters"),
        ("Federal Reserve {fed_action}", "WSJ"),
        ("China manufacturing PMI {pmi_dir} for {nth} month", "Caixin"),
        ("US CPI data comes in {temp} than expected", "Bloomberg"),
        ("Offshore yuan {move} to {period2} {extreme2}", "FT"),
        ("China FX reserves {dir3} to ${reserves} trillion", "Bloomberg"),
        ("US tariff announcement {impact} forex markets", "CNBC"),
        ("Yuan-dollar volatility hits {period3} {extreme3}", "Reuters"),
    ]

    dates = pd.bdate_range(start=START_DATE, end=END_DATE)
    articles = []
    for d in dates:
        if np.random.random() > 0.4:  # ~60% of days have news
            n = np.random.randint(1, 4)
            for _ in range(n):
                tmpl_text, source = templates[np.random.randint(0, len(templates))]
                title = tmpl_text.format(
                    dir=np.random.choice(["weakens", "strengthens", "steadies"]),
                    dir2=np.random.choice(["rallies", "slips", "edges higher"]),
                    dir3=np.random.choice(["rise", "fall", "hold steady"]),
                    catalyst=np.random.choice(["trade tensions", "risk aversion", "rate expectations", "data surprise"]),
                    adj=np.random.choice(["stronger", "weaker"]),
                    action=np.random.choice(["signals rate cut", "holds rates steady", "raises rates"]),
                    month=np.random.choice(["January", "March", "June", "September", "December"]),
                    dxy_dir=np.random.choice(["surges", "retreats", "holds"]),
                    meets=np.random.choice(["beats", "misses", "meets"]),
                    gdp=round(np.random.uniform(4.0, 6.5), 1),
                    progress=np.random.choice(["resume in Washington", "stall over subsidies", "reach tentative deal"]),
                    flow=np.random.choice(["outflows from", "inflows to"]),
                    verb=np.random.choice(["accelerate", "slow", "stabilize"]),
                    period=np.random.choice(["3-month", "6-month", "1-year"]),
                    extreme=np.random.choice(["high", "low"]),
                    reason=np.random.choice(["dovish Fed", "hawkish Fed", "strong jobs data", "weak PMI"]),
                    pboc_action=np.random.choice(["cuts reserve requirement ratio by 50bps",
                                                    "injects $20B via MLF", "maintains stability in daily fixing",
                                                    "strengthens counter-cyclical factor"]),
                    data_type=np.random.choice(["positive trade balance", "weak export", "strong PMI"]),
                    fed_action=np.random.choice(["holds rates steady as expected",
                                                  "cuts rate by 25bps, signals pause",
                                                  "raises rate by 25bps, remains hawkish"]),
                    pmi_dir=np.random.choice(["expands", "contracts"]),
                    nth=np.random.choice(["second", "third", "fourth"]),
                    temp=np.random.choice(["hotter", "cooler"]),
                    move=np.random.choice(["drops", "surges", "edges"]),
                    period2=np.random.choice(["two-week", "one-month", "three-month"]),
                    extreme2=np.random.choice(["low", "high"]),
                    reserves=round(np.random.uniform(3.0, 3.3), 2),
                    impact=np.random.choice(["shakes", "boosts", "weighs on"]),
                    period3=np.random.choice(["3-month", "6-month", "1-year"]),
                    extreme3=np.random.choice(["high", "low"]),
                )
                articles.append({
                    "url": f"https://example.com/news/{d.strftime('%Y%m%d')}/{len(articles)}",
                    "title": title,
                    "seendate": f"{d.strftime('%Y-%m-%d')}T{np.random.randint(6,22):02d}:{np.random.randint(0,60):02d}:00Z",
                    "source": source,
                    "domain": f"{source.lower().replace(' ', '')}.com",
                    "language": "English",
                    "tone": float(np.random.uniform(-5, 5)),
                    "query_group": "yuan dollar",
                })

    with open(NEWS_EN_DIR / "gdelt_cnhusd.jsonl", "w") as f:
        for a in articles:
            f.write(json.dumps(a) + "\n")

    # Finnhub format — subset
    with open(NEWS_EN_DIR / "finnhub_forex.jsonl", "w") as f:
        for a in articles[::5]:  # every 5th
            record = {
                "id": hash(a["title"]),
                "headline": a["title"],
                "summary": f"Market analysis: {a['title']}.",
                "source": a["source"],
                "url": a["url"],
                "datetime": pd.Timestamp(a["seendate"]).timestamp(),
                "category": "forex",
            }
            f.write(json.dumps(record) + "\n")

    logger.info(f"Generated {len(articles)} synthetic English news articles (full date range)")


def _generate_intraday_price(dates_index, base_price=6.8, return_std=0.001):
    """Generate synthetic intraday OHLCV on arbitrary DatetimeIndex."""
    n = len(dates_index)
    returns = np.random.normal(0.00001, return_std, n)
    prices = [base_price]
    for r in returns[1:]:
        mean_rev = -0.0005 * (prices[-1] - 7.0)
        prices.append(prices[-1] * (1 + r + mean_rev))
    close = np.array(prices)
    bar_range = np.abs(np.random.normal(0.003, 0.001, n))
    high = close + bar_range * close * 0.5
    low = close - bar_range * close * 0.5
    open_price = close + np.random.normal(0, 0.0003, n) * close
    volume = np.random.lognormal(12, 1, n).astype(int)
    return pd.DataFrame({
        "Open": open_price, "High": high, "Low": low,
        "Close": close, "Volume": volume,
    }, index=dates_index)


def generate_hourly_price_data(output_dir=None):
    """Generate synthetic hourly OHLCV data (~1 year of hourly bars)."""
    out_dir = output_dir or HOURLY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    end = pd.Timestamp.now()
    start = end - pd.Timedelta(days=HOURLY_LOOKBACK_DAYS)
    # Business-hours only: Mon-Fri, approximate forex hours
    dates = pd.date_range(start=start, end=end, freq="h")
    # Filter to weekdays only (forex market)
    dates = dates[dates.weekday < 5]

    cnh = _generate_intraday_price(dates, base_price=6.8, return_std=0.0008)
    cnh.index.name = "Datetime"
    ohlcv_path = out_dir / "USDCNH_hourly_ohlcv.parquet"
    cnh.to_parquet(ohlcv_path)
    logger.info(f"Generated hourly OHLCV: {len(cnh)} rows -> {ohlcv_path}")

    # Technical indicators
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    from phase1_price_technical import compute_technical_indicators
    tech = compute_technical_indicators(cnh)
    tech_path = out_dir / "hourly_technical_indicators.parquet"
    tech.to_parquet(tech_path)
    logger.info(f"Generated hourly technical indicators: {len(tech.columns)} columns")
    return cnh


def generate_minute_price_data(output_dir=None):
    """Generate synthetic minute OHLCV data (~7 days of minute bars)."""
    out_dir = output_dir or MINUTE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    end = pd.Timestamp.now()
    start = end - pd.Timedelta(days=MINUTE_LOOKBACK_DAYS)
    dates = pd.date_range(start=start, end=end, freq="min")
    # Filter to weekdays only
    dates = dates[dates.weekday < 5]

    cnh = _generate_intraday_price(dates, base_price=6.8, return_std=0.0001)
    cnh.index.name = "Datetime"
    ohlcv_path = out_dir / "USDCNH_minute_ohlcv.parquet"
    cnh.to_parquet(ohlcv_path)
    logger.info(f"Generated minute OHLCV: {len(cnh)} rows -> {ohlcv_path}")

    # Technical indicators
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    from phase1_price_technical import compute_technical_indicators
    tech = compute_technical_indicators(cnh)
    tech_path = out_dir / "minute_technical_indicators.parquet"
    tech.to_parquet(tech_path)
    logger.info(f"Generated minute technical indicators: {len(tech.columns)} columns")
    return cnh


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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--intraday", choices=["1h", "1m"],
                        help="Generate only a specific intraday dataset")
    args, _ = parser.parse_known_args()

    if args.intraday:
        # Called as fallback from phase1 intraday collection
        if args.intraday == "1h":
            generate_hourly_price_data()
        else:
            generate_minute_price_data()
        return

    logger.info("=" * 60)
    logger.info("Generating sample data for pipeline testing")
    logger.info("=" * 60)

    # Price data
    logger.info("\n=== Price Data ===")
    cnh = generate_price_data()

    # Technical indicators
    logger.info("\n=== Technical Indicators ===")
    generate_technical_indicators(cnh)

    # Intraday data
    logger.info("\n=== Hourly Data ===")
    generate_hourly_price_data()

    logger.info("\n=== Minute Data ===")
    generate_minute_price_data()

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
