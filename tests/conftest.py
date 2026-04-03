"""
Shared fixtures for all smoke tests.
Generates sample data in a temporary directory so tests are isolated.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


@pytest.fixture(scope="session")
def tmp_data_dir(tmp_path_factory):
    """Create a temporary data directory mimicking the real dataset layout."""
    base = tmp_path_factory.mktemp("dataset")
    for subdir in [
        "timeseries", "macro",
        "text/news_en", "text/news_cn", "text/central_bank",
        "sentiment", "images/candlestick", "images/gaf",
        "aligned", "explanations", "metadata",
    ]:
        (base / subdir).mkdir(parents=True, exist_ok=True)
    return base


@pytest.fixture(scope="session")
def sample_ohlcv():
    """Generate a small OHLCV DataFrame for testing."""
    np.random.seed(42)
    n = 100
    dates = pd.bdate_range(start="2025-01-01", periods=n)
    close = 7.0 + np.cumsum(np.random.normal(0, 0.01, n))
    return pd.DataFrame({
        "Open": close + np.random.normal(0, 0.005, n),
        "High": close + np.abs(np.random.normal(0.01, 0.005, n)),
        "Low": close - np.abs(np.random.normal(0.01, 0.005, n)),
        "Close": close,
        "Volume": np.random.randint(1000, 100000, n),
    }, index=dates)


@pytest.fixture(scope="session")
def sample_ohlcv_parquet(tmp_data_dir, sample_ohlcv):
    """Save sample OHLCV to a parquet file and return path."""
    path = tmp_data_dir / "timeseries" / "USDCNH_daily_ohlcv.parquet"
    sample_ohlcv.to_parquet(path)
    return path


@pytest.fixture(scope="session")
def sample_spread(tmp_data_dir, sample_ohlcv):
    """Generate and save CNH-CNY spread data."""
    spread = pd.DataFrame({
        "cnh_close": sample_ohlcv["Close"],
        "cny_close": sample_ohlcv["Close"] + np.random.normal(0, 0.003, len(sample_ohlcv)),
        "cnh_cny_spread": np.random.normal(0, 0.003, len(sample_ohlcv)),
    }, index=sample_ohlcv.index)
    path = tmp_data_dir / "timeseries" / "cnh_cny_spread.parquet"
    spread.to_parquet(path)
    return path


@pytest.fixture(scope="session")
def sample_macro(tmp_data_dir):
    """Generate and save sample macro data."""
    dates = pd.date_range("2025-01-01", periods=12, freq="MS")
    us = pd.DataFrame({
        "fed_funds_rate": np.linspace(4.0, 3.5, 12),
        "us_cpi": np.linspace(300, 305, 12),
    }, index=dates)
    cn = pd.DataFrame({
        "cn_cpi": np.linspace(102, 103, 12),
        "dexchus": np.linspace(7.0, 7.1, 12),
    }, index=dates)
    us_path = tmp_data_dir / "macro" / "us_macro.parquet"
    cn_path = tmp_data_dir / "macro" / "cn_macro_fred.parquet"
    us.to_parquet(us_path)
    cn.to_parquet(cn_path)
    return us_path, cn_path


@pytest.fixture(scope="session")
def sample_news_jsonl(tmp_data_dir):
    """Generate sample news JSONL files."""
    news_dir = tmp_data_dir / "text" / "news_en"

    gdelt_path = news_dir / "gdelt_cnhusd.jsonl"
    with open(gdelt_path, "w") as f:
        for i in range(5):
            record = {
                "url": f"https://example.com/news/{i}",
                "title": f"Yuan weakens amid trade tensions day {i}",
                "seendate": f"2025-03-{10+i:02d}T12:00:00Z",
                "source": "Reuters",
                "tone": float(np.random.uniform(-5, 5)),
            }
            f.write(json.dumps(record) + "\n")

    finnhub_path = news_dir / "finnhub_forex.jsonl"
    with open(finnhub_path, "w") as f:
        for i in range(3):
            record = {
                "id": i + 100,
                "headline": f"Fed signals rate changes headline {i}",
                "summary": f"Detailed summary about forex market {i}",
                "source": "Bloomberg",
                "datetime": 1741000000 + i * 86400,
                "category": "forex",
            }
            f.write(json.dumps(record) + "\n")

    return gdelt_path, finnhub_path


@pytest.fixture(scope="session")
def sample_central_bank_jsonl(tmp_data_dir):
    """Generate sample central bank JSONL files."""
    cb_dir = tmp_data_dir / "text" / "central_bank"

    fomc_path = cb_dir / "fomc_labeled.jsonl"
    with open(fomc_path, "w") as f:
        for sentence, label in [
            ("The Committee decided to raise rates.", "hawkish"),
            ("Economic activity expanded at a moderate pace.", "neutral"),
            ("The Committee will adjust policy as appropriate.", "dovish"),
        ]:
            f.write(json.dumps({"sentence": sentence, "label": label, "split": "train"}) + "\n")

    fed_path = cb_dir / "fed_statements.jsonl"
    with open(fed_path, "w") as f:
        f.write(json.dumps({
            "date": "20250129",
            "url": "https://fed.gov/test",
            "text": "The FOMC decided to maintain rates at 4.25 percent. " * 5,
            "type": "fomc_statement",
        }) + "\n")

    return fomc_path, fed_path
