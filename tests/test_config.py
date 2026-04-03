"""Smoke tests for config.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_config_imports():
    """Config module imports without error."""
    import config
    assert hasattr(config, "START_DATE")
    assert hasattr(config, "END_DATE")
    assert hasattr(config, "FRED_SERIES")
    assert hasattr(config, "TECH_PARAMS")
    assert hasattr(config, "CHART_LOOKBACKS")


def test_config_date_range():
    """Date range is valid."""
    from config import START_DATE, END_DATE
    from datetime import datetime
    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    end = datetime.strptime(END_DATE, "%Y-%m-%d")
    assert start < end
    assert (end - start).days > 365  # At least 1 year


def test_config_paths_are_path_objects():
    """All directory configs are Path objects."""
    from config import (
        DATA_DIR, TIMESERIES_DIR, MACRO_DIR, NEWS_EN_DIR,
        CANDLESTICK_DIR, ALIGNED_DIR,
    )
    for p in [DATA_DIR, TIMESERIES_DIR, MACRO_DIR, NEWS_EN_DIR, CANDLESTICK_DIR, ALIGNED_DIR]:
        assert isinstance(p, Path)


def test_fred_series_nonempty():
    """FRED_SERIES has expected keys."""
    from config import FRED_SERIES
    assert len(FRED_SERIES) >= 10
    assert "fed_funds_rate" in FRED_SERIES
    assert "dexchus" in FRED_SERIES
    assert "us_cpi" in FRED_SERIES


def test_tech_params_complete():
    """Technical indicator parameters are present."""
    from config import TECH_PARAMS
    required_keys = ["ema_periods", "rsi_period", "macd_fast", "macd_slow",
                     "macd_signal", "atr_period", "bb_period", "bb_std", "adx_period"]
    for key in required_keys:
        assert key in TECH_PARAMS, f"Missing TECH_PARAMS key: {key}"


def test_chart_lookbacks():
    """Chart lookbacks are defined correctly."""
    from config import CHART_LOOKBACKS, CHART_SIZE
    assert "5d" in CHART_LOOKBACKS
    assert "20d" in CHART_LOOKBACKS
    assert "60d" in CHART_LOOKBACKS
    assert CHART_SIZE == (384, 384)


def test_gdelt_keywords_nonempty():
    """GDELT keywords list is non-empty and contains expected terms."""
    from config import GDELT_KEYWORDS
    assert len(GDELT_KEYWORDS) >= 4
    combined = " ".join(GDELT_KEYWORDS).lower()
    assert "yuan" in combined
    assert "pboc" in combined
    assert "fed" in combined
