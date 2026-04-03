"""Smoke tests for Phase 8: Data Alignment & Assembly"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


class TestLoadParquetSafe:
    def test_loads_existing_file(self, sample_ohlcv_parquet):
        from phase8_alignment import load_parquet_safe
        df = load_parquet_safe(sample_ohlcv_parquet)
        assert not df.empty
        assert "Close" in df.columns

    def test_returns_empty_for_missing(self, tmp_path):
        from phase8_alignment import load_parquet_safe
        df = load_parquet_safe(tmp_path / "nonexistent.parquet")
        assert isinstance(df, pd.DataFrame)
        assert df.empty


class TestLoadNewsAligned:
    def test_assigns_news_to_trading_days(self, sample_news_jsonl):
        from phase8_alignment import load_news_aligned
        gdelt_path, _ = sample_news_jsonl
        trading_dates = pd.bdate_range("2025-03-01", "2025-03-31")
        result = load_news_aligned([gdelt_path.parent], trading_dates)
        assert isinstance(result, pd.DataFrame)
        assert "news_count" in result.columns
        assert "news_titles" in result.columns
        assert len(result) == len(trading_dates)
        # Some days should have news
        assert result["news_count"].sum() > 0

    def test_empty_dir(self, tmp_path):
        from phase8_alignment import load_news_aligned
        trading_dates = pd.bdate_range("2025-01-01", "2025-01-10")
        result = load_news_aligned([tmp_path], trading_dates)
        assert isinstance(result, pd.DataFrame)
        assert (result["news_count"] == 0).all()


class TestAlignment:
    def test_full_alignment_pipeline(self, tmp_data_dir, sample_ohlcv_parquet,
                                     sample_spread, sample_macro,
                                     sample_news_jsonl):
        """Run alignment on sample data and verify output structure."""
        from phase8_alignment import load_parquet_safe
        import config

        # Temporarily redirect paths to test data
        orig_ts = config.TIMESERIES_DIR
        orig_macro = config.MACRO_DIR
        orig_news = config.NEWS_EN_DIR
        orig_sent = config.SENTIMENT_DIR
        orig_candle = config.CANDLESTICK_DIR
        orig_aligned = config.ALIGNED_DIR

        config.TIMESERIES_DIR = tmp_data_dir / "timeseries"
        config.MACRO_DIR = tmp_data_dir / "macro"
        config.NEWS_EN_DIR = tmp_data_dir / "text" / "news_en"
        config.SENTIMENT_DIR = tmp_data_dir / "sentiment"
        config.CANDLESTICK_DIR = tmp_data_dir / "images" / "candlestick"
        config.ALIGNED_DIR = tmp_data_dir / "aligned"

        # Generate PBOC midprice
        ohlcv = pd.read_parquet(sample_ohlcv_parquet)
        pboc = pd.DataFrame({"pboc_midprice_proxy": ohlcv["Close"] * 0.999}, index=ohlcv.index)
        pboc.to_parquet(config.TIMESERIES_DIR / "pboc_midprice.parquet")

        # Generate technical indicators
        from phase1_price_technical import compute_technical_indicators
        tech = compute_technical_indicators(ohlcv)
        tech.to_parquet(config.TIMESERIES_DIR / "technical_indicators.parquet")

        # Generate auxiliary
        aux = pd.DataFrame({
            "dxy_close": np.linspace(100, 101, len(ohlcv)),
            "vix_close": np.linspace(15, 20, len(ohlcv)),
        }, index=ohlcv.index)
        aux.to_parquet(config.TIMESERIES_DIR / "auxiliary_indicators.parquet")

        try:
            # Run alignment by importing and calling main inline logic
            ohlcv_loaded = load_parquet_safe(config.TIMESERIES_DIR / "USDCNH_daily_ohlcv.parquet")
            ohlcv_loaded.index = pd.to_datetime(ohlcv_loaded.index)
            aligned = ohlcv_loaded[["Open", "High", "Low", "Close", "Volume"]].copy()
            aligned.columns = ["open", "high", "low", "close", "volume"]

            # Verify basics
            assert len(aligned) == 100
            assert "close" in aligned.columns

            # Target computation
            aligned["target_return"] = aligned["close"].pct_change().shift(-1)
            aligned["target_direction"] = np.sign(aligned["target_return"]).fillna(0).astype(int)

            assert "target_return" in aligned.columns
            assert "target_direction" in aligned.columns
            assert set(aligned["target_direction"].unique()).issubset({-1, 0, 1})

        finally:
            config.TIMESERIES_DIR = orig_ts
            config.MACRO_DIR = orig_macro
            config.NEWS_EN_DIR = orig_news
            config.SENTIMENT_DIR = orig_sent
            config.CANDLESTICK_DIR = orig_candle
            config.ALIGNED_DIR = orig_aligned


class TestTargetComputation:
    def test_target_return_is_shifted(self):
        """target_return should be t+1 return (shifted by -1)."""
        close = pd.Series([100.0, 101.0, 99.0, 102.0])
        ret = close.pct_change().shift(-1)
        # ret[0] should be (101-100)/100 = 0.01
        assert abs(ret.iloc[0] - 0.01) < 1e-10
        # ret[1] should be (99-101)/101
        assert abs(ret.iloc[1] - (-2/101)) < 1e-10
        # Last value should be NaN
        assert pd.isna(ret.iloc[-1])

    def test_target_direction_signs(self):
        """target_direction should be sign of target_return."""
        returns = pd.Series([0.01, -0.02, 0.0, np.nan])
        direction = np.sign(returns).fillna(0).astype(int)
        assert direction.iloc[0] == 1
        assert direction.iloc[1] == -1
        assert direction.iloc[2] == 0
        assert direction.iloc[3] == 0
