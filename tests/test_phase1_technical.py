"""Smoke tests for Phase 1: Price Data + Technical Indicators"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


class TestEMA:
    def test_ema_returns_series(self, sample_ohlcv):
        from phase1_price_technical import _ema
        result = _ema(sample_ohlcv["Close"], 5)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv)

    def test_ema_no_nan_after_warmup(self, sample_ohlcv):
        from phase1_price_technical import _ema
        result = _ema(sample_ohlcv["Close"], 5)
        assert not result.iloc[5:].isna().any()

    def test_ema_different_periods(self, sample_ohlcv):
        from phase1_price_technical import _ema
        ema5 = _ema(sample_ohlcv["Close"], 5)
        ema50 = _ema(sample_ohlcv["Close"], 50)
        # Shorter EMA should be more responsive (different values)
        assert not ema5.equals(ema50)


class TestRSI:
    def test_rsi_range(self, sample_ohlcv):
        from phase1_price_technical import _rsi
        result = _rsi(sample_ohlcv["Close"], 14)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_returns_series(self, sample_ohlcv):
        from phase1_price_technical import _rsi
        result = _rsi(sample_ohlcv["Close"], 14)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv)


class TestATR:
    def test_atr_positive(self, sample_ohlcv):
        from phase1_price_technical import _atr
        result = _atr(sample_ohlcv["High"], sample_ohlcv["Low"], sample_ohlcv["Close"], 14)
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_atr_returns_series(self, sample_ohlcv):
        from phase1_price_technical import _atr
        result = _atr(sample_ohlcv["High"], sample_ohlcv["Low"], sample_ohlcv["Close"], 14)
        assert isinstance(result, pd.Series)


class TestComputeTechnicalIndicators:
    def test_output_shape(self, sample_ohlcv):
        from phase1_price_technical import compute_technical_indicators
        result = compute_technical_indicators(sample_ohlcv)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv)

    def test_expected_columns(self, sample_ohlcv):
        from phase1_price_technical import compute_technical_indicators
        result = compute_technical_indicators(sample_ohlcv)
        expected = ["ema_5", "ema_20", "ema_50", "rsi_14", "macd", "macd_signal",
                    "macd_hist", "atr_14", "bb_upper", "bb_middle", "bb_lower",
                    "adx", "plus_di", "minus_di", "obv"]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_column_count(self, sample_ohlcv):
        from phase1_price_technical import compute_technical_indicators
        result = compute_technical_indicators(sample_ohlcv)
        assert result.shape[1] == 15

    def test_bollinger_bands_ordering(self, sample_ohlcv):
        from phase1_price_technical import compute_technical_indicators
        result = compute_technical_indicators(sample_ohlcv)
        valid = result.dropna()
        assert (valid["bb_upper"] >= valid["bb_middle"]).all()
        assert (valid["bb_middle"] >= valid["bb_lower"]).all()

    def test_macd_hist_equals_diff(self, sample_ohlcv):
        from phase1_price_technical import compute_technical_indicators
        result = compute_technical_indicators(sample_ohlcv)
        diff = result["macd"] - result["macd_signal"]
        pd.testing.assert_series_equal(result["macd_hist"], diff, check_names=False)

    def test_obv_is_cumulative(self, sample_ohlcv):
        from phase1_price_technical import compute_technical_indicators
        result = compute_technical_indicators(sample_ohlcv)
        # OBV should change by volume each step (not be constant)
        assert result["obv"].nunique() > 1


class TestDownloadTicker:
    def test_download_returns_dataframe(self):
        from phase1_price_technical import download_ticker
        # Will fail due to network, but should return empty DataFrame gracefully
        result = download_ticker("INVALID_TICKER_XYZ", "test")
        assert isinstance(result, pd.DataFrame)
