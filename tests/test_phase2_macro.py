"""Smoke tests for Phase 2: Macroeconomic Data"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


class TestFetchFredSeries:
    def test_returns_dataframe_on_success(self):
        from phase2_macro import fetch_fred_series

        mock_fred = MagicMock()
        mock_fred.get_series.return_value = pd.Series(
            [4.5, 4.5, 4.25], index=pd.date_range("2025-01-01", periods=3, freq="MS")
        )
        result = fetch_fred_series(mock_fred, {"fed_funds_rate": "FEDFUNDS"})
        assert isinstance(result, pd.DataFrame)
        assert "fed_funds_rate" in result.columns
        assert len(result) == 3

    def test_returns_empty_on_all_failures(self):
        from phase2_macro import fetch_fred_series

        mock_fred = MagicMock()
        mock_fred.get_series.side_effect = Exception("Network error")
        result = fetch_fred_series(mock_fred, {"test": "TEST"})
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_handles_partial_failure(self):
        from phase2_macro import fetch_fred_series

        mock_fred = MagicMock()
        series_a = pd.Series([1.0, 2.0], index=pd.date_range("2025-01-01", periods=2, freq="MS"))

        def side_effect(series_id, **kwargs):
            if series_id == "A":
                return series_a
            raise Exception("fail")

        mock_fred.get_series.side_effect = side_effect
        result = fetch_fred_series(mock_fred, {"good": "A", "bad": "B"})
        assert "good" in result.columns
        assert "bad" not in result.columns

    def test_skips_empty_series(self):
        from phase2_macro import fetch_fred_series

        mock_fred = MagicMock()
        mock_fred.get_series.return_value = pd.Series([], dtype=float)
        result = fetch_fred_series(mock_fred, {"empty": "EMPTY"})
        assert result.empty


class TestMainExitsWithoutKey:
    def test_exits_without_fred_key(self):
        """main() should exit if FRED_API_KEY is not set."""
        with patch("phase2_macro.FRED_API_KEY", ""):
            with pytest.raises(SystemExit):
                from phase2_macro import main
                main()
