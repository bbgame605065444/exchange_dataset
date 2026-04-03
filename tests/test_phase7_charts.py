"""Smoke tests for Phase 7: K-line Chart Generation"""
import sys
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


class TestChartGeneration:
    def test_generates_png_files(self, sample_ohlcv, tmp_path):
        """Charts are generated as PNG files."""
        import phase7_charts as mod
        out = tmp_path / "candlestick"
        out.mkdir()

        small = sample_ohlcv.iloc[:15]
        with patch.object(mod, "CANDLESTICK_DIR", out):
            mod.generate_candlestick_charts(small)

        pngs = list(out.glob("*.png"))
        assert len(pngs) > 0

    def test_chart_filename_format(self, sample_ohlcv, tmp_path):
        """Chart filenames follow USDCNH_{date}_{period}.png convention."""
        import phase7_charts as mod
        out = tmp_path / "candlestick"
        out.mkdir()

        small = sample_ohlcv.iloc[:10]
        with patch.object(mod, "CANDLESTICK_DIR", out):
            mod.generate_candlestick_charts(small)

        pngs = list(out.glob("*.png"))
        for png in pngs:
            name = png.stem
            assert name.startswith("USDCNH_")
            parts = name.split("_")
            assert len(parts) == 3
            assert parts[2] in ("5d", "20d", "60d")
            assert len(parts[1]) == 8  # YYYYMMDD
            assert parts[1].isdigit()

    def test_skips_insufficient_data(self, tmp_path):
        """Should skip if less than 3 data points."""
        import phase7_charts as mod
        out = tmp_path / "candlestick"
        out.mkdir()

        tiny = pd.DataFrame({
            "Open": [7.0, 7.1],
            "High": [7.1, 7.2],
            "Low": [6.9, 7.0],
            "Close": [7.05, 7.15],
            "Volume": [1000, 1000],
        }, index=pd.bdate_range("2025-01-01", periods=2))

        with patch.object(mod, "CANDLESTICK_DIR", out):
            mod.generate_candlestick_charts(tiny)

        pngs = list(out.glob("*.png"))
        assert len(pngs) == 0

    def test_handles_zero_volume(self, tmp_path):
        """Should handle zero/NaN volume gracefully."""
        import phase7_charts as mod
        out = tmp_path / "candlestick"
        out.mkdir()

        n = 10
        df = pd.DataFrame({
            "Open": np.linspace(7.0, 7.1, n),
            "High": np.linspace(7.05, 7.15, n),
            "Low": np.linspace(6.95, 7.05, n),
            "Close": np.linspace(7.02, 7.12, n),
            "Volume": [0] * n,
        }, index=pd.bdate_range("2025-01-01", periods=n))

        with patch.object(mod, "CANDLESTICK_DIR", out):
            mod.generate_candlestick_charts(df)

        pngs = list(out.glob("*.png"))
        assert len(pngs) > 0
