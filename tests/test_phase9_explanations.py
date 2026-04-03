"""Smoke tests for Phase 9: LLM Explanations"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


@pytest.fixture
def sample_aligned():
    """Create a minimal aligned DataFrame for testing."""
    n = 30
    dates = pd.bdate_range("2025-03-01", periods=n)
    close = 7.0 + np.cumsum(np.random.normal(0, 0.01, n))
    df = pd.DataFrame({
        "close": close,
        "open": close + np.random.normal(0, 0.005, n),
        "high": close + 0.02,
        "low": close - 0.02,
        "volume": 10000,
        "pboc_midprice": close * 0.999,
        "cnh_cny_spread": np.random.normal(0, 0.003, n),
        "tech_ema_5": close,
        "tech_rsi_14": np.random.uniform(30, 70, n),
        "macro_us_fed_funds_rate": 4.5,
        "macro_cn_cn_cpi": 102.0,
        "dxy_close": 100.0,
        "vix_close": 18.0,
        "news_titles": ["Yuan weakens | Fed holds rates"] * n,
        "news_count": 2,
        "target_return": np.random.normal(0, 0.005, n),
        "target_direction": np.random.choice([-1, 0, 1], n),
    }, index=dates)
    return df


class TestBuildPrompt:
    def test_returns_string(self, sample_aligned):
        from phase9_explanations import build_prompt
        row = sample_aligned.iloc[10]
        history = sample_aligned.iloc[5:10]
        prompt = build_prompt(row, history)
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_contains_market_context(self, sample_aligned):
        from phase9_explanations import build_prompt
        row = sample_aligned.iloc[10]
        history = sample_aligned.iloc[5:10]
        prompt = build_prompt(row, history)
        assert "Market Context" in prompt
        assert "USD/CNH Close" in prompt

    def test_contains_task_section(self, sample_aligned):
        from phase9_explanations import build_prompt
        row = sample_aligned.iloc[10]
        prompt = build_prompt(row, sample_aligned.iloc[5:10])
        assert "## Task" in prompt
        assert "BULLISH" in prompt
        assert "BEARISH" in prompt

    def test_includes_macro_data(self, sample_aligned):
        from phase9_explanations import build_prompt
        row = sample_aligned.iloc[10]
        prompt = build_prompt(row, sample_aligned.iloc[5:10])
        assert "Macroeconomic" in prompt

    def test_includes_news(self, sample_aligned):
        from phase9_explanations import build_prompt
        row = sample_aligned.iloc[10]
        prompt = build_prompt(row, sample_aligned.iloc[5:10])
        assert "News Headlines" in prompt

    def test_handles_empty_history(self, sample_aligned):
        from phase9_explanations import build_prompt
        row = sample_aligned.iloc[0]
        empty = pd.DataFrame()
        prompt = build_prompt(row, empty)
        assert isinstance(prompt, str)


class TestGenerateExplanationsOffline:
    def test_generates_weekly_prompts(self, sample_aligned):
        from phase9_explanations import generate_explanations_offline
        records = generate_explanations_offline(sample_aligned)
        assert isinstance(records, list)
        assert len(records) > 0

    def test_record_structure(self, sample_aligned):
        from phase9_explanations import generate_explanations_offline
        records = generate_explanations_offline(sample_aligned)
        for r in records:
            assert "date" in r
            assert "prompt" in r
            assert "actual_direction" in r
            assert "actual_return" in r
            assert "predicted_direction" in r  # None placeholder
            assert "explanation" in r

    def test_only_fridays_or_week_end(self, sample_aligned):
        from phase9_explanations import generate_explanations_offline
        records = generate_explanations_offline(sample_aligned)
        for r in records:
            dt = pd.Timestamp(r["date"])
            # Should be a Friday (4) or the last day in the series
            assert dt.weekday() == 4 or r == records[-1]
