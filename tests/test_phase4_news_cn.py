"""Smoke tests for Phase 4: Chinese News Collection"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


class TestForexKeywordFilter:
    def test_detects_forex_related(self):
        from phase4_news_cn import _is_forex_related
        assert _is_forex_related("人民币兑美元中间价报7.1234")
        assert _is_forex_related("离岸人民币贬值破7.2关口")
        assert _is_forex_related("央行外汇储备增加200亿")
        assert _is_forex_related("CNH drops to 3-month low")

    def test_rejects_unrelated(self):
        from phase4_news_cn import _is_forex_related
        assert not _is_forex_related("今日天气晴朗适合出行")
        assert not _is_forex_related("科技公司发布新手机")


class TestParseCnDate:
    def test_standard_format(self):
        from phase4_news_cn import _parse_cn_date
        dt = _parse_cn_date("2025-03-15 14:30:00")
        assert dt is not None
        assert dt.year == 2025
        assert dt.month == 3
        assert dt.day == 15

    def test_date_only(self):
        from phase4_news_cn import _parse_cn_date
        dt = _parse_cn_date("2025-03-15")
        assert dt is not None

    def test_invalid_returns_none(self):
        from phase4_news_cn import _parse_cn_date
        assert _parse_cn_date("invalid") is None
        assert _parse_cn_date("") is None


class TestSaveJsonl:
    def test_saves_and_reads(self, tmp_path):
        import json
        from phase4_news_cn import _save_jsonl
        articles = [
            {"title": "人民币升值", "source": "test"},
            {"title": "美元走弱", "source": "test"},
        ]
        path = tmp_path / "test.jsonl"
        _save_jsonl(articles, path)
        assert path.exists()
        with open(path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 2
        assert lines[0]["title"] == "人民币升值"


class TestGaussianSleep:
    def test_doesnt_crash(self):
        from phase4_news_cn import gaussian_sleep
        import time
        start = time.time()
        gaussian_sleep(0.1, 0.01)
        elapsed = time.time() - start
        assert elapsed >= 0.05  # at least some sleep happened
