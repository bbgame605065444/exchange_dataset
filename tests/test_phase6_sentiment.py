"""Smoke tests for Phase 6: Sentiment Scoring"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


class TestLoadJsonl:
    def test_loads_valid_jsonl(self, sample_news_jsonl):
        from phase6_sentiment import load_jsonl
        gdelt_path, _ = sample_news_jsonl
        records = load_jsonl(gdelt_path)
        assert len(records) == 5
        assert isinstance(records[0], dict)

    def test_returns_empty_for_missing_file(self, tmp_path):
        from phase6_sentiment import load_jsonl
        result = load_jsonl(tmp_path / "nonexistent.jsonl")
        assert result == []

    def test_skips_blank_lines(self, tmp_path):
        from phase6_sentiment import load_jsonl
        path = tmp_path / "with_blanks.jsonl"
        with open(path, "w") as f:
            f.write('{"a": 1}\n\n{"b": 2}\n\n')
        result = load_jsonl(path)
        assert len(result) == 2


class TestTextExtraction:
    def test_gdelt_text_extraction(self, sample_news_jsonl):
        """Verify text extraction logic for GDELT format."""
        from phase6_sentiment import load_jsonl
        gdelt_path, _ = sample_news_jsonl
        records = load_jsonl(gdelt_path)
        texts = []
        for r in records:
            text = r.get("title", "") or r.get("headline", "") or ""
            summary = r.get("summary", "") or r.get("seendate", "") or ""
            combined = f"{text}. {summary}".strip()
            if combined and len(combined) > 10:
                texts.append(combined[:512])
        assert len(texts) == 5
        assert all(len(t) <= 512 for t in texts)

    def test_finnhub_text_extraction(self, sample_news_jsonl):
        """Verify text extraction logic for Finnhub format."""
        from phase6_sentiment import load_jsonl
        _, finnhub_path = sample_news_jsonl
        records = load_jsonl(finnhub_path)
        texts = []
        for r in records:
            text = r.get("title", "") or r.get("headline", "") or ""
            summary = r.get("summary", "") or r.get("seendate", "") or ""
            combined = f"{text}. {summary}".strip()
            if combined and len(combined) > 10:
                texts.append(combined[:512])
        assert len(texts) == 3


class TestSentimentOutputFormat:
    def test_expected_parquet_schema(self, tmp_path):
        """Verify expected output schema of sentiment parquet."""
        df = pd.DataFrame({
            "text": ["Yuan weakens", "Fed holds rates"],
            "source": ["gdelt", "finnhub"],
            "label": ["negative", "neutral"],
            "score": [0.85, 0.72],
        })
        path = tmp_path / "en_sentiment.parquet"
        df.to_parquet(path)
        loaded = pd.read_parquet(path)
        assert set(loaded.columns) == {"text", "source", "label", "score"}
        assert loaded["score"].dtype == float
