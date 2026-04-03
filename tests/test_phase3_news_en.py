"""Smoke tests for Phase 3: English News Collection"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


class TestGdeltDeduplication:
    def test_url_dedup_logic(self):
        """Verify the deduplication logic used in collect_gdelt_recent."""
        articles = [
            {"url": "https://a.com/1", "title": "Article 1"},
            {"url": "https://a.com/1", "title": "Article 1 dup"},
            {"url": "https://b.com/2", "title": "Article 2"},
            {"url": "", "title": "No URL"},
        ]
        seen_urls = set()
        unique = []
        for a in articles:
            url = a.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique.append(a)
        assert len(unique) == 2
        assert unique[0]["title"] == "Article 1"
        assert unique[1]["title"] == "Article 2"


class TestFinnhubKeywordFilter:
    def test_keyword_filtering(self):
        """Test the keyword filter used in collect_finnhub_news."""
        keywords = ["yuan", "cnh", "rmb", "pboc", "china trade", "fed rate",
                    "dollar", "forex", "currency", "exchange rate"]
        news = [
            {"headline": "Yuan weakens against dollar", "summary": ""},
            {"headline": "Apple reports earnings", "summary": "Tech stock rises"},
            {"headline": "PBOC sets daily fixing", "summary": "Central bank action"},
            {"headline": "Weather forecast", "summary": "Sunny day"},
        ]
        filtered = [
            n for n in news
            if any(kw in (n.get("headline", "") + " " + n.get("summary", "")).lower()
                   for kw in keywords)
        ]
        assert len(filtered) == 2
        assert "Yuan" in filtered[0]["headline"]
        assert "PBOC" in filtered[1]["headline"]


class TestNewsJsonlFormat:
    def test_gdelt_jsonl_structure(self, sample_news_jsonl):
        """GDELT JSONL has expected fields."""
        gdelt_path, _ = sample_news_jsonl
        with open(gdelt_path) as f:
            record = json.loads(f.readline())
        assert "url" in record
        assert "title" in record
        assert "seendate" in record
        assert "source" in record

    def test_finnhub_jsonl_structure(self, sample_news_jsonl):
        """Finnhub JSONL has expected fields."""
        _, finnhub_path = sample_news_jsonl
        with open(finnhub_path) as f:
            record = json.loads(f.readline())
        assert "id" in record
        assert "headline" in record
        assert "summary" in record
        assert "category" in record

    def test_gdelt_line_count(self, sample_news_jsonl):
        gdelt_path, _ = sample_news_jsonl
        with open(gdelt_path) as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 5

    def test_finnhub_line_count(self, sample_news_jsonl):
        _, finnhub_path = sample_news_jsonl
        with open(finnhub_path) as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 3
