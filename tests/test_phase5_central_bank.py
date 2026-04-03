"""Smoke tests for Phase 5: Central Bank Communications"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


class TestCentralBankJsonl:
    def test_fomc_labeled_structure(self, sample_central_bank_jsonl):
        fomc_path, _ = sample_central_bank_jsonl
        with open(fomc_path) as f:
            records = [json.loads(line) for line in f if line.strip()]
        assert len(records) == 3
        for r in records:
            assert "sentence" in r
            assert "label" in r
            assert r["label"] in ("hawkish", "dovish", "neutral")
            assert "split" in r

    def test_fed_statements_structure(self, sample_central_bank_jsonl):
        _, fed_path = sample_central_bank_jsonl
        with open(fed_path) as f:
            records = [json.loads(line) for line in f if line.strip()]
        assert len(records) >= 1
        for r in records:
            assert "date" in r
            assert "url" in r
            assert "text" in r
            assert len(r["text"]) > 20
            assert r["type"] == "fomc_statement"

    def test_fomc_date_format(self, sample_central_bank_jsonl):
        _, fed_path = sample_central_bank_jsonl
        with open(fed_path) as f:
            record = json.loads(f.readline())
        date_str = record["date"]
        assert len(date_str) == 8
        assert date_str.isdigit()


class TestFedStatementDateExtraction:
    def test_regex_date_extraction(self):
        """Test date extraction regex used in collect_fed_statements."""
        import re
        urls = [
            "/monetarypolicy/fomcpresconf20250129.htm",
            "/monetarypolicy/fomcpresconf20250319.htm",
            "/monetary/nodate.htm",
        ]
        dates = []
        for url in urls:
            match = re.search(r"(\d{8})", url)
            if match:
                dates.append(match.group(1))
        assert dates == ["20250129", "20250319"]


class TestEconomicCalendarFilter:
    def test_country_filter(self):
        """Test the US/CN country filter logic."""
        events = [
            {"country": "US", "event": "FOMC"},
            {"country": "CN", "event": "PBOC rate"},
            {"country": "JP", "event": "BOJ meeting"},
            {"country": "", "event": "Unknown"},
            {"country": "EU", "event": "ECB"},
        ]
        filtered = [e for e in events if e.get("country", "") in ("US", "CN", "")]
        assert len(filtered) == 3
        countries = [e["country"] for e in filtered]
        assert "JP" not in countries
        assert "EU" not in countries
