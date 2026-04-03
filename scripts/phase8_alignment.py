"""
Phase 8: Data Alignment & Assembly
Combines all modalities into a single daily-aligned dataset.
Output: aligned/USDCNH_daily_aligned.parquet
"""
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import (
    TIMESERIES_DIR, MACRO_DIR, NEWS_EN_DIR, SENTIMENT_DIR,
    CANDLESTICK_DIR, ALIGNED_DIR, CHART_LOOKBACKS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_parquet_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.warning(f"Not found: {path}")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    logger.info(f"Loaded {path.name}: {df.shape}")
    return df


def load_news_by_date(news_dir: Path) -> dict:
    """Load news JSONL files and group by date."""
    news_by_date = {}
    for jsonl_file in news_dir.glob("*.jsonl"):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                # Try to extract date
                date_str = (
                    record.get("seendate", "") or
                    record.get("datetime", "") or
                    record.get("date", "") or
                    ""
                )
                if not date_str:
                    continue
                try:
                    dt = pd.to_datetime(date_str)
                    date_key = dt.strftime("%Y-%m-%d")
                except Exception:
                    continue

                title = record.get("title", "") or record.get("headline", "") or ""
                if title:
                    news_by_date.setdefault(date_key, []).append(title)

    return news_by_date


def main():
    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)

    # === Load OHLCV (anchor) ===
    ohlcv = load_parquet_safe(TIMESERIES_DIR / "USDCNH_daily_ohlcv.parquet")
    if ohlcv.empty:
        logger.error("No OHLCV data. Run Phase 1 first.")
        sys.exit(1)

    ohlcv.index = pd.to_datetime(ohlcv.index)
    ohlcv.index.name = "date"

    # Start with OHLCV as the base
    aligned = ohlcv[["Open", "High", "Low", "Close", "Volume"]].copy()
    aligned.columns = ["open", "high", "low", "close", "volume"]

    # === CNH-CNY Spread ===
    spread = load_parquet_safe(TIMESERIES_DIR / "cnh_cny_spread.parquet")
    if not spread.empty:
        spread.index = pd.to_datetime(spread.index)
        if "cnh_cny_spread" in spread.columns:
            aligned["cnh_cny_spread"] = spread["cnh_cny_spread"].reindex(aligned.index)

    # === PBOC Midprice Proxy ===
    pboc = load_parquet_safe(TIMESERIES_DIR / "pboc_midprice.parquet")
    if not pboc.empty:
        pboc.index = pd.to_datetime(pboc.index)
        col = pboc.columns[0]
        aligned["pboc_midprice"] = pboc[col].reindex(aligned.index, method="ffill")

    # === Technical Indicators ===
    tech = load_parquet_safe(TIMESERIES_DIR / "technical_indicators.parquet")
    if not tech.empty:
        tech.index = pd.to_datetime(tech.index)
        for col in tech.columns:
            aligned[f"tech_{col}"] = tech[col].reindex(aligned.index)

    # === Auxiliary (DXY, VIX) ===
    aux = load_parquet_safe(TIMESERIES_DIR / "auxiliary_indicators.parquet")
    if not aux.empty:
        aux.index = pd.to_datetime(aux.index)
        for col in aux.columns:
            aligned[col] = aux[col].reindex(aligned.index, method="ffill")

    # === US Macro (forward-fill to daily) ===
    us_macro = load_parquet_safe(MACRO_DIR / "us_macro.parquet")
    if not us_macro.empty:
        us_macro.index = pd.to_datetime(us_macro.index)
        for col in us_macro.columns:
            aligned[f"macro_{col}"] = us_macro[col].reindex(aligned.index, method="ffill")

    # === CN Macro (forward-fill to daily) ===
    cn_macro = load_parquet_safe(MACRO_DIR / "cn_macro_fred.parquet")
    if not cn_macro.empty:
        cn_macro.index = pd.to_datetime(cn_macro.index)
        for col in cn_macro.columns:
            aligned[f"macro_{col}"] = cn_macro[col].reindex(aligned.index, method="ffill")

    # === News texts (group by date) ===
    news_by_date = load_news_by_date(NEWS_EN_DIR)
    aligned["news_count"] = 0
    aligned["news_texts"] = ""
    for date_str, titles in news_by_date.items():
        try:
            dt = pd.to_datetime(date_str)
            if dt in aligned.index:
                aligned.loc[dt, "news_count"] = len(titles)
                aligned.loc[dt, "news_texts"] = " | ".join(titles[:10])  # Top 10
        except Exception:
            pass

    # === Sentiment scores ===
    en_sent = load_parquet_safe(SENTIMENT_DIR / "en_sentiment.parquet")
    if not en_sent.empty and "label" in en_sent.columns:
        # Aggregate: compute daily positive ratio
        sent_summary = en_sent.groupby("label")["score"].mean()
        aligned["sentiment_positive_ratio"] = sent_summary.get("positive", 0)
        aligned["sentiment_negative_ratio"] = sent_summary.get("negative", 0)

    # === Chart image paths ===
    for label in CHART_LOOKBACKS:
        aligned[f"chart_{label}_path"] = ""
        for dt in aligned.index:
            date_str = dt.strftime("%Y%m%d")
            fname = f"USDCNH_{date_str}_{label}.png"
            fpath = CANDLESTICK_DIR / fname
            if fpath.exists():
                aligned.loc[dt, f"chart_{label}_path"] = str(fpath)

    # === Compute targets ===
    aligned["target_return"] = aligned["close"].pct_change().shift(-1)
    aligned["target_direction"] = np.sign(aligned["target_return"])
    # Map to int: 1 (up), -1 (down), 0 (flat)
    aligned["target_direction"] = aligned["target_direction"].fillna(0).astype(int)

    # === Save ===
    output_path = ALIGNED_DIR / "USDCNH_daily_aligned.parquet"
    aligned.to_parquet(output_path)

    logger.info("=" * 60)
    logger.info("Phase 8 Complete!")
    logger.info(f"  Aligned dataset: {aligned.shape}")
    logger.info(f"  Date range: {aligned.index[0].date()} to {aligned.index[-1].date()}")
    logger.info(f"  Columns: {list(aligned.columns)}")
    logger.info(f"  Target distribution: {aligned['target_direction'].value_counts().to_dict()}")
    logger.info(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
