"""
Phase 8: Data Alignment & Assembly
Combines all modalities into a single daily-aligned dataset.
Output: aligned/USDCNH_daily_aligned.parquet

Timestamp alignment rules (NO look-ahead):
  - Anchor: UTC business day index from OHLCV
  - Macro: forward-fill (only past-published values reach day t)
  - News: window [t-1 17:00 UTC, t 17:00 UTC) assigned to day t
  - Charts: built from data up to and including day t
  - Target: t+1 return/direction (future label, not a feature)
"""
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import (
    TIMESERIES_DIR, MACRO_DIR, NEWS_EN_DIR, NEWS_CN_DIR, SENTIMENT_DIR,
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


def _parse_news_ts(raw) -> pd.Timestamp | None:
    """Parse a news timestamp into a tz-aware UTC Timestamp."""
    if raw is None or raw == "":
        return None
    try:
        # Unix epoch seconds (Finnhub format)
        if isinstance(raw, (int, float)) and raw > 1e9:
            return pd.Timestamp(raw, unit="s", tz="UTC")
        ts = pd.to_datetime(raw, utc=True)
        return ts
    except Exception:
        try:
            ts = pd.to_datetime(raw)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            return ts
        except Exception:
            return None


def load_news_aligned(news_dirs: list[Path], trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Load news from JSONL, assign each article to the correct trading day
    using the window [t-1 17:00 UTC, t 17:00 UTC), and aggregate per day.
    """
    # Build cutoff windows: for each trading day t, news window is
    # (prev_close_cutoff, this_close_cutoff] where cutoff = date 17:00 UTC
    cutoff_hour = 17  # 17:00 UTC ≈ NYSE close, also reasonable for CNH

    records = []
    for news_dir in news_dirs:
        if not news_dir.exists():
            continue
        for jsonl_file in news_dir.glob("*.jsonl"):
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    raw_ts = (
                        rec.get("seendate")
                        or rec.get("datetime")
                        or rec.get("date")
                        or rec.get("published")
                    )
                    ts = _parse_news_ts(raw_ts)
                    if ts is None:
                        continue

                    # Normalise to naive-UTC for comparison with trading_dates
                    if ts.tzinfo is not None:
                        ts = ts.tz_convert("UTC").tz_localize(None)

                    title = rec.get("title") or rec.get("headline") or ""
                    tone = rec.get("tone")
                    if isinstance(tone, str):
                        try:
                            tone = float(tone)
                        except ValueError:
                            tone = None
                    records.append({
                        "ts_utc": ts,
                        "title": title,
                        "tone": tone,
                    })

    if not records:
        logger.warning("No news records found")
        return pd.DataFrame(
            {"news_count": 0, "news_titles": "", "news_avg_tone": np.nan},
            index=trading_dates,
        )

    news_df = pd.DataFrame(records)
    news_df = news_df.sort_values("ts_utc")

    # Assign each article to a trading day via the cutoff window
    # For trading day t: window = (t-1 17:00, t 17:00]
    # Articles outside any window are assigned to the nearest next trading day
    trading_dates_sorted = trading_dates.sort_values()

    # Build cutoff series: each trading day's upper cutoff (naive-UTC)
    cutoffs = pd.Series(
        [pd.Timestamp(d.year, d.month, d.day, cutoff_hour)
         for d in trading_dates_sorted],
        index=trading_dates_sorted,
    )

    assigned_day = []
    cutoff_vals = cutoffs.values
    tday_vals = cutoffs.index

    for ts_utc in news_df["ts_utc"]:
        # Find the first cutoff >= ts_utc
        idx = np.searchsorted(cutoff_vals, ts_utc, side="left")
        if idx < len(tday_vals):
            assigned_day.append(tday_vals[idx])
        else:
            assigned_day.append(tday_vals[-1])  # Assign to last day

    news_df["trading_day"] = assigned_day

    # Aggregate per trading day
    agg = news_df.groupby("trading_day").agg(
        news_count=("title", "count"),
        news_titles=("title", lambda x: " | ".join(t for t in x if t)[:2000]),
        news_avg_tone=("tone", "mean"),
    )

    # Reindex to full trading calendar
    result = agg.reindex(trading_dates)
    result["news_count"] = result["news_count"].fillna(0).astype(int)
    result["news_titles"] = result["news_titles"].fillna("")
    return result


def main():
    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # 1. Load OHLCV anchor
    # ================================================================
    ohlcv = load_parquet_safe(TIMESERIES_DIR / "USDCNH_daily_ohlcv.parquet")
    if ohlcv.empty:
        logger.error("No OHLCV data. Run Phase 1 / collect_all.py first.")
        sys.exit(1)

    ohlcv.index = pd.to_datetime(ohlcv.index)
    ohlcv.index.name = "date"
    # Remove timezone if present (work in naive-UTC internally)
    if ohlcv.index.tz is not None:
        ohlcv.index = ohlcv.index.tz_localize(None)

    aligned = ohlcv[["Open", "High", "Low", "Close", "Volume"]].copy()
    aligned.columns = ["open", "high", "low", "close", "volume"]
    trading_dates = aligned.index

    logger.info(f"Anchor: {len(trading_dates)} trading days, "
                f"{trading_dates[0].date()} to {trading_dates[-1].date()}")

    # ================================================================
    # 2. CNH-CNY spread (same-day, no alignment issue)
    # ================================================================
    spread = load_parquet_safe(TIMESERIES_DIR / "cnh_cny_spread.parquet")
    if not spread.empty:
        spread.index = pd.to_datetime(spread.index)
        if spread.index.tz is not None:
            spread.index = spread.index.tz_localize(None)
        if "cnh_cny_spread" in spread.columns:
            aligned["cnh_cny_spread"] = spread["cnh_cny_spread"].reindex(trading_dates)

    # ================================================================
    # 3. PBOC midprice proxy (daily, forward-fill for holidays)
    # ================================================================
    pboc = load_parquet_safe(TIMESERIES_DIR / "pboc_midprice.parquet")
    if not pboc.empty:
        pboc.index = pd.to_datetime(pboc.index)
        if pboc.index.tz is not None:
            pboc.index = pboc.index.tz_localize(None)
        col = pboc.columns[0]
        aligned["pboc_midprice"] = pboc[col].reindex(trading_dates, method="ffill")

    # ================================================================
    # 4. Technical indicators (same-day, computed from OHLCV up to t)
    # ================================================================
    tech = load_parquet_safe(TIMESERIES_DIR / "technical_indicators.parquet")
    if not tech.empty:
        tech.index = pd.to_datetime(tech.index)
        if tech.index.tz is not None:
            tech.index = tech.index.tz_localize(None)
        for col in tech.columns:
            aligned[f"tech_{col}"] = tech[col].reindex(trading_dates)

    # ================================================================
    # 5. Auxiliary (DXY, VIX) — same-day, ffill for missing
    # ================================================================
    aux = load_parquet_safe(TIMESERIES_DIR / "auxiliary_indicators.parquet")
    if not aux.empty:
        aux.index = pd.to_datetime(aux.index)
        if aux.index.tz is not None:
            aux.index = aux.index.tz_localize(None)
        for col in aux.columns:
            aligned[col] = aux[col].reindex(trading_dates, method="ffill")

    # ================================================================
    # 6. Macro data (monthly/weekly → daily via forward-fill)
    #    ffill ensures only past-published values are used on day t
    # ================================================================
    for fname, prefix in [
        ("us_macro.parquet", "macro_us"),
        ("cn_macro_fred.parquet", "macro_cn"),
    ]:
        macro = load_parquet_safe(MACRO_DIR / fname)
        if not macro.empty:
            macro.index = pd.to_datetime(macro.index)
            if macro.index.tz is not None:
                macro.index = macro.index.tz_localize(None)
            for col in macro.columns:
                aligned[f"{prefix}_{col}"] = macro[col].reindex(trading_dates, method="ffill")

    # ================================================================
    # 7. News — proper window-based alignment
    # ================================================================
    news_dirs = [NEWS_EN_DIR]
    if NEWS_CN_DIR.exists():
        news_dirs.append(NEWS_CN_DIR)
    news_agg = load_news_aligned(news_dirs, trading_dates)
    aligned["news_count"] = news_agg["news_count"]
    aligned["news_titles"] = news_agg["news_titles"]
    aligned["news_avg_tone"] = news_agg["news_avg_tone"]

    # ================================================================
    # 8. Chart image paths (relative, only if file exists)
    # ================================================================
    for label in CHART_LOOKBACKS:
        paths = []
        for dt in trading_dates:
            fname = f"USDCNH_{dt.strftime('%Y%m%d')}_{label}.png"
            fpath = CANDLESTICK_DIR / fname
            paths.append(f"images/candlestick/{fname}" if fpath.exists() else "")
        aligned[f"chart_{label}_path"] = paths

    # ================================================================
    # 9. Targets — t+1 return & direction (this IS future data, labelled)
    # ================================================================
    aligned["target_return"] = aligned["close"].pct_change().shift(-1)
    aligned["target_direction"] = np.sign(aligned["target_return"])
    aligned["target_direction"] = aligned["target_direction"].fillna(0).astype(int)

    # ================================================================
    # 10. Data quality report
    # ================================================================
    n_total = len(aligned)
    n_with_news = (aligned["news_count"] > 0).sum()
    n_with_macro = aligned[[c for c in aligned.columns if c.startswith("macro_")]].notna().any(axis=1).sum()
    n_with_tech = aligned[[c for c in aligned.columns if c.startswith("tech_")]].notna().any(axis=1).sum()
    n_charts = sum(1 for label in CHART_LOOKBACKS
                   if (aligned[f"chart_{label}_path"] != "").any())

    # ================================================================
    # 11. Save
    # ================================================================
    output_path = ALIGNED_DIR / "USDCNH_daily_aligned.parquet"
    aligned.to_parquet(output_path)

    logger.info("=" * 60)
    logger.info("Phase 8 — Alignment Complete")
    logger.info(f"  Shape:           {aligned.shape}")
    logger.info(f"  Date range:      {trading_dates[0].date()} → {trading_dates[-1].date()}")
    logger.info(f"  Trading days:    {n_total}")
    logger.info(f"  Days with news:  {n_with_news}")
    logger.info(f"  Days with macro: {n_with_macro}")
    logger.info(f"  Days with tech:  {n_with_tech}")
    logger.info(f"  Chart modalities: {n_charts}/3")
    logger.info(f"  Target dist:     {aligned['target_direction'].value_counts().to_dict()}")
    logger.info(f"  Columns ({len(aligned.columns)}):")
    for c in aligned.columns:
        non_null = aligned[c].notna().sum() if aligned[c].dtype != object else (aligned[c] != "").sum()
        logger.info(f"    {c:<35s} {non_null:>5d}/{n_total} non-null")
    logger.info(f"  Saved → {output_path}")


if __name__ == "__main__":
    main()
