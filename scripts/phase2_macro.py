"""
Phase 2: Macroeconomic Data
Sources: FRED API
Output: macro/us_macro.parquet, macro/cn_macro_fred.parquet
"""
import sys
import logging

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import pandas as pd

from config import START_DATE, END_DATE, FRED_API_KEY, FRED_SERIES, MACRO_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def fetch_fred_series(fred, series_dict: dict) -> pd.DataFrame:
    """Fetch multiple FRED series and combine into a DataFrame."""
    frames = {}
    for name, series_id in series_dict.items():
        try:
            logger.info(f"  Fetching {name} ({series_id})...")
            s = fred.get_series(series_id, observation_start=START_DATE, observation_end=END_DATE)
            if s is not None and len(s) > 0:
                frames[name] = s
                logger.info(f"    -> {len(s)} observations")
            else:
                logger.warning(f"    -> No data for {series_id}")
        except Exception as e:
            logger.warning(f"    -> Error fetching {series_id}: {e}")
    if frames:
        return pd.DataFrame(frames)
    return pd.DataFrame()


def main():
    if not FRED_API_KEY:
        logger.warning(
            "FRED_API_KEY not set — skipping Phase 2.\n"
            "  export FRED_API_KEY='your_key_here'\n"
            "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
        )
        return

    try:
        from fredapi import Fred
    except ImportError:
        logger.error("fredapi not installed. Run: pip install fredapi")
        sys.exit(1)

    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    fred = Fred(api_key=FRED_API_KEY)

    # === US Macro Indicators ===
    us_series = {k: v for k, v in FRED_SERIES.items() if k.startswith("us_")}
    us_series["fed_funds_rate"] = FRED_SERIES["fed_funds_rate"]
    logger.info(f"Fetching {len(us_series)} US macro series...")
    us_macro = fetch_fred_series(fred, us_series)
    if not us_macro.empty:
        us_macro.to_parquet(MACRO_DIR / "us_macro.parquet")
        logger.info(f"Saved US macro: {us_macro.shape}")

    # === China Macro Indicators (from FRED) ===
    cn_series = {k: v for k, v in FRED_SERIES.items() if k.startswith("cn_")}
    cn_series["dexchus"] = FRED_SERIES["dexchus"]
    logger.info(f"Fetching {len(cn_series)} China macro series from FRED...")
    cn_macro = fetch_fred_series(fred, cn_series)
    if not cn_macro.empty:
        cn_macro.to_parquet(MACRO_DIR / "cn_macro_fred.parquet")
        logger.info(f"Saved China macro (FRED): {cn_macro.shape}")

    # === PBOC midprice proxy (DEXCHUS) ===
    # DEXCHUS is the daily CNY/USD rate published by the Fed, close proxy to PBOC midprice
    if "dexchus" in cn_macro.columns:
        pboc_proxy = cn_macro[["dexchus"]].dropna()
        pboc_proxy.columns = ["pboc_midprice_proxy"]
        from config import TIMESERIES_DIR
        TIMESERIES_DIR.mkdir(parents=True, exist_ok=True)
        pboc_proxy.to_parquet(TIMESERIES_DIR / "pboc_midprice.parquet")
        logger.info(f"Saved PBOC midprice proxy: {len(pboc_proxy)} days")

    # === Summary ===
    logger.info("=" * 60)
    logger.info("Phase 2 Complete!")
    if not us_macro.empty:
        logger.info(f"  US macro: {us_macro.shape[1]} indicators, {us_macro.shape[0]} observations")
    if not cn_macro.empty:
        logger.info(f"  CN macro: {cn_macro.shape[1]} indicators, {cn_macro.shape[0]} observations")


if __name__ == "__main__":
    main()
