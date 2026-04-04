"""
Phase 1: Price Data + Technical Indicators
Sources: yfinance (CNH, CNY, DXY, VIX)
Output: timeseries/USDCNH_daily_ohlcv.parquet, timeseries/technical_indicators.parquet
"""
import sys
import time
import logging

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

from config import (
    START_DATE, END_DATE,
    TICKER_CNH, TICKER_CNY, TICKER_DXY, TICKER_VIX,
    TIMESERIES_DIR, TECH_PARAMS,
    HOURLY_DIR, MINUTE_DIR,
    HOURLY_LOOKBACK_DAYS, MINUTE_LOOKBACK_DAYS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def download_ticker(ticker: str, name: str) -> pd.DataFrame:
    """Download daily OHLCV data from yfinance."""
    if yf is None:
        logger.warning("yfinance not installed")
        return pd.DataFrame()
    logger.info(f"Downloading {name} ({ticker})...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE, interval="1d", progress=False)
    if df.empty:
        logger.warning(f"No data returned for {ticker}")
        return df
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    logger.info(f"  {name}: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
    return df


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators from OHLCV data using pure pandas."""
    import numpy as np

    logger.info("Computing technical indicators...")
    result = pd.DataFrame(index=df.index)
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # EMA
    for period in TECH_PARAMS["ema_periods"]:
        result[f"ema_{period}"] = _ema(close, period)

    # RSI
    result["rsi_14"] = _rsi(close, TECH_PARAMS["rsi_period"])

    # MACD
    fast = _ema(close, TECH_PARAMS["macd_fast"])
    slow = _ema(close, TECH_PARAMS["macd_slow"])
    result["macd"] = fast - slow
    result["macd_signal"] = _ema(result["macd"], TECH_PARAMS["macd_signal"])
    result["macd_hist"] = result["macd"] - result["macd_signal"]

    # ATR
    result["atr_14"] = _atr(high, low, close, TECH_PARAMS["atr_period"])

    # Bollinger Bands
    bb_period = TECH_PARAMS["bb_period"]
    bb_std = TECH_PARAMS["bb_std"]
    sma = close.rolling(bb_period).mean()
    std = close.rolling(bb_period).std()
    result["bb_upper"] = sma + bb_std * std
    result["bb_middle"] = sma
    result["bb_lower"] = sma - bb_std * std

    # ADX
    adx_period = TECH_PARAMS["adx_period"]
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    # Zero out where opposite is larger
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0
    atr_val = _atr(high, low, close, adx_period)
    plus_di = 100 * _ema(plus_dm, adx_period) / atr_val
    minus_di = 100 * _ema(minus_dm, adx_period) / atr_val
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    result["adx"] = _ema(dx, adx_period)
    result["plus_di"] = plus_di
    result["minus_di"] = minus_di

    # OBV
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    result["obv"] = obv

    logger.info(f"  Computed {len(result.columns)} technical indicator columns")
    return result


def download_ticker_intraday(ticker: str, name: str, interval: str = "1h",
                             lookback_days: int = 365) -> pd.DataFrame:
    """Download intraday OHLCV data from yfinance.

    Parameters
    ----------
    ticker : yfinance ticker symbol
    name : human-readable label for logging
    interval : '1h' or '1m'
    lookback_days : how many calendar days back to fetch
    """
    end = pd.Timestamp.now()
    start = end - pd.Timedelta(days=lookback_days)

    if interval not in ("1h", "1m"):
        raise ValueError(f"Unsupported interval: {interval}")

    if yf is None:
        logger.warning("yfinance not installed")
        return pd.DataFrame()

    logger.info(f"Downloading {name} ({ticker}) {interval} data, {start.date()} -> {end.date()}...")

    if interval == "1h":
        # yfinance allows up to ~730 days for hourly; chunk if needed
        chunk_size = 700
        frames = []
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + pd.Timedelta(days=chunk_size), end)
            try:
                df = yf.download(ticker, start=chunk_start.strftime("%Y-%m-%d"),
                                 end=chunk_end.strftime("%Y-%m-%d"),
                                 interval="1h", progress=False)
                if not df.empty:
                    frames.append(df)
            except Exception as e:
                logger.warning(f"  Chunk {chunk_start.date()}-{chunk_end.date()} failed: {e}")
            chunk_start = chunk_end
            time.sleep(0.3)
        if not frames:
            logger.warning(f"No hourly data returned for {ticker}")
            return pd.DataFrame()
        df = pd.concat(frames)
    elif interval == "1m":
        # yfinance allows max 7 days for minute data
        start = end - pd.Timedelta(days=min(lookback_days, 7))
        try:
            df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                             end=end.strftime("%Y-%m-%d"),
                             interval="1m", progress=False)
        except Exception as e:
            logger.warning(f"Minute download failed for {ticker}: {e}")
            return pd.DataFrame()

    if df.empty:
        logger.warning(f"No {interval} data returned for {ticker}")
        return df

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Deduplicate index
    df = df[~df.index.duplicated(keep="first")].sort_index()
    logger.info(f"  {name} {interval}: {len(df)} rows, {df.index[0]} to {df.index[-1]}")
    return df


def collect_intraday():
    """Collect hourly and minute OHLCV + technical indicators for CNH."""
    for interval, out_dir, lookback in [
        ("1h", HOURLY_DIR, HOURLY_LOOKBACK_DAYS),
        ("1m", MINUTE_DIR, MINUTE_LOOKBACK_DAYS),
    ]:
        out_dir.mkdir(parents=True, exist_ok=True)
        label = "hourly" if interval == "1h" else "minute"

        try:
            cnh = download_ticker_intraday(TICKER_CNH, "USD/CNH", interval, lookback)
        except Exception as e:
            logger.warning(f"yfinance intraday error ({interval}): {e}")
            cnh = pd.DataFrame()

        if cnh.empty:
            logger.warning(f"No {label} data from yfinance. Falling back to synthetic.")
            import subprocess
            subprocess.run([
                sys.executable,
                str(__import__("pathlib").Path(__file__).parent / "generate_sample_data.py"),
                "--intraday", interval,
            ])
            continue

        # Save OHLCV
        ohlcv_path = out_dir / f"USDCNH_{label}_ohlcv.parquet"
        cnh.to_parquet(ohlcv_path)
        logger.info(f"Saved {label} OHLCV: {ohlcv_path}")

        # Compute and save technical indicators
        tech = compute_technical_indicators(cnh)
        tech_path = out_dir / f"{label}_technical_indicators.parquet"
        tech.to_parquet(tech_path)
        logger.info(f"Saved {label} technical indicators: {len(tech.columns)} columns")


def main():
    TIMESERIES_DIR.mkdir(parents=True, exist_ok=True)

    # 1.1 Download CNH (offshore RMB)
    try:
        cnh = download_ticker(TICKER_CNH, "USD/CNH")
    except Exception as e:
        logger.warning(f"yfinance error: {e}")
        cnh = pd.DataFrame()

    if cnh.empty:
        logger.warning("Failed to download CNH data from yfinance. Falling back to synthetic data.")
        logger.warning("Re-run with network access for real data.")
        import subprocess
        subprocess.run([sys.executable, str(__import__("pathlib").Path(__file__).parent / "generate_sample_data.py")])
        return

    # 1.2 Download CNY (onshore RMB) for spread calculation
    time.sleep(0.5)
    cny = download_ticker(TICKER_CNY, "USD/CNY")

    # 1.3 Download DXY and VIX
    time.sleep(0.5)
    dxy = download_ticker(TICKER_DXY, "DXY")
    time.sleep(0.5)
    vix = download_ticker(TICKER_VIX, "VIX")

    # === Save OHLCV ===
    cnh.to_parquet(TIMESERIES_DIR / "USDCNH_daily_ohlcv.parquet")
    logger.info(f"Saved CNH OHLCV: {TIMESERIES_DIR / 'USDCNH_daily_ohlcv.parquet'}")

    if not cny.empty:
        cny.to_parquet(TIMESERIES_DIR / "USDCNY_daily_ohlcv.parquet")
        logger.info(f"Saved CNY OHLCV")

    # === Compute CNH-CNY spread ===
    if not cny.empty:
        spread = pd.DataFrame(index=cnh.index)
        cny_aligned = cny["Close"].reindex(cnh.index, method="ffill")
        spread["cnh_close"] = cnh["Close"]
        spread["cny_close"] = cny_aligned
        spread["cnh_cny_spread"] = cnh["Close"] - cny_aligned
        spread.to_parquet(TIMESERIES_DIR / "cnh_cny_spread.parquet")
        logger.info(f"Saved CNH-CNY spread: mean={spread['cnh_cny_spread'].mean():.4f}")

    # === Auxiliary indicators ===
    aux = pd.DataFrame(index=cnh.index)
    if not dxy.empty:
        aux["dxy_close"] = dxy["Close"].reindex(cnh.index, method="ffill")
    if not vix.empty:
        aux["vix_close"] = vix["Close"].reindex(cnh.index, method="ffill")
    if not aux.empty:
        aux.to_parquet(TIMESERIES_DIR / "auxiliary_indicators.parquet")
        logger.info("Saved auxiliary indicators (DXY, VIX)")

    # === Technical indicators ===
    tech = compute_technical_indicators(cnh)
    tech.to_parquet(TIMESERIES_DIR / "technical_indicators.parquet")
    logger.info(f"Saved technical indicators: {len(tech.columns)} columns")

    # === Summary ===
    logger.info("=" * 60)
    logger.info("Phase 1 Complete!")
    logger.info(f"  CNH: {len(cnh)} trading days")
    logger.info(f"  CNY: {len(cny)} trading days")
    logger.info(f"  DXY: {len(dxy)} trading days")
    logger.info(f"  VIX: {len(vix)} trading days")
    logger.info(f"  Technical indicators: {len(tech.columns)} features")
    logger.info(f"  Date range: {cnh.index[0].date()} to {cnh.index[-1].date()}")

    # === Intraday data (hourly + minute) ===
    logger.info("\n=== Intraday Data Collection ===")
    collect_intraday()


if __name__ == "__main__":
    main()
