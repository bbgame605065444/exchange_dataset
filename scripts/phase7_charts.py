"""
Phase 7: K-line Chart Generation
Sources: Phase 1 OHLCV data
Output: images/candlestick/*.png
"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import pandas as pd

from config import TIMESERIES_DIR, CANDLESTICK_DIR, CHART_LOOKBACKS, CHART_SIZE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_candlestick_charts(ohlcv: pd.DataFrame):
    """Generate candlestick charts for each trading day."""
    try:
        import mplfinance as mpf
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
    except ImportError:
        logger.error("mplfinance not installed. Run: pip install mplfinance")
        return

    CANDLESTICK_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure proper column names and DatetimeIndex
    ohlcv = ohlcv.copy()
    ohlcv.index = pd.to_datetime(ohlcv.index)
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in ohlcv.columns:
            logger.error(f"Missing column: {col}")
            return

    # Handle NaN volume (common in FX data)
    if ohlcv["Volume"].isna().all() or (ohlcv["Volume"] == 0).all():
        ohlcv["Volume"] = 1  # Placeholder for mplfinance

    total_charts = 0
    trading_days = ohlcv.index.tolist()

    # Compute technical indicators for overlays (pure pandas)
    close = ohlcv["Close"]
    ema5 = close.ewm(span=5, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb = pd.DataFrame({
        "BBU_20": sma20 + 2 * std20,
        "BBM_20": sma20,
        "BBL_20": sma20 - 2 * std20,
    }, index=ohlcv.index)
    fast_ema = close.ewm(span=12, adjust=False).mean()
    slow_ema = close.ewm(span=26, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd = pd.DataFrame({"MACD": macd_line, "Signal": signal_line}, index=ohlcv.index)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14).mean()
    rsi = 100 - (100 / (1 + gain / loss))

    # DPI calculation for target size
    fig_width_inches = CHART_SIZE[0] / 100
    fig_height_inches = CHART_SIZE[1] / 100

    # Style
    mc = mpf.make_marketcolors(up="green", down="red", inherit=True)
    style = mpf.make_mpf_style(marketcolors=mc, gridstyle=":", y_on_right=False)

    for i, date in enumerate(trading_days):
        date_str = date.strftime("%Y%m%d")

        for label, lookback in CHART_LOOKBACKS.items():
            start_idx = max(0, i - lookback + 1)
            if start_idx >= i:
                continue  # Not enough data

            window = ohlcv.iloc[start_idx:i + 1]
            if len(window) < 3:
                continue

            fname = f"USDCNH_{date_str}_{label}.png"
            fpath = CANDLESTICK_DIR / fname

            if fpath.exists():
                continue

            try:
                addplots = []
                panel_count = 0

                if label == "20d" and ema5 is not None and ema20 is not None:
                    # 20d chart: EMA overlays + RSI subplot
                    ema5_window = ema5.iloc[start_idx:i + 1]
                    ema20_window = ema20.iloc[start_idx:i + 1]
                    addplots.append(mpf.make_addplot(ema5_window, color="blue", width=0.7))
                    addplots.append(mpf.make_addplot(ema20_window, color="orange", width=0.7))
                    if rsi is not None:
                        rsi_window = rsi.iloc[start_idx:i + 1]
                        addplots.append(mpf.make_addplot(rsi_window, panel=2, color="purple", width=0.7))
                        panel_count = 2

                elif label == "60d" and bb is not None and macd is not None:
                    # 60d chart: Bollinger + MACD subplot
                    bb_cols = [c for c in bb.columns if "BBU" in c or "BBL" in c or "BBM" in c]
                    for col in bb_cols:
                        bb_window = bb[col].iloc[start_idx:i + 1]
                        color = "gray" if "BBM" in col else "lightblue"
                        addplots.append(mpf.make_addplot(bb_window, color=color, width=0.5))
                    macd_cols = macd.columns.tolist()
                    if len(macd_cols) >= 2:
                        macd_line = macd[macd_cols[0]].iloc[start_idx:i + 1]
                        signal_line = macd[macd_cols[1]].iloc[start_idx:i + 1]
                        addplots.append(mpf.make_addplot(macd_line, panel=2, color="blue", width=0.7))
                        addplots.append(mpf.make_addplot(signal_line, panel=2, color="red", width=0.7))
                        panel_count = 2

                kwargs = dict(
                    type="candle",
                    style=style,
                    volume=True,
                    figsize=(fig_width_inches, fig_height_inches),
                    savefig=dict(fname=str(fpath), dpi=100, bbox_inches="tight"),
                    tight_layout=True,
                )
                if addplots:
                    kwargs["addplot"] = addplots

                mpf.plot(window, **kwargs)
                total_charts += 1

                import matplotlib.pyplot as plt
                plt.close("all")

            except Exception as e:
                if total_charts == 0:
                    logger.warning(f"Chart generation error for {fname}: {e}")

        if (i + 1) % 100 == 0:
            logger.info(f"  Generated charts for {i + 1}/{len(trading_days)} days ({total_charts} total)")

    logger.info(f"Generated {total_charts} candlestick charts")


def main():
    # Load OHLCV data
    ohlcv_path = TIMESERIES_DIR / "USDCNH_daily_ohlcv.parquet"
    if not ohlcv_path.exists():
        logger.error(f"OHLCV data not found: {ohlcv_path}. Run Phase 1 first.")
        sys.exit(1)

    ohlcv = pd.read_parquet(ohlcv_path)
    logger.info(f"Loaded OHLCV: {len(ohlcv)} rows")

    logger.info("=== Generating Candlestick Charts ===")
    generate_candlestick_charts(ohlcv)

    logger.info("=" * 60)
    logger.info("Phase 7 Complete!")


if __name__ == "__main__":
    main()
