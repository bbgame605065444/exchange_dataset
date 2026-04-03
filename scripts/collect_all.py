#!/usr/bin/env python3
"""
CNH/USD Multimodal Dataset — Full Collection Pipeline
=====================================================
Tries every real API source; skips on missing key / network error and falls
back to synthetic data so the pipeline always completes.

Gaussian noise is injected into all HTTP-request intervals to randomise
spider timing fingerprints.

Usage:
    python scripts/collect_all.py          # full run
    python scripts/collect_all.py --fast   # skip chart generation
"""
import sys, os, json, time, random, math, logging, argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from config import (
    START_DATE, END_DATE,
    TIMESERIES_DIR, MACRO_DIR, NEWS_EN_DIR, CENTRAL_BANK_DIR,
    CANDLESTICK_DIR, ALIGNED_DIR, METADATA_DIR, SENTIMENT_DIR,
    NEWS_CN_DIR,
    TICKER_CNH, TICKER_CNY, TICKER_DXY, TICKER_VIX,
    FRED_API_KEY, FINNHUB_API_KEY,
    FRED_SERIES, GDELT_KEYWORDS, TECH_PARAMS,
    CHART_LOOKBACKS, CHART_SIZE,
)
from phase1_price_technical import compute_technical_indicators

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("collect_all")

# ── Collection log ──────────────────────────────────────────────────
collection_log: list[dict] = []

def log_source(name: str, status: str, detail: str = "", n_records: int = 0):
    entry = {
        "source": name, "status": status,
        "detail": detail, "n_records": n_records,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    collection_log.append(entry)
    sym = "✓" if status == "ok" else ("⊘" if status == "skipped" else "✗")
    logger.info(f"  {sym} {name}: {status}  {detail}  ({n_records} records)")


# ── Gaussian-noise sleep ────────────────────────────────────────────
def gaussian_sleep(mean: float = 1.0, std: float = 0.3):
    """Sleep for N(mean, std) seconds, clipped to [0.1, mean*3]."""
    t = max(0.1, min(mean * 3, random.gauss(mean, std)))
    time.sleep(t)


# ====================================================================
# Phase 1  — Price + Technical Indicators
# ====================================================================
def phase1_prices() -> pd.DataFrame | None:
    """Download OHLCV via yfinance; return CNH DataFrame or None."""
    TIMESERIES_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import yfinance as yf
    except ImportError:
        log_source("yfinance", "skipped", "library not installed")
        return None

    def _dl(ticker, label):
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE,
                             interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty:
                raise ValueError("empty result")
            logger.info(f"    {label}: {len(df)} rows")
            return df
        except Exception as e:
            logger.warning(f"    {label} failed: {e}")
            return pd.DataFrame()

    cnh = _dl(TICKER_CNH, "USD/CNH")
    gaussian_sleep(0.5, 0.15)
    cny = _dl(TICKER_CNY, "USD/CNY")
    gaussian_sleep(0.5, 0.15)
    dxy = _dl(TICKER_DXY, "DXY")
    gaussian_sleep(0.5, 0.15)
    vix = _dl(TICKER_VIX, "VIX")

    if cnh.empty:
        log_source("yfinance/CNH", "failed", "no data returned")
        return None

    cnh.to_parquet(TIMESERIES_DIR / "USDCNH_daily_ohlcv.parquet")
    log_source("yfinance/CNH", "ok", n_records=len(cnh))

    if not cny.empty:
        cny.to_parquet(TIMESERIES_DIR / "USDCNY_daily_ohlcv.parquet")
        spread = pd.DataFrame(index=cnh.index)
        cny_aligned = cny["Close"].reindex(cnh.index, method="ffill")
        spread["cnh_close"] = cnh["Close"]
        spread["cny_close"] = cny_aligned
        spread["cnh_cny_spread"] = cnh["Close"] - cny_aligned
        spread.to_parquet(TIMESERIES_DIR / "cnh_cny_spread.parquet")
        log_source("yfinance/CNY+spread", "ok", n_records=len(cny))

    aux = pd.DataFrame(index=cnh.index)
    if not dxy.empty:
        aux["dxy_close"] = dxy["Close"].reindex(cnh.index, method="ffill")
    if not vix.empty:
        aux["vix_close"] = vix["Close"].reindex(cnh.index, method="ffill")
    if not aux.empty:
        aux.to_parquet(TIMESERIES_DIR / "auxiliary_indicators.parquet")

    # Technical indicators
    tech = compute_technical_indicators(cnh)
    tech.to_parquet(TIMESERIES_DIR / "technical_indicators.parquet")
    log_source("technical_indicators", "ok", n_records=len(tech.columns))

    return cnh


# ====================================================================
# Phase 1-fallback  — Synthetic prices
# ====================================================================
def phase1_synthetic() -> pd.DataFrame:
    """Generate synthetic USD/CNH data as fallback."""
    logger.info("  Generating synthetic price data (fallback)...")
    from generate_sample_data import (
        generate_price_data, generate_technical_indicators,
    )
    cnh = generate_price_data()
    generate_technical_indicators(cnh)
    log_source("synthetic/prices", "ok", "fallback", n_records=len(cnh))
    return cnh


# ====================================================================
# Phase 2  — Macro (FRED)
# ====================================================================
def phase2_macro():
    MACRO_DIR.mkdir(parents=True, exist_ok=True)

    if not FRED_API_KEY:
        log_source("FRED", "skipped", "FRED_API_KEY not set")
        _phase2_synthetic()
        return

    try:
        from fredapi import Fred
    except ImportError:
        log_source("FRED", "skipped", "fredapi not installed")
        _phase2_synthetic()
        return

    fred = Fred(api_key=FRED_API_KEY)

    def _fetch(series_dict, label):
        frames = {}
        for name, sid in series_dict.items():
            try:
                s = fred.get_series(sid, observation_start=START_DATE,
                                    observation_end=END_DATE)
                if s is not None and len(s) > 0:
                    frames[name] = s
            except Exception as e:
                logger.warning(f"    {sid}: {e}")
            gaussian_sleep(0.3, 0.1)
        if frames:
            df = pd.DataFrame(frames)
            log_source(f"FRED/{label}", "ok", n_records=df.shape[0])
            return df
        log_source(f"FRED/{label}", "failed", "no series returned")
        return pd.DataFrame()

    us_series = {k: v for k, v in FRED_SERIES.items() if k.startswith("us_")}
    us_series["fed_funds_rate"] = FRED_SERIES["fed_funds_rate"]
    us = _fetch(us_series, "US")
    if not us.empty:
        us.to_parquet(MACRO_DIR / "us_macro.parquet")

    cn_series = {k: v for k, v in FRED_SERIES.items() if k.startswith("cn_")}
    cn_series["dexchus"] = FRED_SERIES["dexchus"]
    cn = _fetch(cn_series, "CN")
    if not cn.empty:
        cn.to_parquet(MACRO_DIR / "cn_macro_fred.parquet")
        if "dexchus" in cn.columns:
            pboc = cn[["dexchus"]].dropna()
            pboc.columns = ["pboc_midprice_proxy"]
            TIMESERIES_DIR.mkdir(parents=True, exist_ok=True)
            pboc.to_parquet(TIMESERIES_DIR / "pboc_midprice.parquet")

    if us.empty and cn.empty:
        _phase2_synthetic()


def _phase2_synthetic():
    logger.info("  Generating synthetic macro data (fallback)...")
    from generate_sample_data import generate_macro_data
    generate_macro_data()
    log_source("synthetic/macro", "ok", "fallback")


# ====================================================================
# Phase 3  — English News (GDELT + Finnhub)
# ====================================================================
def phase3_news():
    NEWS_EN_DIR.mkdir(parents=True, exist_ok=True)
    got_any = False

    # ── GDELT ──
    try:
        from gdeltdoc import GdeltDoc, Filters
        gd = GdeltDoc()
        all_articles = []
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=89)

        for kw in GDELT_KEYWORDS:
            logger.info(f"    GDELT: {kw[:50]}...")
            try:
                f = Filters(keyword=kw,
                            start_date=start_dt.strftime("%Y-%m-%d"),
                            end_date=end_dt.strftime("%Y-%m-%d"),
                            num_records=250)
                arts = gd.article_search(f)
                if arts is not None and len(arts) > 0:
                    recs = arts.to_dict("records")
                    for r in recs:
                        r["query_group"] = kw
                    all_articles.extend(recs)
            except Exception as e:
                logger.warning(f"      {e}")
            gaussian_sleep(1.5, 0.4)

        # deduplicate
        seen = set()
        unique = []
        for a in all_articles:
            u = a.get("url", "")
            if u and u not in seen:
                seen.add(u)
                unique.append(a)

        out = NEWS_EN_DIR / "gdelt_cnhusd.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for a in unique:
                clean = {}
                for k, v in a.items():
                    if isinstance(v, (pd.Timestamp, datetime)):
                        clean[k] = v.isoformat()
                    elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                        clean[k] = None
                    else:
                        clean[k] = v
                f.write(json.dumps(clean, ensure_ascii=False) + "\n")
        log_source("GDELT", "ok", n_records=len(unique))
        if unique:
            got_any = True
    except ImportError:
        log_source("GDELT", "skipped", "gdeltdoc not installed")
    except Exception as e:
        log_source("GDELT", "failed", str(e)[:120])

    # ── Finnhub ──
    if not FINNHUB_API_KEY:
        log_source("Finnhub/news", "skipped", "FINNHUB_API_KEY not set")
    else:
        try:
            import finnhub
            client = finnhub.Client(api_key=FINNHUB_API_KEY)
            all_news = []
            try:
                news = client.general_news("forex", min_id=0)
                if news:
                    all_news.extend(news)
            except Exception as e:
                logger.warning(f"    Finnhub forex: {e}")
            gaussian_sleep(1.0, 0.3)

            try:
                news = client.general_news("general", min_id=0)
                if news:
                    kws = ["yuan","cnh","rmb","pboc","china trade","fed rate",
                           "dollar","forex","currency","exchange rate"]
                    filt = [n for n in news
                            if any(k in (n.get("headline","")+n.get("summary","")).lower()
                                   for k in kws)]
                    all_news.extend(filt)
            except Exception as e:
                logger.warning(f"    Finnhub general: {e}")

            seen_ids = set()
            unique_news = []
            for n in all_news:
                nid = n.get("id", n.get("url",""))
                if nid and nid not in seen_ids:
                    seen_ids.add(nid)
                    unique_news.append(n)

            out = NEWS_EN_DIR / "finnhub_forex.jsonl"
            with open(out, "w", encoding="utf-8") as f:
                for a in unique_news:
                    f.write(json.dumps(a, ensure_ascii=False) + "\n")
            log_source("Finnhub/news", "ok", n_records=len(unique_news))
            if unique_news:
                got_any = True
        except ImportError:
            log_source("Finnhub/news", "skipped", "finnhub-python not installed")
        except Exception as e:
            log_source("Finnhub/news", "failed", str(e)[:120])

    if not got_any:
        logger.info("  Generating synthetic news data (fallback)...")
        from generate_sample_data import generate_news_data
        generate_news_data()
        log_source("synthetic/news", "ok", "fallback")


# ====================================================================
# Phase 5  — Central Bank Communications
# ====================================================================
def phase5_central_bank():
    CENTRAL_BANK_DIR.mkdir(parents=True, exist_ok=True)
    got_any = False

    # ── FOMC labeled (HuggingFace) ──
    try:
        from datasets import load_dataset
        ds = load_dataset("gtfintechlab/fomc_communication", trust_remote_code=True)
        out = CENTRAL_BANK_DIR / "fomc_labeled.jsonl"
        count = 0
        with open(out, "w", encoding="utf-8") as f:
            for split in ds:
                for row in ds[split]:
                    rec = dict(row)
                    rec["split"] = split
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    count += 1
        log_source("HuggingFace/FOMC", "ok", n_records=count)
        got_any = True
    except ImportError:
        log_source("HuggingFace/FOMC", "skipped", "datasets not installed")
    except Exception as e:
        log_source("HuggingFace/FOMC", "failed", str(e)[:120])

    # ── Fed statements (scrape) ──
    try:
        import requests, re
        from bs4 import BeautifulSoup
        base = "https://www.federalreserve.gov"
        resp = requests.get(f"{base}/monetarypolicy/fomccalendars.htm", timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        links = [a for a in soup.find_all("a", href=True)
                 if "statement" in a["href"].lower()
                 and "monetarypolicy" in a["href"].lower()]

        stmts = []
        for link in links:
            href = link["href"]
            if not href.startswith("http"):
                href = base + href
            m = re.search(r"(\d{8})", href)
            if m:
                d = datetime.strptime(m.group(1), "%Y%m%d")
                if d < datetime.strptime(START_DATE, "%Y-%m-%d"):
                    continue
            gaussian_sleep(2.0, 0.6)
            try:
                r2 = requests.get(href, timeout=30)
                r2.raise_for_status()
                s2 = BeautifulSoup(r2.text, "lxml")
                div = s2.find("div", class_="col-xs-12")
                text = (div or s2).get_text("\n", strip=True)[:5000]
                stmts.append({"date": m.group(1) if m else "",
                              "url": href, "text": text,
                              "type": "fomc_statement"})
            except Exception as e:
                logger.warning(f"    Fed stmt {href}: {e}")

        out = CENTRAL_BANK_DIR / "fed_statements.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for s in stmts:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        log_source("Fed/statements", "ok", n_records=len(stmts))
        if stmts:
            got_any = True
    except Exception as e:
        log_source("Fed/statements", "failed", str(e)[:120])

    # ── Finnhub economic calendar ──
    if not FINNHUB_API_KEY:
        log_source("Finnhub/calendar", "skipped", "no key")
    else:
        try:
            import finnhub
            client = finnhub.Client(api_key=FINNHUB_API_KEY)
            cal = client.calendar_economic(_from=START_DATE, to=END_DATE)
            events = cal.get("economicCalendar", [])
            filt = [e for e in events if e.get("country","") in ("US","CN","")]
            out = CENTRAL_BANK_DIR.parent / "economic_calendar.jsonl"
            with open(out, "w", encoding="utf-8") as f:
                for ev in filt:
                    f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            log_source("Finnhub/calendar", "ok", n_records=len(filt))
        except Exception as e:
            log_source("Finnhub/calendar", "failed", str(e)[:120])

    if not got_any:
        logger.info("  Generating synthetic central bank data (fallback)...")
        from generate_sample_data import generate_central_bank_data
        generate_central_bank_data()
        log_source("synthetic/central_bank", "ok", "fallback")


# ====================================================================
# Phase 7  — Chart Generation (optional, slow)
# ====================================================================
def phase7_charts(every_nth: int = 5):
    """Generate candlestick charts for every Nth trading day."""
    ohlcv_path = TIMESERIES_DIR / "USDCNH_daily_ohlcv.parquet"
    if not ohlcv_path.exists():
        logger.warning("  No OHLCV data for chart generation")
        return

    try:
        import mplfinance as mpf
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log_source("charts", "skipped", "mplfinance not installed")
        return

    CANDLESTICK_DIR.mkdir(parents=True, exist_ok=True)
    ohlcv = pd.read_parquet(ohlcv_path)
    ohlcv.index = pd.to_datetime(ohlcv.index)
    if ohlcv["Volume"].isna().all() or (ohlcv["Volume"] == 0).all():
        ohlcv["Volume"] = 1

    mc = mpf.make_marketcolors(up="green", down="red", inherit=True)
    style = mpf.make_mpf_style(marketcolors=mc, gridstyle=":", y_on_right=False)
    fw = CHART_SIZE[0] / 100
    fh = CHART_SIZE[1] / 100

    total = 0
    days = ohlcv.index.tolist()
    for i, date in enumerate(days):
        if i % every_nth != 0:
            continue
        ds = date.strftime("%Y%m%d")
        for label, lb in CHART_LOOKBACKS.items():
            si = max(0, i - lb + 1)
            if si >= i:
                continue
            window = ohlcv.iloc[si:i+1]
            if len(window) < 3:
                continue
            fname = f"USDCNH_{ds}_{label}.png"
            fpath = CANDLESTICK_DIR / fname
            if fpath.exists():
                continue
            try:
                mpf.plot(window, type="candle", style=style, volume=True,
                         figsize=(fw, fh),
                         savefig=dict(fname=str(fpath), dpi=100,
                                      bbox_inches="tight"),
                         tight_layout=True)
                plt.close("all")
                total += 1
            except Exception:
                pass
        if (i+1) % 200 == 0:
            logger.info(f"    Charts: {i+1}/{len(days)} days, {total} images")

    log_source("charts", "ok", f"every {every_nth}th day", n_records=total)


# ====================================================================
# Phase 4  — Chinese News (Sina + Eastmoney)
# ====================================================================
def phase4_chinese_news():
    NEWS_CN_DIR.mkdir(parents=True, exist_ok=True)
    got_any = False

    try:
        from phase4_news_cn import collect_sina_forex, collect_eastmoney_forex, _save_jsonl

        sina = collect_sina_forex(max_pages=20)
        if sina:
            _save_jsonl(sina, NEWS_CN_DIR / "sina_forex.jsonl")
            log_source("Sina/forex", "ok", n_records=len(sina))
            got_any = True
        else:
            log_source("Sina/forex", "failed", "no articles")
    except Exception as e:
        log_source("Sina/forex", "failed", str(e)[:120])

    try:
        from phase4_news_cn import collect_eastmoney_forex, _save_jsonl

        eastmoney = collect_eastmoney_forex(max_pages=20)
        if eastmoney:
            _save_jsonl(eastmoney, NEWS_CN_DIR / "eastmoney_forex.jsonl")
            log_source("Eastmoney/forex", "ok", n_records=len(eastmoney))
            got_any = True
        else:
            log_source("Eastmoney/forex", "failed", "no articles")
    except Exception as e:
        log_source("Eastmoney/forex", "failed", str(e)[:120])

    if not got_any:
        logger.info("  Generating synthetic Chinese news (fallback)...")
        _generate_cn_news_synthetic()
        log_source("synthetic/news_cn", "ok", "fallback")


def _generate_cn_news_synthetic():
    """Generate synthetic Chinese forex news across the full date range."""
    NEWS_CN_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(123)

    templates_cn = [
        "人民币兑美元中间价报{rate:.4f}，{dir}{pips}个基点",
        "离岸人民币{dir2}破{level}关口，日内{chg}",
        "央行：保持人民币汇率在合理均衡水平上基本稳定",
        "外汇储备{month}末为{reserves}亿美元，环比{dir3}{delta}亿",
        "中美贸易谈判{progress}，市场情绪{sentiment}",
        "美联储{action}利率，美元指数{dxy_dir}",
        "跨境资金流动总体{flow}，外汇市场供求基本平衡",
        "在岸离岸价差{spread_dir}至{spread}点，资本管制压力{pressure}",
        "国家统计局：{month}制造业PMI为{pmi}，{pmi_dir}预期",
        "中国{quarter}季度GDP同比增长{gdp}%，{gdp_dir}市场预期",
    ]

    dates = pd.bdate_range(start=START_DATE, end=END_DATE)
    articles = []
    for d in dates:
        if np.random.random() > 0.3:  # ~70% of days have news
            n_articles = np.random.randint(1, 4)
            for _ in range(n_articles):
                tmpl = np.random.choice(templates_cn)
                title = tmpl.format(
                    rate=np.random.uniform(6.3, 7.4),
                    dir=np.random.choice(["上调", "下调"]),
                    dir2=np.random.choice(["升值", "贬值"]),
                    dir3=np.random.choice(["增加", "减少"]),
                    pips=np.random.randint(5, 300),
                    level=np.random.choice(["6.8", "7.0", "7.1", "7.2", "7.3"]),
                    chg=np.random.choice(["涨超200点", "跌超150点", "波动加大", "窄幅震荡"]),
                    month=np.random.choice(["1月", "2月", "3月", "6月", "9月", "12月"]),
                    reserves=np.random.randint(30000, 33000),
                    delta=np.random.randint(20, 200),
                    progress=np.random.choice(["取得积极进展", "仍存在分歧", "即将重启"]),
                    sentiment=np.random.choice(["偏乐观", "趋于谨慎", "有所改善"]),
                    action=np.random.choice(["维持", "上调", "下调"]),
                    dxy_dir=np.random.choice(["走强", "走弱", "震荡"]),
                    flow=np.random.choice(["稳定", "波动加大", "趋于平衡"]),
                    spread_dir=np.random.choice(["扩大", "收窄"]),
                    spread=np.random.randint(50, 500),
                    pressure=np.random.choice(["上升", "缓解", "基本稳定"]),
                    pmi=round(np.random.uniform(48.5, 52.5), 1),
                    pmi_dir=np.random.choice(["高于", "低于", "符合"]),
                    quarter=np.random.choice(["一", "二", "三", "四"]),
                    gdp=round(np.random.uniform(4.0, 6.5), 1),
                    gdp_dir=np.random.choice(["超出", "低于", "符合"]),
                )
                articles.append({
                    "title": title,
                    "date": d.strftime("%Y-%m-%d"),
                    "source": "synthetic",
                    "language": "zh",
                })

    out = NEWS_CN_DIR / "sina_forex.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for a in articles:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    logger.info(f"  Generated {len(articles)} synthetic Chinese news articles")


# ====================================================================
# Phase 6  — Sentiment Scoring
# ====================================================================
def phase6_sentiment():
    SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)

    # Check if news data exists
    en_files = list(NEWS_EN_DIR.glob("*.jsonl")) if NEWS_EN_DIR.exists() else []
    cn_files = list(NEWS_CN_DIR.glob("*.jsonl")) if NEWS_CN_DIR.exists() else []
    if not en_files and not cn_files:
        log_source("sentiment", "skipped", "no news data")
        return

    # Try FinBERT for English
    try:
        from transformers import pipeline as hf_pipeline
        logger.info("  Loading FinBERT for English sentiment...")
        classifier = hf_pipeline(
            "sentiment-analysis", model="ProsusAI/finbert",
            truncation=True, max_length=512, device=-1,
        )
        texts, sources = [], []
        for f in en_files:
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    t = rec.get("title") or rec.get("headline") or ""
                    s = rec.get("summary", "")
                    combined = f"{t}. {s}".strip()
                    if len(combined) > 10:
                        texts.append(combined[:512])
                        sources.append(f.name)

        if texts:
            results = classifier(texts, batch_size=32)
            df = pd.DataFrame({
                "text": texts, "source": sources,
                "label": [r["label"] for r in results],
                "score": [r["score"] for r in results],
            })
            df.to_parquet(SENTIMENT_DIR / "en_sentiment.parquet")
            log_source("FinBERT/en", "ok", n_records=len(df))
        else:
            log_source("FinBERT/en", "skipped", "no texts")

    except ImportError:
        log_source("FinBERT/en", "skipped", "transformers not installed")
    except Exception as e:
        log_source("FinBERT/en", "failed", str(e)[:120])

    # Keyword-based fallback sentiment for Chinese
    if cn_files:
        _keyword_sentiment_cn(cn_files)


def _keyword_sentiment_cn(cn_files):
    """Simple keyword-based sentiment for Chinese news (no model needed)."""
    pos_kw = ["升值", "走强", "上涨", "乐观", "改善", "超出预期", "增长",
              "稳定", "平衡", "积极", "回升", "扩大顺差"]
    neg_kw = ["贬值", "走弱", "下跌", "悲观", "恶化", "低于预期", "下滑",
              "动荡", "外流", "收紧", "施压", "关税"]

    records = []
    for f in cn_files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                title = rec.get("title", "")
                date = rec.get("date", "")
                pos = sum(1 for k in pos_kw if k in title)
                neg = sum(1 for k in neg_kw if k in title)
                if pos > neg:
                    label = "positive"
                elif neg > pos:
                    label = "negative"
                else:
                    label = "neutral"
                score = max(pos, neg) / (pos + neg + 1)
                records.append({
                    "text": title, "date": date, "source": f.name,
                    "label": label, "score": score,
                })

    if records:
        df = pd.DataFrame(records)
        df.to_parquet(SENTIMENT_DIR / "cn_sentiment.parquet")
        log_source("keyword/cn_sentiment", "ok", n_records=len(df))


# ====================================================================
# Phase 8  — Alignment  (delegates to updated phase8_alignment.py)
# ====================================================================
def phase8_align():
    from phase8_alignment import main as align_main
    align_main()
    log_source("alignment", "ok")


# ====================================================================
# Phase 9  — LLM Explanations
# ====================================================================
def phase9_explanations():
    try:
        from phase9_explanations import main as explain_main
        explain_main()
        log_source("explanations", "ok")
    except Exception as e:
        log_source("explanations", "failed", str(e)[:120])


# ====================================================================
# Main
# ====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Skip chart generation and LLM explanations")
    parser.add_argument("--chart-every", type=int, default=5,
                        help="Generate charts every Nth day (default 5)")
    args = parser.parse_args()

    logger.info("=" * 64)
    logger.info(" CNH/USD Multimodal Dataset — Collection Pipeline")
    logger.info(f" Date range : {START_DATE} → {END_DATE}")
    logger.info(f" FRED key   : {'set' if FRED_API_KEY else 'NOT SET (skip)'}")
    logger.info(f" Finnhub key: {'set' if FINNHUB_API_KEY else 'NOT SET (skip)'}")
    logger.info("=" * 64)

    # ── Phase 1: Prices ──
    logger.info("\n▸ Phase 1 — Price + Technical Indicators")
    cnh = phase1_prices()
    if cnh is None:
        cnh = phase1_synthetic()

    # ── Phase 2: Macro ──
    logger.info("\n▸ Phase 2 — Macroeconomic Data")
    phase2_macro()

    # ── Phase 3: English News ──
    logger.info("\n▸ Phase 3 — English News")
    phase3_news()

    # ── Phase 4: Chinese News ──
    logger.info("\n▸ Phase 4 — Chinese News")
    phase4_chinese_news()

    # ── Phase 5: Central Bank ──
    logger.info("\n▸ Phase 5 — Central Bank Communications")
    phase5_central_bank()

    # ── Phase 6: Sentiment ──
    logger.info("\n▸ Phase 6 — Sentiment Scoring")
    phase6_sentiment()

    # ── Phase 7: Charts ──
    if not args.fast:
        logger.info(f"\n▸ Phase 7 — Chart Generation (every {args.chart_every}th day)")
        phase7_charts(every_nth=args.chart_every)
    else:
        log_source("charts", "skipped", "--fast flag")

    # ── Phase 8: Alignment ──
    logger.info("\n▸ Phase 8 — Timestamp Alignment & Assembly")
    phase8_align()

    # ── Phase 9: Explanations ──
    if not args.fast:
        logger.info("\n▸ Phase 9 — LLM Explanations")
        phase9_explanations()
    else:
        log_source("explanations", "skipped", "--fast flag")

    # ── Save collection log ──
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    log_path = METADATA_DIR / "collection_log.jsonl"
    with open(log_path, "w") as f:
        for entry in collection_log:
            f.write(json.dumps(entry) + "\n")

    # ── Summary ──
    ok = sum(1 for e in collection_log if e["status"] == "ok")
    skip = sum(1 for e in collection_log if e["status"] == "skipped")
    fail = sum(1 for e in collection_log if e["status"] == "failed")
    logger.info("\n" + "=" * 64)
    logger.info(f" Collection complete:  {ok} ok / {skip} skipped / {fail} failed")
    logger.info(f" Log → {log_path}")
    aligned = ALIGNED_DIR / "USDCNH_daily_aligned.parquet"
    if aligned.exists():
        df = pd.read_parquet(aligned)
        logger.info(f" Dataset → {aligned}")
        logger.info(f"   {df.shape[0]} rows × {df.shape[1]} columns")
    logger.info("=" * 64)


if __name__ == "__main__":
    main()
