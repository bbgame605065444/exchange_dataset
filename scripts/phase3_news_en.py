"""
Phase 3: English News Collection
Sources: GDELT DOC API (recent 3 months), Finnhub (recent 1 year)
Output: text/news_en/gdelt_cnhusd.jsonl, text/news_en/finnhub_forex.jsonl
"""
import sys
import json
import time
import logging
from datetime import datetime, timedelta

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import pandas as pd

from config import NEWS_EN_DIR, GDELT_KEYWORDS, FINNHUB_API_KEY

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def collect_gdelt_recent():
    """Collect recent news from GDELT DOC API (last 3 months)."""
    try:
        from gdeltdoc import GdeltDoc, Filters
    except ImportError:
        logger.error("gdeltdoc not installed. Run: pip install gdeltdoc")
        return

    gd = GdeltDoc()
    all_articles = []

    # GDELT DOC API covers last 3 months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=89)

    for keyword_group in GDELT_KEYWORDS:
        logger.info(f"GDELT query: {keyword_group[:60]}...")
        try:
            f = Filters(
                keyword=keyword_group,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                num_records=250,
            )
            articles = gd.article_search(f)
            if articles is not None and len(articles) > 0:
                records = articles.to_dict("records")
                for r in records:
                    r["query_group"] = keyword_group
                all_articles.extend(records)
                logger.info(f"  -> {len(records)} articles")
            else:
                logger.info("  -> 0 articles")
        except Exception as e:
            logger.warning(f"  -> Error: {e}")
        time.sleep(1)  # Be polite

    # Deduplicate by URL
    seen_urls = set()
    unique_articles = []
    for a in all_articles:
        url = a.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_articles.append(a)

    # Save
    output_path = NEWS_EN_DIR / "gdelt_cnhusd.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for article in unique_articles:
            # Convert any non-serializable types
            clean = {}
            for k, v in article.items():
                if isinstance(v, (pd.Timestamp, datetime)):
                    clean[k] = v.isoformat()
                elif pd.isna(v) if isinstance(v, float) else False:
                    clean[k] = None
                else:
                    clean[k] = v
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(unique_articles)} unique GDELT articles to {output_path}")
    return unique_articles


def collect_finnhub_news():
    """Collect forex news from Finnhub API."""
    if not FINNHUB_API_KEY:
        logger.warning(
            "FINNHUB_API_KEY not set. Skipping Finnhub collection.\n"
            "Get a free key at: https://finnhub.io/register"
        )
        return

    try:
        import finnhub
    except ImportError:
        logger.error("finnhub-python not installed. Run: pip install finnhub-python")
        return

    client = finnhub.Client(api_key=FINNHUB_API_KEY)
    all_news = []

    # Finnhub general news (forex category)
    logger.info("Fetching Finnhub forex news...")
    try:
        news = client.general_news("forex", min_id=0)
        if news:
            all_news.extend(news)
            logger.info(f"  -> {len(news)} forex news articles")
    except Exception as e:
        logger.warning(f"  -> Error: {e}")

    # Also try company news for relevant keywords
    # Finnhub forex candles are limited, but news is useful
    logger.info("Fetching Finnhub general news...")
    try:
        news = client.general_news("general", min_id=0)
        if news:
            # Filter for CNH/USD related
            keywords = ["yuan", "cnh", "rmb", "pboc", "china trade", "fed rate",
                        "dollar", "forex", "currency", "exchange rate"]
            filtered = [
                n for n in news
                if any(kw in (n.get("headline", "") + " " + n.get("summary", "")).lower()
                       for kw in keywords)
            ]
            all_news.extend(filtered)
            logger.info(f"  -> {len(filtered)} relevant general news (from {len(news)} total)")
    except Exception as e:
        logger.warning(f"  -> Error: {e}")

    # Deduplicate by id
    seen_ids = set()
    unique_news = []
    for n in all_news:
        nid = n.get("id", n.get("url", ""))
        if nid and nid not in seen_ids:
            seen_ids.add(nid)
            unique_news.append(n)

    # Save
    output_path = NEWS_EN_DIR / "finnhub_forex.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for article in unique_news:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(unique_news)} Finnhub articles to {output_path}")
    return unique_news


def main():
    NEWS_EN_DIR.mkdir(parents=True, exist_ok=True)

    # GDELT (no key needed, no rate limit)
    logger.info("=== GDELT News Collection ===")
    gdelt_articles = collect_gdelt_recent()

    # Finnhub (needs key)
    logger.info("=== Finnhub News Collection ===")
    finnhub_news = collect_finnhub_news()

    # Summary
    logger.info("=" * 60)
    logger.info("Phase 3 Complete!")
    if gdelt_articles:
        logger.info(f"  GDELT: {len(gdelt_articles)} articles")
    if finnhub_news:
        logger.info(f"  Finnhub: {len(finnhub_news)} articles")


if __name__ == "__main__":
    main()
