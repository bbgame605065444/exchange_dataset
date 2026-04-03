"""
Phase 6: Sentiment Scoring
Sources: FinBERT (English), FOMC-RoBERTa (central bank)
Output: sentiment/*.parquet
"""
import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import pandas as pd

from config import NEWS_EN_DIR, CENTRAL_BANK_DIR, SENTIMENT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file into a list of dicts."""
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def score_english_sentiment():
    """Score English news texts with FinBERT."""
    try:
        from transformers import pipeline
    except ImportError:
        logger.error("transformers not installed. Run: pip install transformers torch")
        return

    # Load English news
    texts = []
    sources = []

    for fname in ["gdelt_cnhusd.jsonl", "finnhub_forex.jsonl"]:
        records = load_jsonl(NEWS_EN_DIR / fname)
        for r in records:
            text = r.get("title", "") or r.get("headline", "") or ""
            summary = r.get("summary", "") or r.get("seendate", "") or ""
            combined = f"{text}. {summary}".strip()
            if combined and len(combined) > 10:
                texts.append(combined[:512])  # FinBERT max length
                sources.append(fname)

    if not texts:
        logger.warning("No English texts to score")
        return

    logger.info(f"Scoring {len(texts)} English texts with FinBERT...")
    try:
        classifier = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            truncation=True,
            max_length=512,
            device=-1,  # CPU; change to 0 for GPU
        )

        # Batch processing
        batch_size = 32
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = classifier(batch)
            results.extend(batch_results)
            if (i // batch_size) % 10 == 0:
                logger.info(f"  Processed {i + len(batch)}/{len(texts)}")

        # Build DataFrame
        df = pd.DataFrame({
            "text": texts,
            "source": sources,
            "label": [r["label"] for r in results],
            "score": [r["score"] for r in results],
        })

        SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(SENTIMENT_DIR / "en_sentiment.parquet")
        logger.info(f"Saved English sentiment: {len(df)} records")
        logger.info(f"  Distribution: {df['label'].value_counts().to_dict()}")

    except Exception as e:
        logger.error(f"FinBERT scoring failed: {e}")


def score_central_bank_hawkdove():
    """Score central bank texts for hawkish/dovish tone."""
    try:
        from transformers import pipeline
    except ImportError:
        logger.error("transformers not installed")
        return

    # Load FOMC labeled data and Fed statements
    texts = []
    fomc_records = load_jsonl(CENTRAL_BANK_DIR / "fomc_labeled.jsonl")
    fed_records = load_jsonl(CENTRAL_BANK_DIR / "fed_statements.jsonl")

    for r in fed_records:
        text = r.get("text", "")
        if text and len(text) > 20:
            # Take first 512 chars of each statement
            texts.append({"text": text[:512], "date": r.get("date", ""), "type": "fed_statement"})

    if not texts:
        logger.warning("No central bank texts to score")
        return

    logger.info(f"Scoring {len(texts)} central bank texts...")
    try:
        # Use a zero-shot classifier as fallback
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1,
        )

        labels = ["hawkish", "dovish", "neutral"]
        results = []
        for i, item in enumerate(texts):
            try:
                result = classifier(item["text"], labels)
                item["hawk_dove_label"] = result["labels"][0]
                item["hawk_dove_scores"] = dict(zip(result["labels"], result["scores"]))
                results.append(item)
            except Exception as e:
                logger.warning(f"  Error scoring text {i}: {e}")

            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i + 1}/{len(texts)}")

        if results:
            df = pd.DataFrame(results)
            SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)
            df.to_parquet(SENTIMENT_DIR / "cb_hawkdove.parquet")
            logger.info(f"Saved central bank hawk/dove: {len(df)} records")

    except Exception as e:
        logger.error(f"Central bank scoring failed: {e}")


def main():
    SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=== English News Sentiment (FinBERT) ===")
    score_english_sentiment()

    logger.info("=== Central Bank Hawk/Dove ===")
    score_central_bank_hawkdove()

    logger.info("=" * 60)
    logger.info("Phase 6 Complete!")


if __name__ == "__main__":
    main()
