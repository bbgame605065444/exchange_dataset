"""
Phase 5: Central Bank Communications
Sources: HuggingFace (FOMC labeled data), Fed website (statements), PBOC (reports)
Output: text/central_bank/*.jsonl
"""
import sys
import json
import logging

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from config import CENTRAL_BANK_DIR, START_DATE, END_DATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def collect_fomc_labeled():
    """Download FOMC Hawkish-Dovish labeled dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets not installed. Run: pip install datasets")
        return

    logger.info("Loading FOMC communication dataset from HuggingFace...")
    try:
        ds = load_dataset("gtfintechlab/fomc_communication", trust_remote_code=True)
        output_path = CENTRAL_BANK_DIR / "fomc_labeled.jsonl"
        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for split_name in ds:
                for row in ds[split_name]:
                    record = dict(row)
                    record["split"] = split_name
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
        logger.info(f"Saved {count} FOMC labeled sentences to {output_path}")
    except Exception as e:
        logger.warning(f"Failed to load FOMC dataset: {e}")


def collect_fed_statements():
    """Scrape FOMC statements from the Federal Reserve website."""
    import requests
    from bs4 import BeautifulSoup
    from datetime import datetime

    logger.info("Collecting Fed FOMC statements...")
    base_url = "https://www.federalreserve.gov"
    calendar_url = f"{base_url}/monetarypolicy/fomccalendars.htm"

    output_path = CENTRAL_BANK_DIR / "fed_statements.jsonl"
    statements = []

    try:
        resp = requests.get(calendar_url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # Find links to FOMC statements
        links = soup.find_all("a", href=True)
        statement_links = [
            link for link in links
            if "statement" in link.get("href", "").lower()
            and "monetarypolicy" in link.get("href", "").lower()
        ]

        logger.info(f"Found {len(statement_links)} FOMC statement links")

        for link in statement_links:
            href = link["href"]
            if not href.startswith("http"):
                href = base_url + href

            try:
                # Extract date from URL (format: monetary20210127a.htm)
                import re
                date_match = re.search(r"(\d{8})", href)
                if date_match:
                    date_str = date_match.group(1)
                    stmt_date = datetime.strptime(date_str, "%Y%m%d")
                    if stmt_date < datetime.strptime(START_DATE, "%Y-%m-%d"):
                        continue

                resp2 = requests.get(href, timeout=30)
                resp2.raise_for_status()
                soup2 = BeautifulSoup(resp2.text, "lxml")

                # Extract main content
                content_div = soup2.find("div", class_="col-xs-12")
                if content_div:
                    text = content_div.get_text(separator="\n", strip=True)
                else:
                    text = soup2.get_text(separator="\n", strip=True)

                statements.append({
                    "date": date_str if date_match else "",
                    "url": href,
                    "text": text[:5000],  # Limit to first 5000 chars
                    "type": "fomc_statement",
                })
                logger.info(f"  Collected statement: {date_str if date_match else href}")

            except Exception as e:
                logger.warning(f"  Failed to fetch {href}: {e}")

        with open(output_path, "w", encoding="utf-8") as f:
            for stmt in statements:
                f.write(json.dumps(stmt, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(statements)} Fed statements to {output_path}")

    except Exception as e:
        logger.warning(f"Failed to collect Fed statements: {e}")

    return statements


def collect_finnhub_economic_calendar():
    """Collect economic calendar events from Finnhub."""
    import os
    api_key = os.environ.get("FINNHUB_API_KEY", "")
    if not api_key:
        logger.warning("FINNHUB_API_KEY not set. Skipping economic calendar.")
        return

    try:
        import finnhub
    except ImportError:
        logger.error("finnhub-python not installed. Run: pip install finnhub-python")
        return

    client = finnhub.Client(api_key=api_key)
    output_path = CENTRAL_BANK_DIR.parent / "economic_calendar.jsonl"

    logger.info("Fetching Finnhub economic calendar...")
    try:
        # Finnhub free tier may limit calendar range
        calendar = client.calendar_economic(
            _from=START_DATE, to=END_DATE
        )
        events = calendar.get("economicCalendar", [])

        # Filter for US and CN events
        filtered = [
            e for e in events
            if e.get("country", "") in ("US", "CN", "")
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            for event in filtered:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(filtered)} economic calendar events (from {len(events)} total)")

    except Exception as e:
        logger.warning(f"Failed to fetch economic calendar: {e}")


def main():
    CENTRAL_BANK_DIR.mkdir(parents=True, exist_ok=True)

    # 5.1 FOMC labeled data from HuggingFace
    logger.info("=== FOMC Labeled Data ===")
    collect_fomc_labeled()

    # 5.3 Fed statements
    logger.info("=== Fed FOMC Statements ===")
    collect_fed_statements()

    # 5.6 Economic calendar
    logger.info("=== Economic Calendar ===")
    collect_finnhub_economic_calendar()

    logger.info("=" * 60)
    logger.info("Phase 5 Complete!")


if __name__ == "__main__":
    main()
