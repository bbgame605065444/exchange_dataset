"""
Phase 4: Chinese News Collection
Sources: Sina Finance Forex, Eastmoney Forex, official media (Xinhua/People's Daily)
Output: text/news_cn/sina_forex.jsonl, text/news_cn/eastmoney_forex.jsonl

All HTTP request intervals are jittered with Gaussian noise to avoid
fingerprinting.
"""
import sys
import json
import re
import time
import random
import logging
from datetime import datetime, timedelta

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import requests
from bs4 import BeautifulSoup

from config import NEWS_CN_DIR, START_DATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Gaussian noise sleep ─────────────────────────────────────────────
def gaussian_sleep(mean: float = 2.0, std: float = 0.6):
    t = max(0.2, min(mean * 3, random.gauss(mean, std)))
    time.sleep(t)


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


# ====================================================================
# Sina Finance Forex
# ====================================================================
def collect_sina_forex(max_pages: int = 30) -> list[dict]:
    """Scrape Sina Finance forex news list pages."""
    logger.info("Collecting Sina Finance forex news...")
    base_url = "https://finance.sina.com.cn/forex/"
    # Sina uses roll pages like: https://finance.sina.com.cn/roll/index.d.html?cid=56694&page=1
    roll_url = "https://finance.sina.com.cn/roll/index.d.html"
    articles = []

    for page in range(1, max_pages + 1):
        gaussian_sleep(2.0, 0.6)
        try:
            params = {"cid": "56694", "page": str(page)}
            resp = requests.get(roll_url, params=params, headers=HEADERS, timeout=20)
            resp.encoding = "utf-8"
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")

            # Find news list items
            items = soup.select("ul.list_009 li") or soup.select(".d_list_txt li") or soup.find_all("li")
            found = 0
            for item in items:
                a_tag = item.find("a", href=True)
                if not a_tag:
                    continue
                title = a_tag.get_text(strip=True)
                url = a_tag["href"]
                if not title or len(title) < 5:
                    continue
                # Extract date from span if available
                span = item.find("span")
                date_str = span.get_text(strip=True) if span else ""

                # Check if within date range
                if date_str:
                    try:
                        dt = _parse_cn_date(date_str)
                        if dt and dt < datetime.strptime(START_DATE, "%Y-%m-%d"):
                            logger.info(f"  Reached start date boundary at page {page}")
                            return articles
                    except Exception:
                        pass

                articles.append({
                    "title": title,
                    "url": url,
                    "date": date_str,
                    "source": "sina_finance",
                    "language": "zh",
                })
                found += 1

            logger.info(f"  Page {page}: {found} articles (total: {len(articles)})")
            if found == 0:
                break

        except Exception as e:
            logger.warning(f"  Sina page {page} error: {e}")
            if "403" in str(e) or "Proxy" in str(e):
                logger.warning("  Connection blocked, stopping Sina collection")
                break

    return articles


# ====================================================================
# Eastmoney Forex
# ====================================================================
def collect_eastmoney_forex(max_pages: int = 30) -> list[dict]:
    """Scrape Eastmoney forex news via their JS API."""
    logger.info("Collecting Eastmoney forex news...")
    # Eastmoney uses a JSON API for news lists
    api_url = "https://newsapi.eastmoney.com/kuaixun/v1/getlist_102_ajaxResult_50_{page}_.html"
    articles = []

    for page in range(1, max_pages + 1):
        gaussian_sleep(1.5, 0.4)
        try:
            url = api_url.format(page=page)
            resp = requests.get(url, headers=HEADERS, timeout=20)
            resp.encoding = "utf-8"

            # Eastmoney returns JSONP sometimes, strip callback
            text = resp.text.strip()
            if text.startswith("var "):
                # Extract JSON from: var xxx = {...};
                match = re.search(r"=\s*(\{.*\})\s*;?$", text, re.DOTALL)
                if match:
                    text = match.group(1)
            elif text.startswith("(") or text.startswith("callback"):
                match = re.search(r"\((\{.*\})\)", text, re.DOTALL)
                if match:
                    text = match.group(1)

            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                # Try parsing as HTML instead
                soup = BeautifulSoup(resp.text, "lxml")
                items = soup.select(".news-item") or soup.find_all("li")
                for item in items:
                    a_tag = item.find("a", href=True)
                    if a_tag:
                        title = a_tag.get_text(strip=True)
                        if title and len(title) > 5 and _is_forex_related(title):
                            articles.append({
                                "title": title,
                                "url": a_tag["href"],
                                "source": "eastmoney",
                                "language": "zh",
                            })
                continue

            # Parse JSON response
            news_list = data.get("LivesList", data.get("data", data.get("list", [])))
            if not isinstance(news_list, list):
                news_list = []

            found = 0
            for item in news_list:
                title = item.get("title", "") or item.get("Title", "")
                digest = item.get("digest", "") or item.get("Digest", "")
                url = item.get("url", "") or item.get("Url", "")
                date_str = item.get("showtime", "") or item.get("Date", "")

                if not title or len(title) < 5:
                    continue
                if not _is_forex_related(title + digest):
                    continue

                articles.append({
                    "title": title,
                    "summary": digest[:300] if digest else "",
                    "url": url,
                    "date": date_str,
                    "source": "eastmoney",
                    "language": "zh",
                })
                found += 1

            logger.info(f"  Page {page}: {found} articles (total: {len(articles)})")
            if found == 0 and page > 3:
                break

        except Exception as e:
            logger.warning(f"  Eastmoney page {page} error: {e}")
            if "403" in str(e) or "Proxy" in str(e):
                logger.warning("  Connection blocked, stopping Eastmoney collection")
                break

    return articles


# ====================================================================
# Helpers
# ====================================================================
_FOREX_KEYWORDS = [
    "人民币", "汇率", "外汇", "美元", "离岸", "在岸", "中间价",
    "央行", "PBOC", "外储", "外汇储备", "跨境", "贬值", "升值",
    "贸易战", "关税", "结售汇", "逆周期", "资本流", "CNH", "CNY",
]

def _is_forex_related(text: str) -> bool:
    """Check if text is forex/RMB related."""
    return any(kw in text for kw in _FOREX_KEYWORDS)


def _parse_cn_date(date_str: str) -> datetime | None:
    """Try multiple Chinese date formats."""
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
                "%m月%d日 %H:%M", "%Y年%m月%d日"]:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def _save_jsonl(articles: list[dict], path):
    with open(path, "w", encoding="utf-8") as f:
        for a in articles:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")


# ====================================================================
# Main
# ====================================================================
def main():
    NEWS_CN_DIR.mkdir(parents=True, exist_ok=True)

    # Sina
    sina = collect_sina_forex(max_pages=30)
    if sina:
        _save_jsonl(sina, NEWS_CN_DIR / "sina_forex.jsonl")
        logger.info(f"Saved {len(sina)} Sina articles")
    else:
        logger.warning("No Sina articles collected")

    # Eastmoney
    eastmoney = collect_eastmoney_forex(max_pages=30)
    if eastmoney:
        _save_jsonl(eastmoney, NEWS_CN_DIR / "eastmoney_forex.jsonl")
        logger.info(f"Saved {len(eastmoney)} Eastmoney articles")
    else:
        logger.warning("No Eastmoney articles collected")

    logger.info("=" * 60)
    logger.info(f"Phase 4 Complete! Sina: {len(sina)}, Eastmoney: {len(eastmoney)}")


if __name__ == "__main__":
    main()
