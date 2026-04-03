"""
Phase 9: LLM Explainability Annotations
Generates structured Chain-of-Thought explanations for each trading week.
Reference: Yu et al. "Temporal Data Meets LLM" (EMNLP 2023)

Output: explanations/USDCNH_llm_explanations.jsonl

When a local LLM (e.g. Qwen2.5-14B-Instruct) is available the script calls
it via the transformers pipeline.  Otherwise it builds the structured prompts
and writes them to disk so they can be batch-processed later.
"""
import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import ALIGNED_DIR, EXPLANATIONS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Prompt template ──────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a senior FX analyst specialising in the USD/CNH pair. "
    "Given the market context below, predict the most likely direction "
    "of USD/CNH for the next trading day and explain your reasoning "
    "step by step."
)

def build_prompt(row: pd.Series, history_5d: pd.DataFrame) -> str:
    """Build a structured prompt from one aligned sample."""
    # Price context
    close = row["close"]
    ret_5d = (close / history_5d["close"].iloc[0] - 1) * 100 if len(history_5d) > 0 else 0

    lines = [
        "## Market Context",
        f"Date: {row.name.strftime('%Y-%m-%d')}",
        f"USD/CNH Close: {close:.4f}  (5-day change: {ret_5d:+.2f}%)",
    ]

    # PBOC midprice
    if pd.notna(row.get("pboc_midprice")):
        mid = row["pboc_midprice"]
        deviation = (close - mid) / mid * 100
        lines.append(f"PBOC Midprice: {mid:.4f}  (market deviation: {deviation:+.3f}%)")

    # CNH-CNY spread
    if pd.notna(row.get("cnh_cny_spread")):
        lines.append(f"CNH-CNY Spread: {row['cnh_cny_spread']:.4f}")

    # Technical snapshot
    tech_cols = [c for c in row.index if c.startswith("tech_")]
    if tech_cols:
        lines.append("\n## Technical Indicators")
        for c in tech_cols:
            if pd.notna(row[c]):
                label = c.replace("tech_", "").upper()
                lines.append(f"  {label}: {row[c]:.4f}")

    # Macro snapshot
    macro_cols = [c for c in row.index if c.startswith("macro_")]
    if macro_cols:
        lines.append("\n## Macroeconomic Background")
        for c in macro_cols:
            if pd.notna(row[c]):
                label = c.replace("macro_us_", "US ").replace("macro_cn_", "CN ").replace("_", " ").title()
                lines.append(f"  {label}: {row[c]:.2f}")

    # DXY / VIX
    if pd.notna(row.get("dxy_close")):
        lines.append(f"\n  DXY: {row['dxy_close']:.2f}")
    if pd.notna(row.get("vix_close")):
        lines.append(f"  VIX: {row['vix_close']:.2f}")

    # News
    if row.get("news_titles") and str(row["news_titles"]).strip():
        lines.append("\n## Recent News Headlines")
        for headline in str(row["news_titles"]).split(" | ")[:5]:
            lines.append(f"  - {headline}")

    lines.append(
        "\n## Task\n"
        "1. Summarise the key drivers for USD/CNH today.\n"
        "2. Identify the dominant macro theme (Fed policy, China growth, trade, capital flows).\n"
        "3. Predict the next-day direction: BULLISH (USD strengthens), BEARISH (USD weakens), or NEUTRAL.\n"
        "4. Assign a confidence score (0-100).\n"
        "5. Explain your reasoning in 2-3 sentences."
    )

    return "\n".join(lines)


def generate_explanations_offline(aligned: pd.DataFrame) -> list[dict]:
    """Generate prompts and placeholder explanations (no LLM call)."""
    records = []
    dates = aligned.index.tolist()

    for i, date in enumerate(dates):
        # Weekly: only process Fridays or last trading day of week
        if date.weekday() != 4 and i < len(dates) - 1:
            wd_next = dates[i + 1].weekday()
            wd_cur = date.weekday()
            if not (wd_next < wd_cur):  # not end-of-week
                continue

        row = aligned.loc[date]
        start = max(0, i - 5)
        history = aligned.iloc[start:i]

        prompt = build_prompt(row, history)

        # Determine actual direction for the record
        actual_dir = int(row.get("target_direction", 0))
        actual_ret = float(row.get("target_return", 0)) if pd.notna(row.get("target_return")) else None

        records.append({
            "date": date.strftime("%Y-%m-%d"),
            "prompt": prompt,
            "actual_direction": actual_dir,
            "actual_return": actual_ret,
            # Placeholders — to be filled by LLM
            "predicted_direction": None,
            "confidence": None,
            "explanation": None,
        })

    return records


def generate_explanations_llm(aligned: pd.DataFrame) -> list[dict]:
    """Generate explanations using a local LLM via transformers pipeline."""
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        logger.warning("transformers not installed, using offline mode")
        return generate_explanations_offline(aligned)

    # Try loading a small model; fall back to offline if unavailable
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # smallest viable
    try:
        logger.info(f"Loading LLM: {model_name}")
        generator = hf_pipeline(
            "text-generation",
            model=model_name,
            device=-1,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
        )
    except Exception as e:
        logger.warning(f"Failed to load LLM ({e}), using offline mode")
        return generate_explanations_offline(aligned)

    records = generate_explanations_offline(aligned)

    for i, rec in enumerate(records):
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": rec["prompt"]},
            ]
            output = generator(messages, max_new_tokens=300)
            text = output[0]["generated_text"]
            if isinstance(text, list):
                text = text[-1].get("content", "")

            rec["explanation"] = text

            # Try to parse direction and confidence from response
            text_lower = text.lower()
            if "bullish" in text_lower:
                rec["predicted_direction"] = 1
            elif "bearish" in text_lower:
                rec["predicted_direction"] = -1
            else:
                rec["predicted_direction"] = 0

            # Extract confidence (look for a number after "confidence")
            import re
            conf_match = re.search(r"confidence[:\s]*(\d+)", text_lower)
            if conf_match:
                rec["confidence"] = int(conf_match.group(1))

        except Exception as e:
            logger.warning(f"  LLM error for {rec['date']}: {e}")

        if (i + 1) % 20 == 0:
            logger.info(f"  Generated {i + 1}/{len(records)} explanations")

    return records


def main():
    EXPLANATIONS_DIR.mkdir(parents=True, exist_ok=True)

    aligned_path = ALIGNED_DIR / "USDCNH_daily_aligned.parquet"
    if not aligned_path.exists():
        logger.error(f"Aligned dataset not found: {aligned_path}. Run Phase 8 first.")
        sys.exit(1)

    aligned = pd.read_parquet(aligned_path)
    aligned.index = pd.to_datetime(aligned.index)
    logger.info(f"Loaded aligned dataset: {aligned.shape}")

    # Try LLM first, fall back to offline
    logger.info("=== Generating LLM Explanations ===")
    records = generate_explanations_llm(aligned)

    output_path = EXPLANATIONS_DIR / "USDCNH_llm_explanations.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_with_explanation = sum(1 for r in records if r.get("explanation"))
    logger.info("=" * 60)
    logger.info(f"Phase 9 Complete!")
    logger.info(f"  Total weekly prompts: {len(records)}")
    logger.info(f"  With LLM explanation: {n_with_explanation}")
    logger.info(f"  Saved → {output_path}")


if __name__ == "__main__":
    main()
