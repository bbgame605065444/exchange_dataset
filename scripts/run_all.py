"""
Run all data collection phases sequentially.
Usage: python scripts/run_all.py [--phases 1,2,3]
"""
import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent

PHASES = {
    1: ("phase1_price_technical.py", "Price + Technical Indicators"),
    2: ("phase2_macro.py", "Macroeconomic Data (FRED)"),
    3: ("phase3_news_en.py", "English News (GDELT + Finnhub)"),
    5: ("phase5_central_bank.py", "Central Bank Communications"),
    6: ("phase6_sentiment.py", "Sentiment Scoring"),
    7: ("phase7_charts.py", "K-line Chart Generation"),
    8: ("phase8_alignment.py", "Data Alignment & Assembly"),
}


def main():
    parser = argparse.ArgumentParser(description="Run CNH/USD dataset collection pipeline")
    parser.add_argument(
        "--phases", type=str, default=None,
        help="Comma-separated phase numbers to run (e.g., '1,2,3'). Default: all."
    )
    args = parser.parse_args()

    if args.phases:
        phase_nums = [int(x.strip()) for x in args.phases.split(",")]
    else:
        phase_nums = sorted(PHASES.keys())

    for num in phase_nums:
        if num not in PHASES:
            print(f"Unknown phase: {num}")
            continue
        script, desc = PHASES[num]
        print(f"\n{'='*60}")
        print(f"Phase {num}: {desc}")
        print(f"{'='*60}")
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / script)],
            cwd=str(SCRIPTS_DIR.parent),
        )
        if result.returncode != 0:
            print(f"Phase {num} failed with return code {result.returncode}")
            resp = input("Continue? [y/N] ")
            if resp.lower() != "y":
                sys.exit(1)

    print(f"\n{'='*60}")
    print("All phases completed!")


if __name__ == "__main__":
    main()
