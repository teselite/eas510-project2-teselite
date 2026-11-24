import argparse
import config
from src.pipeline import run_baseline


def main():
    parser = argparse.ArgumentParser(description="EAS 510 Project 2: Why Aren't You Famous Yet?")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run on the full dataset in data/raw/USvideos.csv instead of the bundled sample.",
    )
    args = parser.parse_args()

    if args.full:
        run_baseline(config.FULL_DATA_PATH, config.FULL_RESULTS_DIR, mode="full")
    else:
        run_baseline(config.SAMPLE_DATA_PATH, config.SAMPLE_RESULTS_DIR, mode="sample")


if __name__ == "__main__":
    main()
