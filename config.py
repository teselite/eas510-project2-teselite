from pathlib import Path

RANDOM_SEED = 42
TEST_SIZE = 0.2
AMPLIFIED_QUANTILE = 0.80  # Top 20% of peak views = Amplified (1)

SAMPLE_DATA_PATH = Path("data/sample/youtube_sample.csv")
FULL_DATA_PATH = Path("data/raw/USvideos.csv")

SAMPLE_RESULTS_DIR = Path("results/sample")
FULL_RESULTS_DIR = Path("results/full")

LOGREG_PARAMS = {
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
}
