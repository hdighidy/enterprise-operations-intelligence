from pathlib import Path


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"


# ============================================================
# Dataset Configuration
# ============================================================

RANDOM_SEED = 42

HISTORICAL_START_YEAR = 2017
HISTORICAL_END_YEAR = 2026


# ============================================================
# Master Data Sizes
# ============================================================

NUM_MATERIALS = 250
NUM_SUPPLIERS = 350
NUM_EMPLOYEES = 2_000
NUM_EQUIPMENT = 1_000


# ============================================================
# Data Generation Helpers
# ============================================================

def create_directories() -> None:
    """Create required project directories."""

    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        METADATA_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)