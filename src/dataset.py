"""
dataset.py — Scripts to load, download, or generate datasets.
"""
import pandas as pd
from loguru import logger
from my_project.config import RAW_DIR, PROCESSED_DIR


def load_raw(filename: str) -> pd.DataFrame:
    """Load a CSV from the raw data directory."""
    path = RAW_DIR / filename
    logger.info(f"Loading raw data from {path}")
    return pd.read_csv(path)


def save_processed(df: pd.DataFrame, filename: str) -> None:
    """Save a DataFrame to the processed data directory."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / filename
    df.to_csv(path, index=False)
    logger.info(f"Saved processed data to {path}")


if __name__ == "__main__":
    logger.info("Running dataset pipeline...")
    # TODO: add your data acquisition logic here
