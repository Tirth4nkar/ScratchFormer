"""
train.py — Model training script.
"""
from loguru import logger
from my_project.config import MODELS_DIR, PROCESSED_DIR


def train():
    logger.info("Starting model training...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # TODO: load data, build/train model, serialize to MODELS_DIR


if __name__ == "__main__":
    train()
