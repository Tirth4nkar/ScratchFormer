"""
predict.py — Model inference script.
"""
from loguru import logger
from my_project.config import MODELS_DIR


def predict(input_data):
    logger.info("Running inference...")
    # TODO: load model from MODELS_DIR, run predictions
    raise NotImplementedError


if __name__ == "__main__":
    predict(None)
