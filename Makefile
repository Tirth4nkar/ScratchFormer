.PHONY: all setup data features train predict test lint format clean

PYTHON = python
SRC    = src/my_project

## all        : Run full pipeline (data → features → train)
all: data features train

## setup      : Install dependencies
setup:
	pip install -r requirements.txt

## data       : Download / prepare raw data
data:
	$(PYTHON) $(SRC)/dataset.py

## features   : Build features from processed data
features:
	$(PYTHON) $(SRC)/features.py

## train      : Train model
train:
	$(PYTHON) $(SRC)/modeling/train.py

## predict    : Run inference with trained model
predict:
	$(PYTHON) $(SRC)/modeling/predict.py

## test       : Run test suite
test:
	pytest

## lint       : Lint source code
lint:
	ruff check src tests

## format     : Auto-format source code
format:
	black src tests

## clean      : Remove compiled files and caches
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +

help:
	@grep -E '^## ' Makefile | sed 's/## /  /'
