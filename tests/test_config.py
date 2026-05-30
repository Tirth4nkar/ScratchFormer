"""Basic smoke tests for project configuration."""
from my_project.config import ROOT_DIR, DATA_DIR, MODELS_DIR


def test_root_dir_exists():
    assert ROOT_DIR.exists()


def test_data_subdirs_defined():
    assert DATA_DIR.name == "data"


def test_models_dir_defined():
    assert MODELS_DIR.name == "models"
