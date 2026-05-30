"""
config.py — Project-wide model settings.
All modules should import from here rather than hard-coding paths.
"""
import os
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelConfig:
    name: str = "scratchformer"
    vocab_size: int = 128
    vocab_file: Path = Path("data/vocab.txt")
    n_head: int = 8
    n_layer: int = 6
    n_embed: int = 256
    block_size: int = 128
    batch_size: int = 64
    max_iters: int = 5000
    

@dataclass(frozen=True)
class PathsConfig:
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    logs_dir: Path = Path("logs")
    figures_dir: Path = Path("figures")
    raw_dir: Path = data_dir / "raw"
    processed_dir: Path = data_dir / "processed"
    
    
@dataclass(frozen=True)
class TokensConfig:
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    bos_token: str = "<BOS>"
    eos_token: str = "<EOS>"
    mask_token: str = "<MASK>"
    cls_token: str = "<CLS>"
    sep_token: str = "<SEP>"
    
    
def get_config():
    return {
        "model": ModelConfig(),
        "paths": PathsConfig(),
        "tokens": TokensConfig(),
    }


config = get_config()
model_config = config["model"]
paths_config = config["paths"]
tokens_config = config["tokens"]