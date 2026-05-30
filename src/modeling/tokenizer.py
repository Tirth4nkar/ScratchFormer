"""
tokenizer.py — Tokenizer for text preprocessing.
"""

import os
from pathlib import Path
from typing import List, Dict
from config import config
import re
import torch


class Tokenizer:
    """Tokenizer class to convert text to token IDs and vice versa."""
    name = "scratchformer_tokenizer"
    
    def __init__(self, vocab_file: Path):
        self.vocab_file = vocab_file
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._load_vocab()
        
    def _load_vocab(self):
        if not self.vocab_file.exists():
            raise VocabularyError(f"Vocabulary file not found: {self.vocab_file}")
        
        with open(self.vocab_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                token = line.strip()
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token
                
    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)
    
    @property
    def __len__(self):
        return self.vocab_size
    
    def word_piece_tokenize(self, text: str) -> List[str]:
        """Splits text into word pieces using a simple regex-based approach."""
        return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    
    def encode(self, text: str) -> List[int]:
        """Tokenizes input text and converts to token IDs"""
        # Add special tokens and convert to IDs
        tokens = self.word_piece_tokenize(text)
        input_tokens = [config.tokens.bos_token] + tokens + [config.tokens.eos_token]
        embeddings = [self.token_to_id.get(token, self.token_to_id[config.tokens.unk_token]) for token in input_tokens]
        return {
            "tokens": input_tokens,
            "token_ids": torch.Tensor(embeddings)
        }
    
    def decode(self, token_ids: List[int], remove_special_tokens: bool = True) -> str:
        """Converts token IDs back to text"""
        tokens = [self.id_to_token.get(token_id, config.tokens.unk_token) for token_id in token_ids]
        if remove_special_tokens:
            tokens = [
                token for token in tokens if token not in [config.tokens.bos_token, config.tokens.eos_token, config.tokens.unk_token]
                ]
        return " ".join(tokens)
    
    
    def __getitem__(self, token: str) -> int:
        return self.token_to_id.get(token, self.token_to_id[config.tokens.unk_token])