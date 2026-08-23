"""
tokenizer.py — Tokenizer for text preprocessing.
"""

import os
from pathlib import Path
from typing import List, Dict
from config import config
import re
import torch
import utils.exceptions as E
from transformers import AutoTokenizer
from transformers.utils import HFValidationError
from huggingface_hub.utils import RepositoryNotFoundError
from tokenizers import (
    Tokenizer, 
    models, 
    trainers, 
    pre_tokenizers, 
    decoders
    )


class Tokenizer:
    """Tokenizer class to convert text to token IDs and vice versa."""
    name = "scratchformer_tokenizer"
    
    def __init__(self, model_card: str | None) -> None:
        self.base_model = model_card
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._load_vocab()
        self.instruction_template = config.tokens.enable_instruction_template
        
        if self.base_model is None:
            self.build_vocab()
            
        
    def _load_vocab(self):
        """Loads vocabulary from the specified model card using Hugging Face's AutoTokenizer."""
        try:
            hf_tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        except (RepositoryNotFoundError, HFValidationError, OSError) as e:
            raise E.VocabularyError(f"Could not load model card: {self.base_model}") from e

        vocab = hf_tokenizer.get_vocab() 

        for token, idx in vocab.items():
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
                
    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)
    
    @property
    def __len__(self):
        return self.vocab_size
    
    
    def build_vocab(self, training_data: List[str] = None) -> None:
        """Builds a vocabulary from the training data.
        
        Args:
            training_data: List of paths to .txt files used as training corpus.
        """
        if not training_data:
            raise E.VocabularyError("No training data provided for vocabulary building.")

        missing = [f for f in training_data if not Path(f).exists()]
        if missing:
            raise E.VocabularyError(f"Training files not found: {missing}")

        # Set up a fresh BPE tokenizer for training
        hf_tokenizer = Tokenizer(models.BPE(unk_token=self.unk_token))
        hf_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        hf_tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,  # e.g. [<unk>, <pad>, <bos>, <eos>]
            show_progress=True,
        )

        hf_tokenizer.train(files=training_data, trainer=trainer)

        # Populate this class's own token maps from the trained vocab
        vocab = hf_tokenizer.get_vocab()
        self.token_to_id = {}
        self.id_to_token = {}
        for token, idx in vocab.items():
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

        # Keep the underlying HF tokenizer around if you need .encode/.decode later
        self._trained_tokenizer = hf_tokenizer

    
    @staticmethod
    def positional_encoding(
        d_model: int, 
        dropout: float = 0.1, 
        max_len: int = 5000
        )-> torch.Tensor:
        """Generates positional encoding for input sequences."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        if dropout > 0:
            pe = torch.nn.Dropout(p=dropout)(pe)
        return pe
    
    @staticmethod
    def word_piece_tokenize(text: str) -> List[str]:
        """Splits text into word pieces using a simple regex-based approach."""
        return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

    @staticmethod
    def apply_chat_template(chat: List[dict]) -> str:
        """
            Apply a simple chat template to the input text; assumes chat is a list of messages with 'role' and 'content' 
            of the format [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]. 
            Specifically, it wraps each message with special tokens and formats for an instruction finetuned chat model, 
            it as: "<BOS> role content <EOS>". This helps the model understand the structure of the conversation and differentiate between user and assistant messages.
        """
        assert isinstance(chat, list), "Chat input must be a list of messages"
        for message in chat:
            assert "role" in message and "content" in message, "Each message must have 'role' and 'content' keys"
        return [f"{config.tokens.bos_token} {message['role']} {message['content']} {config.tokens.eos_token}" for message in chat]


    def encode(self, text: str, pad_token_id: int = 0) -> List[int]:
        """Tokenizes input text and converts to token IDs"""
        assert isinstance(text, str), "Input text must be a string"
        assert len(text) > 0, "Input text cannot be empty"
        assert isinstance(pad_token_id, int), "pad_token_id must be an integer"
        assert pad_token_id >= 0, "pad_token_id must be non-negative"
        
        if pad_token_id not in self.id_to_token:
            raise E.TokenizerError(f"pad_token_id {pad_token_id} is not in the vocabulary")
        
        # Add special tokens and convert to IDs
        tokens = Tokenizer.word_piece_tokenize(text)
        input_tokens = [config.tokens.bos_token] + tokens + [config.tokens.eos_token]
        embeddings = [self.token_to_id.get(token, self.token_to_id[config.tokens.unk_token]) for token in input_tokens]
        embeddings = embeddings[:config.model.block_size]  # Truncate to block size
        # Pad the sequence if it's shorter than the block size
        if len(embeddings) < config.model.block_size:
            embeddings.extend([pad_token_id] * (config.model.block_size - len(embeddings)))
        
        # add positional encoding
        positional_encodings = Tokenizer.positional_encoding(
            d_model=config.model.n_embed, 
            max_len=config.model.block_size
            )
        embeddings = torch.tensor(embeddings) + positional_encodings.squeeze(1)[:len(embeddings)]
        
        return {
            "tokens": input_tokens,
            "token_ids": embeddings,
            "text": text
        }
    
    def decode(self, token_ids: List[int], remove_special_tokens: bool = True) -> str:
        """Converts token IDs back to text"""
        tokens = [self.id_to_token.get(token_id, config.tokens.unk_token) for token_id in token_ids] + [config.tokens.eos_token, config.tokens.bos_token]
        if remove_special_tokens:
            tokens = [
                token for token in tokens if token not in [config.tokens.bos_token, config.tokens.eos_token, config.tokens.unk_token]
                ]
        return " ".join(tokens)
    
    
    def __getitem__(self, token: str) -> int:
        return self.token_to_id.get(token, self.token_to_id[config.tokens.unk_token])