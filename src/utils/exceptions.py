"""exceptions.py — Custom exceptions for tokenizer errors."""

class TokenizerError(Exception):
    """Custom exception for tokenizer errors."""
    def __init__(self, message: str):
        super().__init__(message)
        
    def __str__(self):
        return self.args[0]


class VocabularyError(TokenizerError):
    """Exception raised for vocabulary-related errors."""
    def __init__(self, message: str):
        super().__init__(f"VocabularyError: {message}")
        
    def __str__(self):
        return self.args[0]


class EncodingError(TokenizerError):
    """Exception raised for encoding errors."""
    def __init__(self, message: str):
        super().__init__(f"EncodingError: {message}")

    def __str__(self):
        return self.args[0]



class DecodingError(TokenizerError):
    """Exception raised for decoding errors."""
    def __init__(self, message: str):
        super().__init__(f"DecodingError: {message}")

    def __str__(self):
        return self.args[0]


class TokenizationError(TokenizerError):
    """Exception raised for tokenization errors."""
    def __init__(self, message: str):
        super().__init__(f"TokenizationError: {message}")

    def __str__(self):
        return self.args[0]