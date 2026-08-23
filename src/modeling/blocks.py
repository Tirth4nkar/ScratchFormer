import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor



class EmbeddingBlock(nn.Module):
    """Embedding block to convert input tokens into dense vectors."""
    def __init__(self, input_dim, vocab_size, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        return x



class SelfAttentionBlock(nn.Module):
    """Self-attention block to capture dependencies between tokens in a sequence."""
    def __init__(self, embed_dim, num_heads, mask=None, apply_mask=False)->None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.Q = nn.Linear(embed_dim, embed_dim)
        self.K = nn.Linear(embed_dim, embed_dim)
        self.V = nn.Linear(embed_dim, embed_dim)
        self.apply_mask = apply_mask

    def forward(self, x):
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)
        attn_output, _ = F.multi_head_attention_forward(
            query=Q,
            key=K,
            value=V,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads
        )
        if self.apply_mask:
            attn_output = attn_output.masked_fill(self.attention.attn_mask == float('-inf'), 0)
        return attn_output


class CrossAttentionBlock(nn.Module):
    """Cross-attention block: queries attend to a different sequence (key/value source)."""
    def __init__(self, embed_dim, num_heads, dropout=0.0) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.Q = nn.Linear(embed_dim, embed_dim)
        self.K = nn.Linear(embed_dim, embed_dim)
        self.V = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query_input, kv_input, key_padding_mask=None):
        """
        query_input: (B, T_q, embed_dim)  -- e.g. decoder hidden states
        kv_input:    (B, T_kv, embed_dim) -- e.g. encoder output
        key_padding_mask: (B, T_kv) bool, True at positions to IGNORE (padding)
        """
        B, Tq, _ = query_input.shape
        Tkv = kv_input.shape[1]

        Q = self.Q(query_input)
        K = self.K(kv_input)
        V = self.V(kv_input)

        # split into heads: (B, num_heads, T, head_dim)
        Q = Q.view(B, Tq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Tkv, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Tkv, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if key_padding_mask is not None:
            # expand to (B, 1, 1, T_kv) so it broadcasts over heads and query positions
            attn_mask = key_padding_mask[:, None, None, :]

        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=~attn_mask if attn_mask is not None else None,  # SDPA wants True=keep
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,  # cross-attention is never causal by default
        )

        # merge heads back: (B, Tq, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Tq, self.embed_dim)
        return self.out_proj(attn_output)


class FeedForwardBlock(nn.Module):
    """Feed-forward block to apply non-linear transformations to the input."""
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x
    
class LayerNormBlock(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps)

    def forward(self, x):
        return self.layer_norm(x)