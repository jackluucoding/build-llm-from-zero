"""
Implement single-head self-attention.
This file belongs to Chapter 6.
Run: python src/ch05_self_attention.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

# Let Python find the book's own modules in src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import GPTConfig

# Settings
config = GPTConfig()
HEAD_SIZE = config.n_embd // config.n_heads

# --- The Idea ---
class SingleHeadAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        C = config.n_embd

        # Linear layers to compute query, key, and value vectors
        self.query = nn.Linear(C, head_size, bias=False)
        self.key   = nn.Linear(C, head_size, bias=False)
        self.value = nn.Linear(C, head_size, bias=False)

        # Lower triangular matrix to prevent attending to future tokens
        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(config.block_size, config.block_size)
            )
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Compute attention scores and scale them to keep variance stable
        scale = self.head_size ** -0.5
        scores = q @ k.transpose(-2, -1) * scale

        # Mask future tokens by setting their scores to negative infinity
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # Convert scores to probabilities and apply dropout
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # Compute final output by taking weighted sum of values
        out = weights @ v
        return out

# --- Demo ---
if __name__ == "__main__":
    torch.manual_seed(42)
    print("Chapter 6: Self-Attention (Single Head)\n")
    print(f"Config: n_embd={config.n_embd}, head_size={HEAD_SIZE}")

    head = SingleHeadAttention(HEAD_SIZE)
    total_params = sum(p.numel() for p in head.parameters())
    print(f"SingleHeadAttention parameters: {total_params:,}")

    B, T = 2, 10
    x = torch.randn(B, T, config.n_embd)
    out = head(x)
    print(f"\nInput shape : {x.shape}")
    print(f"Output shape: {out.shape}   (B, T, head_size)")

    print("\n--- Attention weights (what token 5 attends to) ---")
    # Disable gradient tracking since we're just analyzing weights
    with torch.no_grad():
        q = head.query(x)
        k = head.key(x)
        scale = HEAD_SIZE ** -0.5
        scores = q @ k.transpose(-2, -1) * scale
        scores = scores.masked_fill(head.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)

    w = weights[0, 5, :].tolist()
    print("Token 5 attends to tokens 0..5 (future tokens masked):")
    for i, wi in enumerate(w):
        bar = "#" * int(wi * 30)
        print(f"  token {i}: {wi:.3f}  {bar}")

    print("\nSelf-attention done! Ready for Chapter 7.")
