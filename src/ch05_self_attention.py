"""
ch05_self_attention.py - Self-Attention (Single Head)

Self-attention is the secret sauce of transformers.
Here's the big idea:

When reading the word "bank" in "I went to the river bank",
you mentally look back at "river" to understand what "bank" means.
Self-attention does exactly this - each token looks at all other
tokens and decides how much to "pay attention" to each one.

The mechanism uses three concepts: Query, Key, and Value.
Think of it like a library:
  - Query (Q): "I'm looking for information about rivers."
  - Key   (K): Each book's index card says what it's about.
  - Value (V): The actual content of each book.

You compare your Query to every Key, find the best matches,
then read those books (Values). The output is a weighted mix
of all the Values based on how well Queries matched Keys.

Run this file:
    python src/ch05_self_attention.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import GPTConfig

config = GPTConfig()
HEAD_SIZE = config.n_embd // config.n_heads  # = 128 // 4 = 32


class SingleHeadAttention(nn.Module):
    """
    One head of self-attention.

    Input : x of shape (B, T, C)   - a batch of token sequences
    Output: y of shape (B, T, head_size) - attended representations
    """

    def __init__(self, head_size):
        super().__init__()
        C = config.n_embd

        # Three linear projections: Q, K, V
        # Each takes the C-dimensional input and projects to head_size.
        # bias=False is conventional (matches GPT-2 style).
        self.query = nn.Linear(C, head_size, bias=False)
        self.key   = nn.Linear(C, head_size, bias=False)
        self.value = nn.Linear(C, head_size, bias=False)

        # The causal mask: a lower-triangular matrix of ones.
        # It prevents token i from attending to token j when j > i.
        # We use register_buffer so it is saved with the model but
        # NOT treated as a learnable parameter.
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.block_size, config.block_size))
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape

        # Step 1: Project inputs to Q, K, V
        q = self.query(x)   # (B, T, head_size)
        k = self.key(x)     # (B, T, head_size)
        v = self.value(x)   # (B, T, head_size)

        # Step 2: Compute attention scores
        # How well does each Query match each Key?
        # We divide by sqrt(head_size) to keep values in a stable range -
        # without this, large values would make softmax output near-zero
        # for all but the top entry (like a confidence dial turned too high).
        scale = HEAD_SIZE ** -0.5
        scores = q @ k.transpose(-2, -1) * scale   # (B, T, T)

        # Step 3: Apply the causal mask
        # Set future positions to -inf so softmax turns them to 0.
        # This ensures token at position t can only see positions 0..t.
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # Step 4: Convert scores to probabilities with softmax
        weights = F.softmax(scores, dim=-1)   # (B, T, T)  - rows sum to 1
        weights = self.dropout(weights)

        # Step 5: Weighted sum of values
        out = weights @ v   # (B, T, T) @ (B, T, head_size) = (B, T, head_size)

        return out


if __name__ == "__main__":
    print("=" * 50)
    print("Chapter 5: Self-Attention (Single Head)")
    print("=" * 50)

    print(f"\nConfig: n_embd={config.n_embd}, head_size={HEAD_SIZE}")

    head = SingleHeadAttention(HEAD_SIZE)
    total_params = sum(p.numel() for p in head.parameters())
    print(f"SingleHeadAttention parameters: {total_params:,}")

    # Create a fake batch: B=2 sequences, T=10 tokens, C=128 features
    B, T = 2, 10
    x = torch.randn(B, T, config.n_embd)

    out = head(x)
    print(f"\nInput shape : {x.shape}")
    print(f"Output shape: {out.shape}   (B, T, head_size)")

    # ------------------------------------------------------------------
    # Visualise the attention weights for one head on one sequence
    # ------------------------------------------------------------------
    print("\n--- Attention weights (what token 5 attends to) ---")

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

    print("\nSelf-attention done! Ready for Chapter 6.")
