"""
ch06_multihead_attention.py - Multi-Head Attention

In Chapter 5 we built one attention head - one "reader" that looks at
the text and decides what's important.

But one reader might miss things! Different aspects of a sentence matter
for different reasons:
  - One head might learn to track who is speaking.
  - Another might learn to track what tense the verb is in.
  - Another might track rhyme patterns (in Shakespeare, that matters!).

Multi-head attention runs N attention heads IN PARALLEL,
then concatenates all their outputs and projects back to the
original size. It's like having N different readers, each
highlighting different things, then combining their notes.

Run this file:
    python src/ch06_multihead_attention.py
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import GPTConfig
from src.ch05_self_attention import SingleHeadAttention

config = GPTConfig()


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention: n_heads independent attention heads
    running in parallel, with their outputs concatenated and projected.

    Input : x of shape (B, T, C)
    Output: y of shape (B, T, C)  - same shape as input
    """

    def __init__(self):
        super().__init__()
        head_size = config.n_embd // config.n_heads   # 128 // 4 = 32

        # Create n_heads independent attention heads in a list.
        self.heads = nn.ModuleList([
            SingleHeadAttention(head_size)
            for _ in range(config.n_heads)
        ])

        # After concatenating all heads, we project back to n_embd.
        # n_heads * head_size = 4 * 32 = 128 = n_embd
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Run each head on the same input x.
        # Each head outputs (B, T, head_size).
        head_outputs = [h(x) for h in self.heads]

        # Concatenate along the last dimension (channels).
        # 4 heads x head_size(32) = 128 = n_embd
        out = torch.cat(head_outputs, dim=-1)   # (B, T, n_embd)

        # Final linear projection + dropout.
        out = self.dropout(self.proj(out))       # (B, T, n_embd)
        return out


if __name__ == "__main__":
    print("=" * 50)
    print("Chapter 6: Multi-Head Attention")
    print("=" * 50)

    mha = MultiHeadAttention()

    total_params = sum(p.numel() for p in mha.parameters())
    print(f"\nConfig: n_heads={config.n_heads}, head_size={config.n_embd // config.n_heads}")
    print(f"MultiHeadAttention total parameters: {total_params:,}")

    head_params = sum(p.numel() for h in mha.heads for p in h.parameters())
    proj_params  = sum(p.numel() for p in mha.proj.parameters())
    print(f"  From {config.n_heads} heads: {head_params:,}")
    print(f"  From output proj : {proj_params:,}")

    B, T = 2, 10
    x = torch.randn(B, T, config.n_embd)
    out = mha(x)

    print(f"\nInput  shape: {x.shape}")
    print(f"Output shape: {out.shape}   (same shape as input!)")

    print("\n--- Comparing one head vs multi-head ---")

    single_head = SingleHeadAttention(config.n_embd // config.n_heads)

    with torch.no_grad():
        single_out = single_head(x)
        multi_out  = mha(x)

    print(f"Single head output  shape: {single_out.shape}  (B, T, head_size=32)")
    print(f"Multi-head output   shape: {multi_out.shape}  (B, T, C=128)")
    print(f"\nMulti-head output has {multi_out.shape[-1] // single_out.shape[-1]}x "
          f"more channels - it sees {config.n_heads} perspectives at once.")

    print("\nMulti-head attention done! Ready for Chapter 7.")
