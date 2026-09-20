"""
Implement multi-head self-attention.
This file belongs to Chapter 7.
Run: python src/ch06_multihead_attention.py
"""
import torch
import torch.nn as nn

import os
import sys

# Let Python find the book's own modules in src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import GPTConfig
from src.ch05_self_attention import SingleHeadAttention

# Settings
config = GPTConfig()

# --- The Idea ---
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = config.n_embd // config.n_heads

        # Create multiple independent attention heads
        self.heads = nn.ModuleList([
            SingleHeadAttention(head_size) for _ in range(config.n_heads)
        ])

        # Linear layer to project the concatenated head outputs back
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Run each head in parallel
        head_outputs = [h(x) for h in self.heads]

        # Concatenate outputs along the last dimension
        out = torch.cat(head_outputs, dim=-1)

        # Apply projection and dropout
        out = self.dropout(self.proj(out))
        return out

# --- Demo ---
if __name__ == "__main__":
    torch.manual_seed(42)
    print("Chapter 7: Multi-Head Attention\n")

    mha = MultiHeadAttention()

    total_params = sum(p.numel() for p in mha.parameters())
    print(f"Config: n_heads={config.n_heads}, "
          f"head_size={config.n_embd // config.n_heads}")
    print(f"MultiHeadAttention total parameters: {total_params:,}")

    head_params = sum(p.numel() for h in mha.heads for p in h.parameters())
    proj_params = sum(p.numel() for p in mha.proj.parameters())
    print(f"  From {config.n_heads} heads: {head_params:,}")
    print(f"  From output proj : {proj_params:,}")

    B, T = 2, 10
    x = torch.randn(B, T, config.n_embd)
    out = mha(x)

    print(f"\nInput  shape: {x.shape}")
    print(f"Output shape: {out.shape}   (same shape as input!)")

    print("\n--- Comparing one head vs multi-head ---")
    single_head = SingleHeadAttention(config.n_embd // config.n_heads)

    # Use no_grad to skip tracking operations since we don't need backprop
    with torch.no_grad():
        single_out = single_head(x)
        multi_out  = mha(x)

    print(f"Single head output  shape: {single_out.shape}  (head_size=32)")
    print(f"Multi-head output   shape: {multi_out.shape}  (C=128)")
    ratio = multi_out.shape[-1] // single_out.shape[-1]
    print(f"\nMulti-head output has {ratio}x "
          f"more channels - it sees {config.n_heads} perspectives at once.")

    print("\nMulti-head attention done! Ready for Chapter 8.")
