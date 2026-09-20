"""
Combine attention and feed-forward into a transformer block.
This file belongs to Chapter 9.
Run: python src/ch08_transformer_block.py
"""
import torch
import torch.nn as nn
import os
import sys

# Let Python find the book's own modules in src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import GPTConfig
from src.ch06_multihead_attention import MultiHeadAttention
from src.ch07_feedforward import FeedForward

# Settings
config = GPTConfig()

# --- The Idea ---
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention()
        self.ff   = FeedForward()
        # Layer normalization applied before attention and feed-forward
        self.ln1  = nn.LayerNorm(config.n_embd)
        self.ln2  = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        # The + creates a residual connection, adding new info
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

# --- Demo ---
if __name__ == "__main__":
    torch.manual_seed(42)
    print("Chapter 9: The Transformer Block\n")

    block = TransformerBlock()
    total_params = sum(p.numel() for p in block.parameters())
    attn_params  = sum(p.numel() for p in block.attn.parameters())
    ff_params    = sum(p.numel() for p in block.ff.parameters())
    ln_params    = sum(p.numel() for p in block.ln1.parameters()) + \
                   sum(p.numel() for p in block.ln2.parameters())

    print(f"One TransformerBlock parameters: {total_params:,}")
    print(f"  MultiHeadAttention : {attn_params:,}")
    print(f"  FeedForward        : {ff_params:,}")
    print(f"  LayerNorms (x2)    : {ln_params:,}")

    B, T = 2, 10
    x = torch.randn(B, T, config.n_embd)
    out = block(x)
    print(f"\nInput  shape: {x.shape}")
    print(f"Output shape: {out.shape}   (same as input)")

    print("\n--- Residual connection demonstration ---")
    print("The input is never lost -- it always flows through.")

    with torch.no_grad():
        x_sample  = x[0, 0, :]
        attn_out  = block.attn(block.ln1(x[0:1]))[0, 0, :]
        final_out = block(x[0:1])[0, 0, :]

    print(f"\nOriginal token norm  : {x_sample.norm():.3f}")
    print(f"Attention output norm: {attn_out.norm():.3f}")
    print(f"After residual norm  : {final_out.norm():.3f}  (combined)")

    print("\n--- Stacking multiple blocks ---")
    n_layers = config.n_layers
    blocks = nn.Sequential(*[TransformerBlock() for _ in range(n_layers)])
    total_stack_params = sum(p.numel() for p in blocks.parameters())

    print(f"Stacking {n_layers} blocks:")
    print(f"  Params per block: {total_params:,}")
    print(
        f"  Total params    : {total_stack_params:,}  "
        f"({n_layers} x {total_params:,})"
    )

    x = torch.randn(B, T, config.n_embd)
    out = blocks(x)
    print(f"\nInput  shape: {x.shape}")
    print(f"Output shape: {out.shape}  (unchanged after {n_layers} blocks)")

    print("\nTransformer block done! Ready for Chapter 10.")
