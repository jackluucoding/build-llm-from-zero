"""
ch08_transformer_block.py - The Transformer Block

We now have all the pieces. Time to assemble them.

One transformer block = Multi-Head Attention + Feed-Forward,
glued together with two important tricks:
  1. RESIDUAL CONNECTIONS: we ADD the input to the output.
     Instead of: output = layer(x)
     We do:      output = x + layer(x)
     This is like passing a "highway" next to every layer so
     information doesn't get lost as it flows deeper.

  2. PRE-LAYER NORM: we normalize BEFORE each sub-layer.
     This is the modern convention (used in GPT-2 and later).
     It stabilizes training at the start.

The full forward pass of one block:
    x = x + MultiHeadAttention(LayerNorm(x))
    x = x + FeedForward(LayerNorm(x))

This pattern is repeated n_layers times.

Run this file:
    python src/ch08_transformer_block.py
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import GPTConfig
from src.ch06_multihead_attention import MultiHeadAttention
from src.ch07_feedforward import FeedForward

config = GPTConfig()


class TransformerBlock(nn.Module):
    """
    One complete transformer block.

    Input : x of shape (B, T, C)
    Output: y of shape (B, T, C)  - same shape
    """

    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention()
        self.ff   = FeedForward()
        self.ln1  = nn.LayerNorm(config.n_embd)   # norm before attention
        self.ln2  = nn.LayerNorm(config.n_embd)   # norm before feed-forward

    def forward(self, x):
        # Residual connection around multi-head attention
        x = x + self.attn(self.ln1(x))

        # Residual connection around feed-forward
        x = x + self.ff(self.ln2(x))

        return x


if __name__ == "__main__":
    print("=" * 50)
    print("Chapter 8: The Transformer Block")
    print("=" * 50)

    block = TransformerBlock()

    total_params = sum(p.numel() for p in block.parameters())
    attn_params  = sum(p.numel() for p in block.attn.parameters())
    ff_params    = sum(p.numel() for p in block.ff.parameters())
    ln_params    = sum(p.numel() for p in block.ln1.parameters()) + \
                   sum(p.numel() for p in block.ln2.parameters())

    print(f"\nOne TransformerBlock parameters: {total_params:,}")
    print(f"  MultiHeadAttention : {attn_params:,}")
    print(f"  FeedForward        : {ff_params:,}")
    print(f"  LayerNorms (x2)    : {ln_params:,}")

    B, T = 2, 10
    x = torch.randn(B, T, config.n_embd)
    out = block(x)

    print(f"\nInput  shape: {x.shape}")
    print(f"Output shape: {out.shape}   (same as input)")

    # ------------------------------------------------------------------
    # Show the residual connection in action
    # ------------------------------------------------------------------
    print("\n--- Residual connection demonstration ---")
    print("The output is a MIX of the original input and what attention learned.")
    print("The input is never lost -- it always flows through.")

    with torch.no_grad():
        x_sample  = x[0, 0, :]
        attn_out  = block.attn(block.ln1(x[0:1]))[0, 0, :]
        final_out = block(x[0:1])[0, 0, :]

    print(f"\nOriginal token norm  : {x_sample.norm():.3f}")
    print(f"Attention output norm: {attn_out.norm():.3f}")
    print(f"After residual norm  : {final_out.norm():.3f}  (combined)")

    # ------------------------------------------------------------------
    # Stack multiple blocks
    # ------------------------------------------------------------------
    print("\n--- Stacking multiple blocks ---")

    n_layers = config.n_layers
    blocks = nn.Sequential(*[TransformerBlock() for _ in range(n_layers)])
    total_stack_params = sum(p.numel() for p in blocks.parameters())

    print(f"Stacking {n_layers} blocks:")
    print(f"  Params per block: {total_params:,}")
    print(f"  Total params    : {total_stack_params:,}  ({n_layers} x {total_params:,})")

    x = torch.randn(B, T, config.n_embd)
    out = blocks(x)
    print(f"\nInput  shape: {x.shape}")
    print(f"Output shape: {out.shape}  (unchanged after {n_layers} blocks)")

    print("\nTransformer block done! Ready for Chapter 9.")
