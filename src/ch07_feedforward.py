"""
Implement the feed-forward network and layer normalization.
This file belongs to Chapter 8.
Run: python src/ch07_feedforward.py
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

# --- The Idea ---
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        C = config.n_embd
        # A small neural network applied to each token independently
        self.net = nn.Sequential(
            nn.Linear(C, 4 * C),
            nn.GELU(),
            nn.Linear(4 * C, C),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

# --- Demo ---
if __name__ == "__main__":
    torch.manual_seed(42)
    print("Chapter 8: Feed-Forward Layer and Layer Norm\n")

    ff = FeedForward()
    total_params = sum(p.numel() for p in ff.parameters())
    print(f"FeedForward parameters: {total_params:,}")
    print(f"  (C={config.n_embd} -> 4C={4*config.n_embd} -> C={config.n_embd})")

    B, T = 2, 10
    x = torch.randn(B, T, config.n_embd)
    out = ff(x)
    print(f"\nInput  shape: {x.shape}")
    print(f"Output shape: {out.shape}   (same shape as input)")

    print("\n--- GELU activation ---")
    print("GELU is like ReLU (zeros out negatives) but with a smooth curve.")
    sample = torch.tensor([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0])
    gelu_out = F.gelu(sample)
    relu_out = F.relu(sample)
    print(f"\n  Input : {sample.tolist()}")
    print(f"  GELU  : {[round(v, 3) for v in gelu_out.tolist()]}")
    print(f"  ReLU  : {relu_out.tolist()}")

    print("\n--- Layer Normalization ---")
    print("LayerNorm re-centers and re-scales values at each position.")

    # LayerNorm ensures each token's vector has mean 0 and variance 1
    # initially
    ln = nn.LayerNorm(config.n_embd)
    x_single = torch.randn(config.n_embd) * 10 + 5
    x_normed = ln(x_single)

    print(f"\nBefore LayerNorm: mean={x_single.mean():.2f}, "
          f"std={x_single.std():.2f}")
    print(f"After  LayerNorm: mean={x_normed.mean():.4f}, "
          f"std={x_normed.std():.4f}")
    print("(After normalization: mean ~0, std ~1)")

    print("\nFeed-forward and LayerNorm done! Ready for Chapter 9.")
