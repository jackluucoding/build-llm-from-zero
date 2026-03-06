"""
ch07_feedforward.py - Feed-Forward Layer and Layer Normalization

After attention, each token has gathered information from its neighbours.
But it still needs to THINK about what it learned.

The feed-forward layer does that "thinking". It's a small two-layer
neural network applied independently to each token:
  1. Expand: project from n_embd (128) to 4*n_embd (512)
  2. Activate: apply GELU - a smooth version of "ignore negatives"
  3. Compress: project back from 512 down to 128

Think of it like: attention lets you READ the other students' notes,
then the feed-forward layer is you PROCESSING those notes at your desk.

We also introduce LAYER NORMALIZATION, which keeps the numbers in a
stable range during training. Without it, values can blow up or shrink
to zero, making training unstable. Think of it as "re-centering" the
numbers after each step so nothing goes out of control.

Run this file:
    python src/ch07_feedforward.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import GPTConfig

config = GPTConfig()


class FeedForward(nn.Module):
    """
    Two-layer MLP applied to each token independently.

    Architecture:
        Linear(C -> 4C)  ->  GELU  ->  Linear(4C -> C)  ->  Dropout

    Input : x of shape (B, T, C)
    Output: y of shape (B, T, C)  - same shape as input
    """

    def __init__(self):
        super().__init__()
        C = config.n_embd
        self.net = nn.Sequential(
            nn.Linear(C, 4 * C),    # expand
            nn.GELU(),              # activation (smooth ReLU)
            nn.Linear(4 * C, C),   # compress back
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)   # applied identically to every token position


if __name__ == "__main__":
    print("=" * 50)
    print("Chapter 7: Feed-Forward Layer and Layer Norm")
    print("=" * 50)

    ff = FeedForward()
    total_params = sum(p.numel() for p in ff.parameters())
    print(f"\nFeedForward parameters: {total_params:,}")
    print(f"  (C={config.n_embd} -> 4C={4*config.n_embd} -> C={config.n_embd})")

    B, T = 2, 10
    x = torch.randn(B, T, config.n_embd)
    out = ff(x)
    print(f"\nInput  shape: {x.shape}")
    print(f"Output shape: {out.shape}   (same shape as input)")

    # ------------------------------------------------------------------
    # Demo: GELU activation
    # ------------------------------------------------------------------
    print("\n--- GELU activation ---")
    print("GELU is like ReLU (zeros out negatives) but with a smooth curve.")
    print("Negative inputs get mostly zeroed, positive inputs pass through.")

    sample = torch.tensor([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0])
    gelu_out = F.gelu(sample)
    relu_out = F.relu(sample)
    print(f"\n  Input : {sample.tolist()}")
    print(f"  GELU  : {[round(v, 3) for v in gelu_out.tolist()]}")
    print(f"  ReLU  : {relu_out.tolist()}")

    # ------------------------------------------------------------------
    # Demo: Layer Normalization
    # ------------------------------------------------------------------
    print("\n--- Layer Normalization ---")
    print("LayerNorm re-centers and re-scales the values at each token position.")
    print("Goal: keep the mean near 0 and std near 1, so training stays stable.")

    ln = nn.LayerNorm(config.n_embd)
    x_single = torch.randn(config.n_embd) * 10 + 5   # intentionally large values
    x_normed = ln(x_single)

    print(f"\nBefore LayerNorm: mean={x_single.mean():.2f}, std={x_single.std():.2f}")
    print(f"After  LayerNorm: mean={x_normed.mean():.4f}, std={x_normed.std():.4f}")
    print("(After normalization: mean ~0, std ~1)")

    print("\nFeed-forward and LayerNorm done! Ready for Chapter 8.")
