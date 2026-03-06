"""
ch04_embeddings.py - Token and Positional Embeddings

After tokenization, we have a list of integers like [20, 17, 30, 30, 33].
But integers by themselves carry no useful information for a neural network.
We can't do meaningful math on them as-is.

The solution: an EMBEDDING TABLE.

Think of it like a dictionary. For each token ID, the embedding table
stores a row of numbers (called an "embedding vector").
These numbers are learned during training - the model figures out
what numbers best represent each character.

We also need POSITIONAL EMBEDDINGS because a transformer treats all
tokens the same by default - it has no built-in sense of order.
We add a second embedding that encodes each position (0, 1, 2, ...).

Run this file:
    python src/ch04_embeddings.py
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import GPTConfig

config = GPTConfig()

print("=" * 50)
print("Chapter 4: Embeddings")
print("=" * 50)

# ------------------------------------------------------------------
# 1. The embedding table
# ------------------------------------------------------------------
print("\n--- 1. The embedding table ---")

# nn.Embedding(vocab_size, n_embd) creates a table with:
#   - vocab_size rows (one per token)
#   - n_embd columns (the numbers that represent that token)
#
# When we pass a token ID, it returns the corresponding row.

token_emb = nn.Embedding(config.vocab_size, config.n_embd)

print(f"Embedding table shape: {token_emb.weight.shape}")
print(f"  vocab_size = {config.vocab_size}  (one row per character)")
print(f"  n_embd     = {config.n_embd}  (numbers per character)")

# ------------------------------------------------------------------
# 2. Looking up embeddings
# ------------------------------------------------------------------
print("\n--- 2. Looking up embeddings ---")

# Simulate a small batch: 2 sequences (B=2), each with 5 tokens (T=5).
B, T = 2, 5
# Make up some token IDs (values between 0 and vocab_size-1)
token_ids = torch.tensor([[3, 14, 7, 2, 50],
                           [10, 22, 45, 1, 8]])   # shape: (B, T)

# Look up embeddings: each integer becomes a row of n_embd numbers.
token_embeddings = token_emb(token_ids)   # shape: (B, T, C)

print(f"Input token_ids shape : {token_ids.shape}")    # (2, 5)
print(f"Token embeddings shape: {token_embeddings.shape}")  # (2, 5, 128)
print(f"  (B={B}, T={T}, C={config.n_embd})")
print(f"\nFirst token's embedding (first 8 values): {token_embeddings[0, 0, :8].tolist()}")

# ------------------------------------------------------------------
# 3. Positional embeddings
# ------------------------------------------------------------------
print("\n--- 3. Positional embeddings ---")

# A second embedding table - same idea, but indexed by POSITION.
# Position 0 gets one row of numbers, position 1 gets another, etc.

pos_emb = nn.Embedding(config.block_size, config.n_embd)

# Create position indices: [0, 1, 2, ..., T-1]
positions = torch.arange(T)   # shape: (T,)
print(f"Position indices: {positions.tolist()}")

position_embeddings = pos_emb(positions)   # shape: (T, C)
print(f"Position embeddings shape: {position_embeddings.shape}")  # (5, 128)

# ------------------------------------------------------------------
# 4. Combining token + position embeddings
# ------------------------------------------------------------------
print("\n--- 4. Combining token + position embeddings ---")

# We add the two embeddings together.
# Each token now "knows" both WHAT it is and WHERE it appears.
#
# token_embeddings has shape (B, T, C)
# position_embeddings has shape (T, C)
# PyTorch broadcasts the (T, C) shape to match (B, T, C) automatically.

x = token_embeddings + position_embeddings   # shape: (B, T, C)

print(f"Final x shape: {x.shape}")
print(f"  (B={B}, T={T}, C={config.n_embd})")
print(f"\nThis tensor x is the input to the transformer blocks.")
print(f"Each of the {B*T} token slots has {config.n_embd} numbers describing it.")

# ------------------------------------------------------------------
# 5. Summary
# ------------------------------------------------------------------
print("\n--- Summary ---")
total_params = (config.vocab_size * config.n_embd) + (config.block_size * config.n_embd)
print(f"Token embedding parameters  : {config.vocab_size} x {config.n_embd} = {config.vocab_size * config.n_embd:,}")
print(f"Position embedding parameters: {config.block_size} x {config.n_embd} = {config.block_size * config.n_embd:,}")
print(f"Total embedding parameters   : {total_params:,}")
print("\nEmbeddings done! Ready for Chapter 5.")
