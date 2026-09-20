"""
Introduce embedding tables and positional embeddings.
This file belongs to Chapter 5.
Run: python src/ch04_embeddings.py
"""
import torch
import torch.nn as nn

import os
import sys

# Let Python find the book's own modules in src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import GPTConfig

# Settings
config = GPTConfig()

# --- Demo ---
if __name__ == "__main__":
    torch.manual_seed(42)
    print("Chapter 5: Embeddings\n")

    print("--- 1. The embedding table ---")
    # Creates a lookup table mapping each vocab ID to a vector
    token_emb = nn.Embedding(config.vocab_size, config.n_embd)

    print(f"Embedding table shape: {token_emb.weight.shape}")
    print(f"  vocab_size = {config.vocab_size}  (one row per character)")
    print(f"  n_embd     = {config.n_embd}  (numbers per character)")

    print("\n--- 2. Looking up embeddings ---")
    B, T = 2, 5
    token_ids = torch.tensor([[3, 14, 7, 2, 50], [10, 22, 45, 1, 8]])

    # Convert token IDs into their corresponding embedding vectors
    token_embeddings = token_emb(token_ids)

    print(f"Input token_ids shape : {token_ids.shape}")
    print(f"Token embeddings shape: {token_embeddings.shape}")
    print(f"  (B={B}, T={T}, C={config.n_embd})")
    print(f"\nFirst token's embedding (first 8 values): "
          f"{token_embeddings[0, 0, :8].tolist()}")

    print("\n--- 3. Positional embeddings ---")
    # Creates a lookup table mapping each position to a vector
    pos_emb = nn.Embedding(config.block_size, config.n_embd)

    # Generate position indices 0, 1, 2, ..., T-1
    positions = torch.arange(T)
    print(f"Position indices: {positions.tolist()}")

    position_embeddings = pos_emb(positions)
    print(f"Position embeddings shape: {position_embeddings.shape}")

    print("\n--- 4. Combining token + position embeddings ---")
    # Add token and position embeddings so the model knows what and where it is
    x = token_embeddings + position_embeddings

    print(f"Final x shape: {x.shape}")
    print(f"  (B={B}, T={T}, C={config.n_embd})")
    print("\nThis tensor x is the input to the transformer blocks.")
    print(
        f"Each of the {B*T} slots has {config.n_embd} numbers describing it."
    )

    print("\n--- Summary ---")
    total_params = (config.vocab_size + config.block_size) * config.n_embd
    print(f"Token embedding parameters  : {config.vocab_size} x {config.n_embd} "
          f"= {config.vocab_size * config.n_embd:,}")
    print(f"Position embedding parameters: {config.block_size} x {config.n_embd} "
          f"= {config.block_size * config.n_embd:,}")
    print(f"Total embedding parameters   : {total_params:,}")
    print("\nEmbeddings done! Ready for Chapter 6.")
