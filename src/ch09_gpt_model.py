"""
ch09_gpt_model.py - The Full GPT Model

We now assemble the complete model by stacking all the pieces:

  1. TOKEN EMBEDDING  - maps token IDs to vectors
  2. POSITION EMBEDDING - adds position information
  3. N TRANSFORMER BLOCKS - processes the sequence
  4. FINAL LAYER NORM  - stabilizes the output
  5. LM HEAD (linear layer) - maps back to vocabulary scores

The output is called "logits": one score per vocabulary token
at each position. The highest score = the model's best guess
for what comes next.

This is essentially the same architecture as GPT-2, just smaller.
Our model has ~825K parameters. GPT-2 small has 117M. GPT-4 is
estimated at ~1.8 trillion. Same architecture, very different scale.

Run this file:
    python src/ch09_gpt_model.py
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import GPTConfig
from src.ch08_transformer_block import TransformerBlock

config = GPTConfig()


class GPT(nn.Module):
    """
    A GPT-style language model.

    Input : token_ids of shape (B, T)  - integers in [0, vocab_size)
    Output: logits    of shape (B, T, vocab_size)  - one score per token
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        # Token embedding table: vocab_size rows, n_embd columns
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)

        # Position embedding table: block_size rows, n_embd columns
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)

        # The stack of transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock() for _ in range(cfg.n_layers)
        ])

        # Final layer norm (applied after all blocks)
        self.ln_f = nn.LayerNorm(cfg.n_embd)

        # The "language model head": projects from n_embd to vocab_size.
        # This gives one score per vocabulary token at each position.
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, token_ids):
        B, T = token_ids.shape
        assert T <= self.cfg.block_size, \
            f"Sequence length {T} exceeds block_size {self.cfg.block_size}"

        # Token embeddings: (B, T) -> (B, T, C)
        tok_emb = self.token_emb(token_ids)

        # Position embeddings: (T,) -> (T, C), broadcast to (B, T, C)
        positions = torch.arange(T, device=token_ids.device)
        pos_emb   = self.pos_emb(positions)

        # Combine embeddings
        x = tok_emb + pos_emb    # (B, T, C)

        # Pass through all transformer blocks
        x = self.blocks(x)       # (B, T, C)

        # Final layer norm
        x = self.ln_f(x)         # (B, T, C)

        # Project to vocabulary scores
        logits = self.lm_head(x) # (B, T, vocab_size)

        return logits


if __name__ == "__main__":
    print("=" * 50)
    print("Chapter 9: The Full GPT Model")
    print("=" * 50)

    model = GPT(config)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameter breakdown:")
    print(f"  Token embedding  : {model.token_emb.weight.numel():>10,}")
    print(f"  Position embedding: {model.pos_emb.weight.numel():>9,}")

    block_params = sum(p.numel() for p in model.blocks.parameters())
    print(f"  {config.n_layers} Transformer blocks: {block_params:>9,}")

    lm_head_params = sum(p.numel() for p in model.lm_head.parameters())
    print(f"  LM head          : {lm_head_params:>10,}")
    print(f"  LayerNorm (final): {sum(p.numel() for p in model.ln_f.parameters()):>10,}")
    print(f"  ---")
    print(f"  TOTAL            : {total_params:>10,}")

    B, T = 2, 10
    token_ids = torch.randint(0, config.vocab_size, (B, T))
    logits = model(token_ids)

    print(f"\nInput  shape: {token_ids.shape}   (B, T)")
    print(f"Output shape: {logits.shape}  (B, T, vocab_size)")
    print(f"\nAt each position, the model outputs {config.vocab_size} scores --")
    print(f"one per character. The highest score = best guess for next character.")

    first_pos_logits = logits[0, 0]
    print(f"\nFirst position logits (top 5 scores):")
    top5 = first_pos_logits.topk(5)
    for score, idx in zip(top5.values.tolist(), top5.indices.tolist()):
        print(f"  token {idx:2d}: {score:.3f}")

    print(f"\nNote: these are random (untrained model). After training they become meaningful!")
    print("\nFull GPT model done! Ready for Chapter 10.")
