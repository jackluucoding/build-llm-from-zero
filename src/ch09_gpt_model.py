"""
Construct the full GPT architecture.
This file belongs to Chapter 10.
Run: python src/ch09_gpt_model.py
"""
import torch
import torch.nn as nn
import os
import sys

# Let Python find the book's own modules in src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import GPTConfig
from src.ch08_transformer_block import TransformerBlock

# Settings
config = GPTConfig()

# --- The Idea ---
class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        # Look up table for token vectors
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        # Look up table for position vectors
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)

        # A sequence of transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock() for _ in range(cfg.n_layers)
        ])

        # Final layer normalization and linear layer to output scores
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, token_ids):
        B, T = token_ids.shape
        assert T <= self.cfg.block_size, \
            f"Sequence length {T} exceeds block_size " \
            f"{self.cfg.block_size}"

        tok_emb = self.token_emb(token_ids)
        positions = torch.arange(T, device=token_ids.device)
        pos_emb = self.pos_emb(positions)

        # Combine token and position embeddings
        x = tok_emb + pos_emb

        # Pass through the transformer blocks
        x = self.blocks(x)

        # Final normalization and produce scores
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

# --- Demo ---
if __name__ == "__main__":
    torch.manual_seed(42)
    print("Chapter 10: The Full GPT Model\n")

    model = GPT(config)
    total_params = sum(p.numel() for p in model.parameters())

    print("Model parameter breakdown:")
    print(f"  Token embedding  : {model.token_emb.weight.numel():>10,}")
    print(f"  Position embedding: {model.pos_emb.weight.numel():>9,}")
    block_params = sum(p.numel() for p in model.blocks.parameters())
    print(f"  {config.n_layers} Transformer blocks: {block_params:>9,}")
    lm_head_params = sum(p.numel() for p in model.lm_head.parameters())
    print(f"  LM head          : {lm_head_params:>10,}")
    ln_f_params = sum(p.numel() for p in model.ln_f.parameters())
    print(f"  LayerNorm (final): {ln_f_params:>10,}")
    print("  ---")
    print(f"  TOTAL            : {total_params:>10,}")

    B, T = 2, 10
    token_ids = torch.randint(0, config.vocab_size, (B, T))
    logits = model(token_ids)

    print(f"\nInput  shape: {token_ids.shape}   (B, T)")
    print(f"Output shape: {logits.shape}  (B, T, vocab_size)")
    print(f"\nAt each position, model outputs {config.vocab_size} scores.")
    print("The highest score = best guess for next character.")

    first_pos_logits = logits[0, 0]
    print("\nFirst position logits (top 5 scores):")
    top5 = first_pos_logits.topk(5)
    for score, idx in zip(top5.values.tolist(), top5.indices.tolist()):
        print(f"  token {idx:2d}: {score:.3f}")

    print("\nNote: these are random (untrained model).")
    print("\nFull GPT model done! Ready for Chapter 11.")
