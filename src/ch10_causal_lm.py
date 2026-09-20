"""
Train the model to predict the next token and generate text.
This file belongs to Chapter 11.
Run: python src/ch10_causal_lm.py
"""
import torch
import torch.nn.functional as F
import os
import sys

# Let Python find the book's own modules in src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import GPTConfig
from src.ch09_gpt_model import GPT

# Settings
config = GPTConfig()

# --- The Idea ---
def generate(model, start_ids, max_new_tokens):
    # Set model to evaluation mode (e.g. disable dropout)
    model.eval()
    context = start_ids.clone()

    for _ in range(max_new_tokens):
        # Crop context to the maximum block size the model can handle
        ctx = context[:, -config.block_size:]
        logits = model(ctx)

        # Focus on the very last time step to predict the next token
        next_logits = logits[:, -1, :]
        probs = F.softmax(next_logits, dim=-1)

        # Randomly sample the next token based on probabilities
        next_id = torch.multinomial(probs, num_samples=1)

        # Append the new token to the sequence
        context = torch.cat([context, next_id], dim=1)

    return context

# --- Demo ---
if __name__ == "__main__":
    torch.manual_seed(42)
    print("Chapter 11: Causal Language Modeling and Generation\n")

    print("--- 1. Constructing input/target pairs ---")
    example_ids = torch.tensor([20, 17, 30, 30, 33, 1, 35, 53, 56, 30])
    T = len(example_ids)

    x = example_ids[:-1]
    y = example_ids[1:]

    print(f"Sequence: {example_ids.tolist()}")
    print(f"Input  x: {x.tolist()}")
    print(f"Target y: {y.tolist()}")
    print("At each position i, x[i] predicts y[i].")

    print("\n--- 2. Computing cross-entropy loss ---")
    model = GPT(config)

    B, T_len = 4, 20
    token_ids = torch.randint(0, config.vocab_size, (B, T_len))
    targets   = torch.randint(0, config.vocab_size, (B, T_len))

    logits = model(token_ids)

    # Calculate loss by comparing predictions to actual targets
    loss = F.cross_entropy(
        logits.view(B * T_len, config.vocab_size),
        targets.view(B * T_len)
    )

    print(f"Logits shape : {logits.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Loss (random model): {loss.item():.4f}")
    print(f"Expected loss for random: "
          f"{torch.log(torch.tensor(config.vocab_size)):.4f}")

    print("\n--- 3. Text generation ---")
    print("We extend a starting sequence one token at a time.")

    start = torch.zeros((1, 1), dtype=torch.long)
    output_ids = generate(model, start, max_new_tokens=50)

    print(f"\nGenerated IDs (first 10): {output_ids[0, :10].tolist()}")
    print(f"Output shape: {output_ids.shape}")
    print("After training (Chapter 13), this will produce real text!")
    print("\nCausal LM done! Ready for Chapter 12.")
