"""
ch10_causal_lm.py - Causal Language Modeling and Text Generation

We have a model that produces logits. Now what?

TRAINING OBJECTIVE: "Predict the next token."
Given the sequence "To be or not to b", the model should predict "e".
Given "To be or not to be, that is the questio", it should predict "n".

The trick: we can train on ALL positions at once!
- Input : "To be or not to b"    (tokens 0 to N-1)
- Target: "o be or not to be"    (tokens 1 to N)
Each position predicts the NEXT character. One forward pass = N training examples.

CAUSAL MASKING ensures the model can't "cheat" by looking ahead.
Token 5 can only see tokens 0-5, not 6,7,8,...
This is enforced by the triangular mask in SingleHeadAttention (Chapter 5).

GENERATION: once trained, we can feed a "prompt" and ask the model
to keep extending it one character at a time.

Run this file:
    python src/ch10_causal_lm.py
"""

import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import GPTConfig
from src.ch09_gpt_model import GPT

config = GPTConfig()

print("=" * 50)
print("Chapter 10: Causal Language Modeling and Generation")
print("=" * 50)

# ------------------------------------------------------------------
# 1. The training target: input and target are the same sequence, shifted
# ------------------------------------------------------------------
print("\n--- 1. Constructing input/target pairs ---")

# Example: encode the word "HELLO"
# In a real training step, we use a real chunk of Shakespeare.
example_ids = torch.tensor([20, 17, 30, 30, 33, 1, 35, 53, 56, 30])  # made up IDs
T = len(example_ids)

# Input = tokens 0 to T-2
# Target = tokens 1 to T-1  (shifted right by 1)
x = example_ids[:-1]   # shape (T-1,)
y = example_ids[1:]    # shape (T-1,)

print(f"Sequence: {example_ids.tolist()}")
print(f"Input  x: {x.tolist()}")
print(f"Target y: {y.tolist()}")
print(f"At each position i, x[i] should predict y[i] (the next token).")

# ------------------------------------------------------------------
# 2. Computing the loss
# ------------------------------------------------------------------
print("\n--- 2. Computing cross-entropy loss ---")

model = GPT(config)

# Forward pass with a small batch
B, T = 4, 20
token_ids = torch.randint(0, config.vocab_size, (B, T))
targets   = torch.randint(0, config.vocab_size, (B, T))

logits = model(token_ids)   # (B, T, vocab_size)

# Cross-entropy loss: measures how wrong our predictions are.
# Lower = better. Perfect predictions = 0. Random = ~log(65) ~ 4.17
# We reshape: (B, T, vocab_size) -> (B*T, vocab_size)
loss = F.cross_entropy(
    logits.view(B * T, config.vocab_size),
    targets.view(B * T)
)

print(f"Logits shape : {logits.shape}")
print(f"Targets shape: {targets.shape}")
print(f"Loss (random model): {loss.item():.4f}")
print(f"Expected loss for random: {torch.log(torch.tensor(config.vocab_size)):.4f}")
print(f"(These should be close - an untrained model is essentially random)")

# ------------------------------------------------------------------
# 3. Text generation
# ------------------------------------------------------------------
print("\n--- 3. Text generation ---")
print("We extend a starting sequence one token at a time.")
print("(Untrained model = random output, but the MECHANISM is correct.)")

def generate(model, start_ids, max_new_tokens):
    """
    Generate max_new_tokens new tokens given a starting sequence.

    Args:
        model       : the GPT model
        start_ids   : starting token IDs, shape (1, T)
        max_new_tokens: how many tokens to generate

    Returns:
        Full sequence including the starting tokens.
    """
    model.eval()
    context = start_ids.clone()   # (1, T)

    for _ in range(max_new_tokens):
        # Crop context to the last block_size tokens
        # (can't give the model more than it was trained on)
        ctx = context[:, -config.block_size:]   # (1, T')

        # Forward pass: get logits
        logits = model(ctx)   # (1, T', vocab_size)

        # We only care about the LAST position - that's our next-token prediction
        next_logits = logits[:, -1, :]   # (1, vocab_size)

        # Sample from the distribution (not just argmax - more variety!)
        probs = F.softmax(next_logits, dim=-1)   # convert to probabilities
        next_id = torch.multinomial(probs, num_samples=1)   # (1, 1)

        # Append the new token to the context
        context = torch.cat([context, next_id], dim=1)   # (1, T+1)

    return context


# Start with a single token (token ID 0)
start = torch.zeros((1, 1), dtype=torch.long)
output_ids = generate(model, start, max_new_tokens=50)

print(f"\nGenerated IDs (first 10): {output_ids[0, :10].tolist()}")
print(f"Output shape: {output_ids.shape}")
print(f"\nNote: these IDs are random since the model is untrained.")
print(f"After training (Chapter 12), this will produce Shakespeare-like text!")

print("\nCausal LM done! Ready for Chapter 11.")
