"""
ch14_generate_greedy.py - Greedy Decoding and Sampling

The model is trained. Now let's make it write!

We have two basic strategies for picking the next token:

  GREEDY: always pick the token with the HIGHEST score.
    Pro: consistent and reproducible.
    Con: often repetitive. ("the the the the the...")

  SAMPLING: pick a token RANDOMLY, weighted by the scores.
    Pro: more varied and creative.
    Con: sometimes picks weird tokens.

Greedy is like always ordering your "usual" at a restaurant.
Sampling is like rolling a weighted die - the popular dishes
come up more often, but surprises happen.

Run this file (requires a trained checkpoint from ch12_train.py):
    python src/ch14_generate_greedy.py

If you haven't trained yet, run ch12_train.py first (20-30 minutes).
"""

import torch
import torch.nn.functional as F
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import GPTConfig
from src.ch09_gpt_model import GPT

print("=" * 50)
print("Chapter 14: Greedy Decoding vs Sampling")
print("=" * 50)

# ------------------------------------------------------------------
# Load model and tokenizer
# ------------------------------------------------------------------
CHECKPOINT_PATH = "checkpoints/model.pt"
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "shakespeare.txt")

if not os.path.exists(DATA_PATH):
    print("ERROR: shakespeare.txt not found. Run: python src/utils/download_data.py")
    sys.exit(1)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
stoi  = {ch: i for i, ch in enumerate(chars)}
itos  = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda ids: "".join([itos[i] for i in ids])

checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
cfg = checkpoint["gpt_cfg"]
model = GPT(cfg)
model.load_state_dict(checkpoint["model_state"])
model.eval()

trained_steps = checkpoint["step"]
val_loss      = checkpoint["val_loss"]
print(f"\nModel loaded: trained for {trained_steps} steps, val loss = {val_loss:.4f}")


# ------------------------------------------------------------------
# Generation functions
# ------------------------------------------------------------------

def generate_greedy(model, prompt, max_new_tokens=200):
    """Always pick the token with the highest score (argmax)."""
    ids = torch.tensor([encode(prompt)], dtype=torch.long)   # (1, T)

    for _ in range(max_new_tokens):
        ctx     = ids[:, -cfg.block_size:]
        logits  = model(ctx)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)   # greedy
        ids     = torch.cat([ids, next_id], dim=1)

    return decode(ids[0].tolist())


def generate_sample(model, prompt, max_new_tokens=200):
    """Sample from the probability distribution over tokens."""
    ids = torch.tensor([encode(prompt)], dtype=torch.long)   # (1, T)

    for _ in range(max_new_tokens):
        ctx    = ids[:, -cfg.block_size:]
        logits = model(ctx)
        probs  = F.softmax(logits[:, -1, :], dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)   # random sample
        ids = torch.cat([ids, next_id], dim=1)

    return decode(ids[0].tolist())


# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------
prompt = "ROMEO:\n"

print(f"\nPrompt: {repr(prompt)}")
print(f"Generating 200 characters each...\n")

with torch.no_grad():
    greedy_text = generate_greedy(model, prompt, max_new_tokens=200)
    sample_text = generate_sample(model, prompt, max_new_tokens=200)

print("=" * 50)
print("GREEDY (always picks highest-score token):")
print("=" * 50)
print(greedy_text)

print("\n" + "=" * 50)
print("SAMPLING (picks randomly from distribution):")
print("=" * 50)
print(sample_text)

print("\n--- Observations ---")
print("Greedy tends to be more repetitive but grammatically predictable.")
print("Sampling is more varied but can make unexpected choices.")
print("For better output, see Chapter 15 (temperature and top-k).")
