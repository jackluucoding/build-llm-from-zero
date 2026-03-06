"""
ch15_generate_sampling.py - Temperature and Top-k Sampling

Plain sampling (Chapter 14) can be improved with two controls:

  TEMPERATURE controls HOW RANDOM the sampling is.
    - temperature = 1.0 : normal sampling
    - temperature < 1.0 : more focused (picks confident tokens more often)
    - temperature > 1.0 : more chaotic (gives unusual tokens a bigger chance)
    Think of it like a thermostat for creativity.
    Low temperature = formal, repetitive Shakespeare.
    High temperature = Shakespeare after too much mead.

  TOP-K keeps only the K most likely tokens and ignores the rest.
    - top_k = 1   : greedy decoding (same as argmax)
    - top_k = 40  : good balance of quality and variety
    - top_k = 65  : no filtering (all tokens allowed)
    Think of it like a talent show: only the top K contestants advance.
    The rest are eliminated BEFORE the random draw.

Together: we filter to top-k tokens first, then apply temperature,
then sample. This gives us fine-grained control over generation quality.

Run this file (requires a trained checkpoint from ch12_train.py):
    python src/ch15_generate_sampling.py
"""

import torch
import torch.nn.functional as F
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import GPTConfig
from src.ch09_gpt_model import GPT

print("=" * 50)
print("Chapter 15: Temperature and Top-k Sampling")
print("=" * 50)

# ------------------------------------------------------------------
# Load model and tokenizer
# ------------------------------------------------------------------
CHECKPOINT_PATH = "checkpoints/model.pt"
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "shakespeare.txt")

if not os.path.exists(DATA_PATH):
    print("ERROR: Run: python src/utils/download_data.py")
    sys.exit(1)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

chars  = sorted(set(text))
stoi   = {ch: i for i, ch in enumerate(chars)}
itos   = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda ids: "".join([itos[i] for i in ids])

checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
cfg   = checkpoint["gpt_cfg"]
model = GPT(cfg)
model.load_state_dict(checkpoint["model_state"])
model.eval()

print(f"\nModel loaded (trained {checkpoint['step']} steps)")


# ------------------------------------------------------------------
# Generation with temperature + top-k
# ------------------------------------------------------------------

def generate(model, prompt, max_new_tokens=200, temperature=1.0, top_k=None):
    """
    Generate text with temperature and optional top-k filtering.

    Args:
        model         : the GPT model
        prompt        : starting text string
        max_new_tokens: number of new tokens to generate
        temperature   : controls randomness (lower = more focused)
        top_k         : if set, only sample from the top k tokens
    """
    ids = torch.tensor([encode(prompt)], dtype=torch.long)

    for _ in range(max_new_tokens):
        ctx    = ids[:, -cfg.block_size:]
        logits = model(ctx)
        logits = logits[:, -1, :]   # (1, vocab_size)

        # Apply temperature: divide logits before softmax.
        # Lower temperature = sharper distribution (more confident).
        logits = logits / temperature

        # Apply top-k: zero out all logits except the top k.
        if top_k is not None:
            # Find the k-th largest value
            threshold = logits.topk(top_k).values[:, -1, None]
            # Set all logits below threshold to -inf
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        probs   = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids     = torch.cat([ids, next_id], dim=1)

    return decode(ids[0].tolist())


# ------------------------------------------------------------------
# Side-by-side comparison
# ------------------------------------------------------------------
prompt = "JULIET:\n"
n_tokens = 150

print(f"\nPrompt: {repr(prompt)}")
print(f"Generating {n_tokens} characters each...\n")

configs = [
    {"temperature": 0.5, "top_k": 40,   "label": "temp=0.5, top_k=40  (focused)"},
    {"temperature": 0.8, "top_k": 40,   "label": "temp=0.8, top_k=40  (balanced - recommended default)"},
    {"temperature": 1.0, "top_k": 40,   "label": "temp=1.0, top_k=40  (the model's raw distribution)"},
    {"temperature": 1.5, "top_k": 40,   "label": "temp=1.5, top_k=40  (creative/chaotic)"},
    {"temperature": 1.0, "top_k": None, "label": "temp=1.0, no top_k  (all tokens eligible)"},
]

with torch.no_grad():
    for cfg_dict in configs:
        text_out = generate(
            model, prompt,
            max_new_tokens=n_tokens,
            temperature=cfg_dict["temperature"],
            top_k=cfg_dict["top_k"]
        )
        print("=" * 50)
        print(cfg_dict["label"])
        print("=" * 50)
        print(text_out)
        print()

print("--- Sweet spot ---")
print("temperature=0.8 to 1.0 and top_k=40 usually gives the best results.")
print("Experiment with these values to see what you prefer!")
