"""
Explore temperature and top-k sampling.
This file belongs to Chapter 16.
Run: python src/ch15_generate_sampling.py
"""
import os
import sys
import torch
import torch.nn.functional as F


# Let Python find the book's own modules in src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import GPTConfig
from src.ch09_gpt_model import GPT

# Settings
CHECKPOINT_PATH = "checkpoints/model.pt"
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "shakespeare.txt")

# --- The Idea ---

def generate(
    model, prompt, encode, decode, cfg, max_new_tokens=200,
    temperature=1.0, top_k=None
):
    ids = torch.tensor([encode(prompt)], dtype=torch.long)

    for _ in range(max_new_tokens):
        ctx    = ids[:, -cfg.block_size:]
        logits = model(ctx)
        logits = logits[:, -1, :]

        # Temperature controls randomness: lower is less random
        logits = logits / temperature

        # Top-k sampling limits the choices to the k most likely tokens
        if top_k is not None:
            threshold = logits.topk(top_k).values[:, -1, None]
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        probs   = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids     = torch.cat([ids, next_id], dim=1)

    return decode(ids[0].tolist())

# --- Demo ---
if __name__ == "__main__":
    torch.manual_seed(42)
    print("Chapter 16: Temperature and Top-k Sampling\n")

    if not os.path.exists(DATA_PATH):
        print("ERROR: Run: python src/utils/download_data.py")
        sys.exit(1)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    chars  = sorted(set(text))
    char_to_id   = {ch: i for i, ch in enumerate(chars)}
    id_to_char   = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [char_to_id[c] for c in s if c in char_to_id]
    decode = lambda ids: "".join([id_to_char[i] for i in ids])

    checkpoint = torch.load(
        CHECKPOINT_PATH, map_location="cpu", weights_only=False
    )
    cfg   = checkpoint["gpt_cfg"]
    model = GPT(cfg)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"Model loaded (trained {checkpoint['step']} steps)\n")

    prompt = "JULIET:\n"
    n_tokens = 150

    print(f"Prompt: {repr(prompt)}")
    print(f"Generating {n_tokens} characters each...\n")

    configs = [
        {"temperature": 0.5, "top_k": 40,   "label": "temp=0.5, top_k=40"},
        {"temperature": 0.8, "top_k": 40,   "label": "temp=0.8, top_k=40"},
        {"temperature": 1.0, "top_k": 40,   "label": "temp=1.0, top_k=40"},
        {"temperature": 1.5, "top_k": 40,   "label": "temp=1.5, top_k=40"},
        {"temperature": 1.0, "top_k": None, "label": "temp=1.0, no top_k"},
    ]

    with torch.no_grad():
        for cfg_dict in configs:
            text_out = generate(
                model, prompt, encode, decode, cfg,
                max_new_tokens=n_tokens,
                temperature=cfg_dict["temperature"],
                top_k=cfg_dict["top_k"]
            )
            print("---", cfg_dict["label"], "---")
            print(text_out)
            print()

    print("--- Sweet spot ---")
    print("temperature=0.8 to 1.0 and top_k=40 usually gives the best results.")
