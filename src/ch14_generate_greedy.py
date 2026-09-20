"""
Generate text using greedy decoding and sampling.
This file belongs to Chapter 15.
Run: python src/ch14_generate_greedy.py
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

def generate_greedy(model, prompt, encode, decode, cfg, max_new_tokens=200):
    # Always picks the token with the highest predicted score
    ids = torch.tensor([encode(prompt)], dtype=torch.long)
    for _ in range(max_new_tokens):
        ctx     = ids[:, -cfg.block_size:]
        logits  = model(ctx)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids     = torch.cat([ids, next_id], dim=1)
    return decode(ids[0].tolist())

def generate_sample(model, prompt, encode, decode, cfg, max_new_tokens=200):
    # Picks the next token randomly based on the model's probabilities
    ids = torch.tensor([encode(prompt)], dtype=torch.long)
    for _ in range(max_new_tokens):
        ctx    = ids[:, -cfg.block_size:]
        logits = model(ctx)
        probs  = F.softmax(logits[:, -1, :], dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
    return decode(ids[0].tolist())

# --- Demo ---
if __name__ == "__main__":
    torch.manual_seed(42)
    print("Chapter 15: Greedy Decoding vs Sampling\n")

    if not os.path.exists(DATA_PATH):
        print("ERROR: shakespeare.txt not found. Run download_data.py")
        sys.exit(1)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(set(text))
    char_to_id  = {ch: i for i, ch in enumerate(chars)}
    id_to_char  = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [char_to_id[c] for c in s if c in char_to_id]
    decode = lambda ids: "".join([id_to_char[i] for i in ids])

    checkpoint = torch.load(
        CHECKPOINT_PATH, map_location="cpu", weights_only=False
    )
    cfg = checkpoint["gpt_cfg"]
    model = GPT(cfg)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    trained_steps = checkpoint["step"]
    val_loss      = checkpoint["val_loss"]
    print(f"\nModel loaded: trained for {trained_steps} steps, "
          f"val loss = {val_loss:.4f}")

    prompt = "ROMEO:\n"
    print(f"\nPrompt: {repr(prompt)}")
    print("Generating 200 characters each...\n")

    with torch.no_grad():
        greedy_text = generate_greedy(
            model, prompt, encode, decode, cfg, max_new_tokens=200
        )
        sample_text = generate_sample(
            model, prompt, encode, decode, cfg, max_new_tokens=200
        )

    print("--- GREEDY (always picks highest-score token) ---")
    print(greedy_text)

    print("\n--- SAMPLING (picks randomly from distribution) ---")
    print(sample_text)

    print("\n--- Observations ---")
    print("Greedy tends to be more repetitive.")
    print("Sampling is more varied but can make unexpected choices.")
    print("For better output, see Chapter 16 (temperature and top-k).")
