"""
End-to-end pipeline to build and train the LLM.
This file belongs to Chapter 17.
Run: python src/ch16_full_pipeline.py
"""
import os
import time
import requests
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import sys

# Let Python find the book's own modules in src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import GPTConfig, TrainConfig
from src.ch09_gpt_model import GPT

# Settings
gpt_cfg   = GPTConfig()
train_cfg = TrainConfig()

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "shakespeare.txt")
DATA_URL  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
CKPT_PATH = os.path.join(train_cfg.checkpoint_dir, "model_final.pt")

# --- The Idea ---

class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
    def __len__(self):
        return len(self.data) - self.block_size
    def __getitem__(self, idx):
        x = self.data[idx     : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y

@torch.no_grad()
def val_loss_estimate(model, val_loader, vocab_size):
    model.eval()
    it = iter(val_loader)
    losses = []
    for _ in range(min(50, len(val_loader))):
        x, y = next(it)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

def generate(
    model, prompt, encode, decode, max_new_tokens=300,
    temperature=0.8, top_k=40
):
    ids = torch.tensor([encode(prompt)], dtype=torch.long)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            ctx    = ids[:, -gpt_cfg.block_size:]
            logits = model(ctx)[:, -1, :] / temperature
            if top_k:
                thresh = logits.topk(top_k).values[:, -1, None]
                logits = logits.masked_fill(logits < thresh, float("-inf"))
            probs   = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids     = torch.cat([ids, next_id], dim=1)
    return decode(ids[0].tolist())

# --- Demo ---
if __name__ == "__main__":
    print("Chapter 17: Building an LLM from Zero -- Full Pipeline\n")

    torch.manual_seed(42)

    print("[1/7] Downloading dataset...")
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        r = requests.get(DATA_URL)
        r.raise_for_status()
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            f.write(r.text)
        print(f"      Downloaded {os.path.getsize(DATA_PATH)//1024} KB")
    else:
        print(f"      Already exists ({os.path.getsize(DATA_PATH)//1024} KB)")

    print("\n[2/7] Tokenizing...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    chars  = sorted(set(text))
    vocab_size = len(chars)
    char_to_id = {ch: i for i, ch in enumerate(chars)}
    id_to_char = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [char_to_id[c] for c in s]
    decode = lambda ids: "".join([id_to_char[i] for i in ids])

    data  = torch.tensor(encode(text), dtype=torch.long)
    n     = len(data)
    split = int(0.9 * n)
    train_data = data[:split]
    val_data   = data[split:]
    print(f"      {n:,} tokens, vocab={vocab_size}, "
          f"train={len(train_data):,}, val={len(val_data):,}")

    print("\n[3/7] Creating DataLoaders...")

    train_loader = DataLoader(
        TextDataset(train_data, gpt_cfg.block_size),
        batch_size=train_cfg.batch_size, shuffle=True
    )
    val_loader   = DataLoader(
        TextDataset(val_data,   gpt_cfg.block_size),
        batch_size=train_cfg.batch_size, shuffle=False
    )
    train_iter   = iter(train_loader)
    print(f"      {len(train_loader):,} train batches, "
          f"{len(val_loader):,} val batches")

    print("\n[4/7] Building model...")
    model     = GPT(gpt_cfg)
    n_params  = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    print(f"      {n_params:,} parameters")

    print(f"\n[5/7] Training for {train_cfg.max_iters} steps...")
    print(f"      Logging every {train_cfg.eval_interval} steps\n")

    start = time.time()
    recent_losses = []
    val_loss = float("inf")

    for step in range(1, train_cfg.max_iters + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        logits = model(x)
        loss   = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        recent_losses.append(loss.item())

        if step % train_cfg.eval_interval == 0 or step == 1:
            val_loss  = val_loss_estimate(model, val_loader, vocab_size)
            recent = recent_losses[-train_cfg.eval_interval:]
            avg_train = sum(recent) / len(recent)
            elapsed   = time.time() - start
            eta       = (elapsed / step) * (train_cfg.max_iters - step)
            print(f"step {step:5d} | train: {avg_train:.4f} | "
                  f"val: {val_loss:.4f} | "
                  f"elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")

    total_time = time.time() - start
    print(f"      Training done in {total_time:.0f}s")

    print("\n[6/7] Saving checkpoint...")
    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "gpt_cfg"    : gpt_cfg,
        "step"       : train_cfg.max_iters,
        "val_loss"   : val_loss,
    }, CKPT_PATH)
    print(f"      Saved to: {CKPT_PATH}")

    print("\n[7/7] Generating text...")
    model.eval()

    print("\nGENERATED TEXT (temperature=0.8, top_k=40):\n")
    print(generate(model, "ROMEO:\n", encode, decode, max_new_tokens=400))
    print("\nCongratulations! You just built and trained an LLM from zero.")
