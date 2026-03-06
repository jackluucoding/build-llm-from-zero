"""
ch16_full_pipeline.py - Putting It All Together

This is the grand finale.

This single script runs the ENTIRE pipeline from start to finish:
  1. Download the Shakespeare dataset
  2. Tokenize the text
  3. Create DataLoaders
  4. Build the GPT model
  5. Train for 3000 steps (~20-30 min on CPU)
  6. Save a checkpoint
  7. Generate Shakespeare-like text

If you run only one file from this tutorial, run this one.
It is a complete, self-contained demonstration of everything
you have learned across all 16 chapters.

Run this file:
    python src/ch16_full_pipeline.py

Expected output at the end (approximate - varies by run):

  ROMEO:
  The king of all my love, and I will stay,
  And if thou hast no more than this poor man,
  That is the best of all the world's...

Run time: approximately 20-30 minutes on a CPU-only laptop.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import sys
import time
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import GPTConfig, TrainConfig
from src.ch09_gpt_model import GPT

print("=" * 60)
print("Chapter 16: Building an LLM from Zero -- Full Pipeline")
print("=" * 60)

gpt_cfg   = GPTConfig()
train_cfg = TrainConfig()
torch.manual_seed(42)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "shakespeare.txt")
DATA_URL  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
CKPT_PATH = os.path.join(train_cfg.checkpoint_dir, "model_final.pt")

# =========================================================
# Step 1: Download data
# =========================================================
print("\n[1/7] Downloading dataset...")
if not os.path.exists(DATA_PATH):
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    r = requests.get(DATA_URL)
    r.raise_for_status()
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        f.write(r.text)
    print(f"      Downloaded {os.path.getsize(DATA_PATH)//1024} KB")
else:
    print(f"      Already exists ({os.path.getsize(DATA_PATH)//1024} KB)")

# =========================================================
# Step 2: Tokenize
# =========================================================
print("\n[2/7] Tokenizing...")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

chars  = sorted(set(text))
vocab_size = len(chars)
stoi   = {ch: i for i, ch in enumerate(chars)}
itos   = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join([itos[i] for i in ids])

data  = torch.tensor(encode(text), dtype=torch.long)
n     = len(data)
split = int(0.9 * n)
train_data = data[:split]
val_data   = data[split:]
print(f"      {n:,} tokens, vocab={vocab_size}, train={len(train_data):,}, val={len(val_data):,}")

# =========================================================
# Step 3: DataLoaders
# =========================================================
print("\n[3/7] Creating DataLoaders...")

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

train_loader = DataLoader(TextDataset(train_data, gpt_cfg.block_size),
                          batch_size=train_cfg.batch_size, shuffle=True)
val_loader   = DataLoader(TextDataset(val_data,   gpt_cfg.block_size),
                          batch_size=train_cfg.batch_size, shuffle=False)
train_iter   = iter(train_loader)
print(f"      {len(train_loader):,} train batches, {len(val_loader):,} val batches")

# =========================================================
# Step 4: Build model
# =========================================================
print("\n[4/7] Building model...")
model     = GPT(gpt_cfg)
n_params  = sum(p.numel() for p in model.parameters())
optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
print(f"      {n_params:,} parameters")

# =========================================================
# Step 5: Train
# =========================================================
print(f"\n[5/7] Training for {train_cfg.max_iters} steps...")
print(f"      Logging every {train_cfg.eval_interval} steps")
print("-" * 60)

@torch.no_grad()
def val_loss_estimate():
    model.eval()
    it = iter(val_loader)
    losses = [F.cross_entropy(model(x).view(-1, vocab_size), y.view(-1)).item()
              for x, y in [next(it) for _ in range(min(50, len(val_loader)))]]
    model.train()
    return sum(losses) / len(losses)

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
        val_loss  = val_loss_estimate()
        avg_train = sum(recent_losses[-train_cfg.eval_interval:]) / \
                    len(recent_losses[-train_cfg.eval_interval:])
        elapsed   = time.time() - start
        eta       = (elapsed / step) * (train_cfg.max_iters - step)
        print(f"step {step:5d} | train: {avg_train:.4f} | val: {val_loss:.4f} | "
              f"elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")

total_time = time.time() - start
print("-" * 60)
print(f"      Training done in {total_time:.0f}s ({total_time/60:.1f} min)")

# =========================================================
# Step 6: Save checkpoint
# =========================================================
print("\n[6/7] Saving checkpoint...")
os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
torch.save({
    "model_state": model.state_dict(),
    "gpt_cfg"    : gpt_cfg,
    "step"       : train_cfg.max_iters,
    "val_loss"   : val_loss,
}, CKPT_PATH)
print(f"      Saved to: {CKPT_PATH}")

# =========================================================
# Step 7: Generate text
# =========================================================
print("\n[7/7] Generating text...")
model.eval()

def generate(prompt, max_new_tokens=300, temperature=0.8, top_k=40):
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

print("\n" + "=" * 60)
print("GENERATED TEXT (temperature=0.8, top_k=40):")
print("=" * 60)
print(generate("ROMEO:\n", max_new_tokens=400))
print("=" * 60)
print("\nCongratulations! You just built and trained an LLM from zero.")
