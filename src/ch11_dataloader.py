"""
ch11_dataloader.py - Dataset and DataLoader

Before training, we need to organise our text into BATCHES.

Here is the idea:
  - We have ~1 million characters of Shakespeare encoded as integers.
  - We slice it into overlapping windows of length block_size (128).
  - Each window is one TRAINING EXAMPLE: x (input) and y (target).
  - We group examples into BATCHES of 32 and shuffle them.

Why batches? Training on one example at a time is slow.
Processing 32 examples at once is ~32x faster because modern
CPUs (and GPUs) are good at doing many things in parallel.

Think of it like washing dishes: one at a time vs loading a full dishwasher.

Run this file:
    python src/ch11_dataloader.py
"""

import torch
from torch.utils.data import Dataset, DataLoader
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import GPTConfig, TrainConfig

gpt_cfg   = GPTConfig()
train_cfg = TrainConfig()

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "shakespeare.txt")

# ------------------------------------------------------------------
# Load and tokenize the data
# (module-level so ch12 can import train_dataset / val_dataset)
# ------------------------------------------------------------------
if not os.path.exists(DATA_PATH):
    print("ERROR: shakespeare.txt not found. Run: python src/utils/download_data.py")
    sys.exit(1)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
stoi  = {ch: i for i, ch in enumerate(chars)}

data  = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n     = len(data)
split = int(0.9 * n)
train_data = data[:split]
val_data   = data[split:]


class TextDataset(Dataset):
    """
    Slices a token sequence into overlapping (input, target) windows.

    For each starting position i:
      x = tokens[i   : i + block_size]    (the input)
      y = tokens[i+1 : i + block_size+1]  (the target -- shifted by 1)
    """

    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx     : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


# Module-level datasets - exported and imported by ch12_train.py
train_dataset = TextDataset(train_data, gpt_cfg.block_size)
val_dataset   = TextDataset(val_data,   gpt_cfg.block_size)

train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=train_cfg.batch_size, shuffle=False)


if __name__ == "__main__":
    print("=" * 50)
    print("Chapter 11: Dataset and DataLoader")
    print("=" * 50)

    print(f"\nData loaded: {n:,} total tokens")
    print(f"  Train : {len(train_data):,} tokens")
    print(f"  Val   : {len(val_data):,} tokens")

    print(f"\nDataset sizes:")
    print(f"  Train examples: {len(train_dataset):,}")
    print(f"  Val   examples: {len(val_dataset):,}")
    print(f"\nDataLoader config:")
    print(f"  Batch size  : {train_cfg.batch_size}")
    print(f"  Train batches per epoch: {len(train_loader):,}")

    print("\n--- Inspecting one batch ---")

    x_batch, y_batch = next(iter(train_loader))
    print(f"x_batch shape: {x_batch.shape}  (batch_size, block_size)")
    print(f"y_batch shape: {y_batch.shape}  (batch_size, block_size)")

    itos   = {i: c for i, c in enumerate(chars)}
    decode = lambda ids: "".join([itos[i.item()] for i in ids])

    print(f"\nFirst example in batch:")
    print(f"  x (input)  : {repr(decode(x_batch[0]))[:60]}...")
    print(f"  y (target) : {repr(decode(y_batch[0]))[:60]}...")
    print(f"  (y is x shifted by 1 character)")

    print("\nDataLoader ready! Ready for Chapter 12.")
