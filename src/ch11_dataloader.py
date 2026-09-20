"""
Create the Dataset and DataLoader for training.
This file belongs to Chapter 12.
Run: python src/ch11_dataloader.py
"""
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader


# Let Python find the book's own modules in src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import GPTConfig, TrainConfig

# Settings
gpt_cfg   = GPTConfig()
train_cfg = TrainConfig()

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "shakespeare.txt")

if not os.path.exists(DATA_PATH):
    print("ERROR: shakespeare.txt not found. Run download_data.py")
    sys.exit(1)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
char_to_id = {ch: i for i, ch in enumerate(chars)}

# Store IDs in a PyTorch tensor
data  = torch.tensor([char_to_id[c] for c in text], dtype=torch.long)
n     = len(data)
split = int(0.9 * n)
train_data = data[:split]
val_data   = data[split:]

# --- The Idea ---
class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        # We need block_size + 1 tokens to form one (input, target) pair
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # Input is a block of text
        x = self.data[idx     : idx + self.block_size]
        # Target is the same text, shifted one character to the right
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y

train_dataset = TextDataset(train_data, gpt_cfg.block_size)
val_dataset   = TextDataset(val_data,   gpt_cfg.block_size)

# DataLoader automatically batches the data for us
train_loader = DataLoader(
    train_dataset, batch_size=train_cfg.batch_size, shuffle=True
)
val_loader = DataLoader(
    val_dataset, batch_size=train_cfg.batch_size, shuffle=False
)

# --- Demo ---
if __name__ == "__main__":
    torch.manual_seed(42)
    print("Chapter 12: Dataset and DataLoader\n")

    print(f"Data loaded: {n:,} total tokens")
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

    id_to_char = {i: c for i, c in enumerate(chars)}
    decode = lambda ids: "".join([id_to_char[i.item()] for i in ids])

    print(f"\nFirst example in batch:")
    print(f"  x (input)  : {repr(decode(x_batch[0]))[:60]}...")
    print(f"  y (target) : {repr(decode(y_batch[0]))[:60]}...")
    print(f"  (y is x shifted by 1 character)")

    print("\nDataLoader ready! Ready for Chapter 13.")
