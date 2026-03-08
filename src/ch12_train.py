"""
ch12_train.py - The Training Loop

This is the most important file in the whole tutorial.
Everything else was preparation for this moment.

The training loop repeats the same 4 steps over and over:

  1. FORWARD PASS    - feed a batch of text through the model, get predictions
  2. COMPUTE LOSS    - measure how wrong the predictions are
  3. BACKWARD PASS   - figure out which parameters caused the mistakes
  4. UPDATE          - nudge the parameters to make fewer mistakes next time

Each repetition of these 4 steps = one "training step".
After thousands of steps, the model gets surprisingly good.

The optimizer (Adam) handles steps 3 and 4 automatically.
We just call loss.backward() and optimizer.step().

Run this file (takes ~20-30 min on a CPU-only laptop):
    python src/ch12_train.py

Tip: You can reduce MAX_ITERS to 200 for a quick test run.
"""

import torch
import torch.nn.functional as F
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import GPTConfig, TrainConfig
from src.ch09_gpt_model import GPT
from src.ch11_dataloader import train_dataset, val_dataset, TextDataset

from torch.utils.data import DataLoader

gpt_cfg   = GPTConfig()
train_cfg = TrainConfig()

print("=" * 50)
print("Chapter 12: The Training Loop")
print("=" * 50)

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
torch.manual_seed(42)   # for reproducibility

device = "cpu"   # no GPU needed!

model = GPT(gpt_cfg).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {total_params:,}")
print(f"Training on     : {device}")
print(f"Steps           : {train_cfg.max_iters:,}")
print(f"Batch size      : {train_cfg.batch_size}")
print(f"Block size      : {gpt_cfg.block_size}")

# Adam optimizer: the go-to optimizer for transformers.
# It adapts the learning rate for each parameter automatically.
optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=train_cfg.batch_size, shuffle=False)
train_iter   = iter(train_loader)


def get_batch(loader, loader_iter):
    """Get the next batch, restarting the iterator if exhausted."""
    try:
        x, y = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        x, y = next(loader_iter)
    return x.to(device), y.to(device), loader_iter


@torch.no_grad()
def estimate_val_loss():
    """Compute average loss on 50 validation batches (no gradient needed)."""
    model.eval()
    val_iter = iter(val_loader)
    losses = []
    for _ in range(min(50, len(val_loader))):
        x, y = next(val_iter)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, gpt_cfg.vocab_size),
            y.view(-1)
        )
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------
print(f"\nStarting training... (eval every {train_cfg.eval_interval} steps)")
print("-" * 50)

start_time = time.time()
train_losses = []

for step in range(1, train_cfg.max_iters + 1):

    # --- Step 1 & 2: Forward pass + loss ---
    x, y, train_iter = get_batch(train_loader, train_iter)
    logits = model(x)   # (B, T, vocab_size)
    loss = F.cross_entropy(
        logits.view(-1, gpt_cfg.vocab_size),
        y.view(-1)
    )

    # --- Step 3: Backward pass (compute gradients) ---
    optimizer.zero_grad()   # clear old gradients from previous step
    loss.backward()         # compute new gradients

    # --- Step 4: Update parameters ---
    optimizer.step()

    train_losses.append(loss.item())

    # Log progress every eval_interval steps
    if step % train_cfg.eval_interval == 0 or step == 1:
        val_loss   = estimate_val_loss()
        avg_train  = sum(train_losses[-train_cfg.eval_interval:]) / len(train_losses[-train_cfg.eval_interval:])
        elapsed    = time.time() - start_time
        steps_left = train_cfg.max_iters - step
        eta        = (elapsed / step) * steps_left if step > 0 else 0

        print(f"step {step:5d}/{train_cfg.max_iters} | "
              f"train loss: {avg_train:.4f} | "
              f"val loss: {val_loss:.4f} | "
              f"elapsed: {elapsed:.0f}s | "
              f"ETA: {eta:.0f}s")

# ------------------------------------------------------------------
# Save the trained model
# ------------------------------------------------------------------
os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(train_cfg.checkpoint_dir, "model.pt")

torch.save({
    "model_state": model.state_dict(),
    "gpt_cfg"    : gpt_cfg,
    "step"       : train_cfg.max_iters,
    "val_loss"   : val_loss,
}, checkpoint_path)

total_time = time.time() - start_time
print("-" * 50)
print(f"\nTraining complete! Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"Checkpoint saved to: {checkpoint_path}")
print("\nReady for Chapter 13 (checkpointing) and Chapter 14 (generation)!")
