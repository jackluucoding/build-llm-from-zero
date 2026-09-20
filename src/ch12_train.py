"""
Implement the main training loop.
This file belongs to Chapter 13.
Run: python src/ch12_train.py
"""
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys

# Let Python find the book's own modules in src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import GPTConfig, TrainConfig
from src.ch09_gpt_model import GPT
from src.ch11_dataloader import train_dataset, val_dataset

# Settings
gpt_cfg   = GPTConfig()
train_cfg = TrainConfig()

# --- The Idea ---

# Disable gradient tracking for evaluation to save memory and compute
@torch.no_grad()
def estimate_val_loss(model, val_loader, device):
    # Set model to evaluation mode (disables dropout)
    model.eval()
    val_iter = iter(val_loader)
    losses = []
    for _ in range(min(50, len(val_loader))):
        x, y = next(val_iter)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        # Compute average loss for this batch
        loss = F.cross_entropy(logits.view(-1, gpt_cfg.vocab_size), y.view(-1))
        losses.append(loss.item())

    # Return to training mode
    model.train()
    return sum(losses) / len(losses)

def get_batch(loader, loader_iter, device):
    # Fetch the next batch, restarting the iterator if needed
    try:
        x, y = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        x, y = next(loader_iter)
    return x.to(device), y.to(device), loader_iter

# --- Demo ---
if __name__ == "__main__":
    print("Chapter 13: The Training Loop\n")

    torch.manual_seed(42)
    device = "cpu"

    model = GPT(gpt_cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Training on     : {device}")
    print(f"Steps           : {train_cfg.max_iters:,}")
    print(f"Batch size      : {train_cfg.batch_size}")
    print(f"Block size      : {gpt_cfg.block_size}")

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)

    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_cfg.batch_size, shuffle=False
    )
    train_iter = iter(train_loader)

    print(f"\nStarting training... (eval every {train_cfg.eval_interval} steps)")
    start_time = time.time()
    train_losses = []

    for step in range(1, train_cfg.max_iters + 1):
        x, y, train_iter = get_batch(train_loader, train_iter, device)

        # Forward pass
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, gpt_cfg.vocab_size), y.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if step % train_cfg.eval_interval == 0 or step == 1:
            val_loss = estimate_val_loss(model, val_loader, device)
            rec_losses = train_losses[-train_cfg.eval_interval:]
            avg_train = sum(rec_losses) / len(rec_losses)
            elapsed = time.time() - start_time
            steps_left = train_cfg.max_iters - step
            eta = (elapsed / step) * steps_left if step > 0 else 0

            print(f"step {step:5d}/{train_cfg.max_iters} | "
                  f"train loss: {avg_train:.4f} | "
                  f"val loss: {val_loss:.4f} | "
                  f"elapsed: {elapsed:.0f}s | "
                  f"ETA: {eta:.0f}s")

    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(train_cfg.checkpoint_dir, "model.pt")

    torch.save({
        "model_state": model.state_dict(),
        "gpt_cfg"    : gpt_cfg,
        "step"       : train_cfg.max_iters,
        "val_loss"   : val_loss,
    }, checkpoint_path)

    total_time = time.time() - start_time
    print(f"\nTraining complete! Total time: {total_time:.0f}s")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print("Ready for Chapter 14 (checkpointing) and Chapter 15 (generation)!")
