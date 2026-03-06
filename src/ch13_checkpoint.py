"""
ch13_checkpoint.py - Saving and Loading Checkpoints

Training on a CPU takes several minutes.
What if your laptop runs out of battery halfway through?
What if you want to try different generation settings without retraining?

CHECKPOINTING solves this. We periodically save the model's parameters
to disk, so we can resume from where we left off.

A checkpoint is just a dictionary saved as a .pt file:
  {
    "model_state": the learned weights (everything the model knows),
    "gpt_cfg"    : the hyperparameters (so we can rebuild the model),
    "step"       : how far training had progressed,
    "val_loss"   : the validation loss at that point,
  }

Think of it like saving a video game. The model is your character.
The checkpoint file is the save file.

Run this file (requires a trained checkpoint from ch12_train.py):
    python src/ch13_checkpoint.py
"""

import torch
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import GPTConfig
from src.ch09_gpt_model import GPT

print("=" * 50)
print("Chapter 13: Saving and Loading Checkpoints")
print("=" * 50)

CHECKPOINT_PATH = "checkpoints/model.pt"

# ------------------------------------------------------------------
# 1. Check if a checkpoint exists
# ------------------------------------------------------------------
if not os.path.exists(CHECKPOINT_PATH):
    print(f"\nNo checkpoint found at {CHECKPOINT_PATH}.")
    print("Creating a dummy model to demonstrate save/load...")

    # Create and save a fresh (untrained) model for demonstration
    cfg   = GPTConfig()
    model = GPT(cfg)
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "gpt_cfg"    : cfg,
        "step"       : 0,
        "val_loss"   : float("inf"),
    }, CHECKPOINT_PATH)
    print(f"Dummy checkpoint saved to {CHECKPOINT_PATH}")

# ------------------------------------------------------------------
# 2. Load the checkpoint
# ------------------------------------------------------------------
print(f"\nLoading checkpoint from: {CHECKPOINT_PATH}")

checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

step     = checkpoint["step"]
val_loss = checkpoint["val_loss"]
cfg      = checkpoint["gpt_cfg"]

print(f"  Trained for  : {step} steps")
print(f"  Val loss     : {val_loss:.4f}")
print(f"  Model config : {cfg.n_layers} layers, {cfg.n_embd} embd, {cfg.n_heads} heads")

# ------------------------------------------------------------------
# 3. Rebuild the model from the checkpoint
# ------------------------------------------------------------------
model = GPT(cfg)
model.load_state_dict(checkpoint["model_state"])
model.eval()   # switch off dropout for inference

total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel rebuilt successfully: {total_params:,} parameters loaded")

# ------------------------------------------------------------------
# 4. Verify the model works
# ------------------------------------------------------------------
print("\n--- Quick forward pass to verify model ---")

dummy_ids = torch.zeros((1, 10), dtype=torch.long)
with torch.no_grad():
    logits = model(dummy_ids)

print(f"Input shape : {dummy_ids.shape}")
print(f"Output shape: {logits.shape}   (looks good!)")

# ------------------------------------------------------------------
# 5. Show checkpoint file size
# ------------------------------------------------------------------
size_mb = os.path.getsize(CHECKPOINT_PATH) / (1024 * 1024)
print(f"\nCheckpoint file size: {size_mb:.2f} MB")
print(f"(Small enough to share by email!)")

print("\nCheckpointing done! Ready for Chapter 14.")
