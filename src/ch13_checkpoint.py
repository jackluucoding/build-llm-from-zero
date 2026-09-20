"""
Save and load model checkpoints.
This file belongs to Chapter 14.
Run: python src/ch13_checkpoint.py
"""
import os
import torch

import sys

# Let Python find the book's own modules in src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import GPTConfig
from src.ch09_gpt_model import GPT

# Settings
CHECKPOINT_PATH = "checkpoints/model.pt"

# --- Demo ---
if __name__ == "__main__":
    torch.manual_seed(42)
    print("Chapter 14: Saving and Loading Checkpoints\n")

    print("--- 1. Check if a checkpoint exists ---")
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\nNo checkpoint found at {CHECKPOINT_PATH}.")
        print("Creating a dummy model to demonstrate save/load...")
        cfg = GPTConfig()
        model = GPT(cfg)
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "gpt_cfg"    : cfg,
            "step"       : 0,
            "val_loss"   : float("inf"),
        }, CHECKPOINT_PATH)
        print(f"Dummy checkpoint saved to {CHECKPOINT_PATH}")

    print("\n--- 2. Load the checkpoint ---")
    print(f"Loading checkpoint from: {CHECKPOINT_PATH}")

    checkpoint = torch.load(
        CHECKPOINT_PATH, map_location="cpu", weights_only=False
    )
    step     = checkpoint["step"]
    val_loss = checkpoint["val_loss"]
    cfg      = checkpoint["gpt_cfg"]

    print(f"  Trained for  : {step} steps")
    print(f"  Val loss     : {val_loss:.4f}")
    print(f"  Model config : {cfg.n_layers} layers, {cfg.n_embd} embd, "
          f"{cfg.n_heads} heads")

    print("\n--- 3. Rebuild the model from the checkpoint ---")
    model = GPT(cfg)
    model.load_state_dict(checkpoint["model_state"])

    # Set to evaluation mode (disables dropout, making outputs deterministic)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel rebuilt successfully: {total_params:,} parameters loaded")

    print("\n--- 4. Verify the model works ---")
    dummy_ids = torch.zeros((1, 10), dtype=torch.long)
    with torch.no_grad():
        logits = model(dummy_ids)

    print(f"Input shape : {dummy_ids.shape}")
    print(f"Output shape: {logits.shape}   (looks good!)")

    print("\n--- 5. Show checkpoint file size ---")
    size_mb = os.path.getsize(CHECKPOINT_PATH) / (1024 * 1024)
    print(f"\nCheckpoint file size: {size_mb:.2f} MB")
    print("(Small enough to share by email!)")

    print("\nCheckpointing done! Ready for Chapter 15.")
