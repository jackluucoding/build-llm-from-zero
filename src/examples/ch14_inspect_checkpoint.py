"""Inspect the contents of the saved PyTorch checkpoint."""
import os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

def main():
    torch.manual_seed(42)
    checkpoint_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "checkpoints", "model.pt"
    )
    
    if not os.path.exists(checkpoint_path):
        print("Checkpoint not found.")
        return
        
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    print("Keys in checkpoint:", list(checkpoint.keys()))
    print("Training step:", checkpoint.get("step"))

if __name__ == "__main__":
    main()
