"""
Central configuration holding all hyperparameters.
This file is used across multiple chapters.
Run: python src/utils/config.py
"""

from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 128
    n_embd: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1

@dataclass
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_iters: int = 3000
    eval_interval: int = 300
    checkpoint_dir: str = "checkpoints"
    val_split: float = 0.1

gpt_config = GPTConfig()
train_config = TrainConfig()

if __name__ == "__main__":
    print("=== GPT Model Config ===")
    for field, value in gpt_config.__dict__.items():
        print(f"  {field}: {value}")

    print("\n=== Training Config ===")
    for field, value in train_config.__dict__.items():
        print(f"  {field}: {value}")

    assert gpt_config.n_embd % gpt_config.n_heads == 0, \
        "n_embd must be divisible by n_heads!"
    head_size = gpt_config.n_embd // gpt_config.n_heads
    print(f"\nhead_size = {gpt_config.n_embd} / {gpt_config.n_heads} "
          f"= {head_size} (OK)")
