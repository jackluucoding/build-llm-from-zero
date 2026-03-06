"""
config.py - Central configuration for "Building an LLM from Zero"

All hyperparameters live here. Every other file imports from this module
so you only need to change things in one place.

Designed to run on a CPU-only laptop.
Tested on: ThinkPad T14 Gen 1, Intel Core i7, 32 GB RAM (released 2020).
Expected training time: ~20-30 minutes for 3000 iterations.
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Defines the shape and size of the GPT model."""

    # Vocabulary: how many unique tokens exist?
    # For character-level Shakespeare this is ~65 (letters, digits, punctuation).
    vocab_size: int = 65

    # Context window: how many tokens can the model "see" at once?
    # Think of it like short-term memory - 128 characters back.
    block_size: int = 128

    # Embedding dimension: how many numbers represent each token?
    # Bigger = richer representation, but slower. 128 is a good starter.
    n_embd: int = 128

    # Number of attention heads (must divide n_embd evenly).
    # Each head looks at the text from a different angle.
    # head_size = n_embd // n_heads = 128 // 4 = 32
    n_heads: int = 4

    # Number of transformer blocks stacked on top of each other.
    # More layers = deeper understanding, but slower training.
    n_layers: int = 4

    # Dropout: randomly zeroes some values during training to prevent overfitting.
    # 0.1 means 10% of values are zeroed at each step.
    dropout: float = 0.1


@dataclass
class TrainConfig:
    """Defines how training is run."""

    # How many examples to process at once (one "batch").
    batch_size: int = 32

    # Learning rate: how big a step to take when updating the model.
    # Too high -> training is unstable. Too low -> training is very slow.
    learning_rate: float = 3e-4

    # Total number of training steps.
    max_iters: int = 3000

    # How often to print the current loss (every N steps).
    eval_interval: int = 300

    # Where to save checkpoints during training.
    checkpoint_dir: str = "checkpoints"

    # Fraction of data held out for validation (not used in training).
    val_split: float = 0.1


# Create default instances that other files can import directly.
gpt_config = GPTConfig()
train_config = TrainConfig()


if __name__ == "__main__":
    print("=== GPT Model Config ===")
    for field, value in gpt_config.__dict__.items():
        print(f"  {field}: {value}")

    print("\n=== Training Config ===")
    for field, value in train_config.__dict__.items():
        print(f"  {field}: {value}")

    # Quick sanity check: head_size must be a whole number.
    assert gpt_config.n_embd % gpt_config.n_heads == 0, \
        "n_embd must be divisible by n_heads!"
    head_size = gpt_config.n_embd // gpt_config.n_heads
    print(f"\nhead_size = {gpt_config.n_embd} / {gpt_config.n_heads} = {head_size} (OK)")
