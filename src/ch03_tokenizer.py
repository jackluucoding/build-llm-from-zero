"""
ch03_tokenizer.py - Tokenization

Neural networks can only work with numbers, not letters.
Tokenization is the process of converting text into numbers.

In this tutorial we use character-level tokenization:
  - We build a list of every unique character in our dataset.
  - Each character gets a unique integer ID.
  - "Hello" -> [20, 17, 30, 30, 33]  (the exact IDs depend on the vocab)

This is the simplest possible tokenizer. Real models like GPT-4 use
"Byte Pair Encoding" (BPE) which groups common sequences of characters
into single tokens - but character-level is much easier to understand.

Run this file:
    python src/ch03_tokenizer.py
"""

import os
import sys

# Make sure we can import from src/utils/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "shakespeare.txt")

print("=" * 50)
print("Chapter 3: Tokenization")
print("=" * 50)

# ------------------------------------------------------------------
# 1. Load the text
# ------------------------------------------------------------------
if not os.path.exists(DATA_PATH):
    print("ERROR: shakespeare.txt not found.")
    print("Please run:  python src/utils/download_data.py")
    sys.exit(1)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

print(f"\nLoaded {len(text):,} characters from shakespeare.txt")
print(f"First 100 chars: {repr(text[:100])}")

# ------------------------------------------------------------------
# 2. Build the vocabulary
# ------------------------------------------------------------------
print("\n--- Building the vocabulary ---")

# Find every unique character in the text.
# sorted() puts them in a consistent, predictable order.
chars = sorted(set(text))
vocab_size = len(chars)

print(f"Unique characters ({vocab_size} total):")
print("  " + "".join(chars))

# ------------------------------------------------------------------
# 3. Create the encode and decode functions
# ------------------------------------------------------------------
print("\n--- Creating encode() and decode() ---")

# stoi: String TO Integer - maps each character to a number
# itos: Integer TO String - maps each number back to a character
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(text):
    """Convert a string into a list of integers."""
    return [stoi[ch] for ch in text]

def decode(ids):
    """Convert a list of integers back into a string."""
    return "".join([itos[i] for i in ids])

# ------------------------------------------------------------------
# 4. Test the tokenizer
# ------------------------------------------------------------------
print("\n--- Testing encode / decode ---")

sample = "Hello, World!"
encoded = encode(sample)
decoded = decode(encoded)

print(f"Original : {repr(sample)}")
print(f"Encoded  : {encoded}")
print(f"Decoded  : {repr(decoded)}")
print(f"Round-trip matches: {sample == decoded}")

# ------------------------------------------------------------------
# 5. Encode the full dataset
# ------------------------------------------------------------------
print("\n--- Encoding the full dataset ---")

import torch

all_ids = encode(text)
data = torch.tensor(all_ids, dtype=torch.long)
# dtype=torch.long means 64-bit integers - needed for embedding lookups

print(f"Full dataset as tensor: shape={data.shape}, dtype={data.dtype}")
print(f"First 20 token IDs: {data[:20].tolist()}")
print(f"Decoded back: {repr(decode(data[:20].tolist()))}")

# ------------------------------------------------------------------
# 6. Train / validation split
# ------------------------------------------------------------------
print("\n--- Splitting into train and validation sets ---")

split = int(0.9 * len(data))        # 90% for training
train_data = data[:split]
val_data   = data[split:]

print(f"Training tokens  : {len(train_data):,}")
print(f"Validation tokens: {len(val_data):,}")

print("\nTokenizer ready! Ready for Chapter 4.")
