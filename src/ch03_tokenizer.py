"""
Build a character-level tokenizer.
This file belongs to Chapter 4.
Run: python src/ch03_tokenizer.py
"""
import os
import sys
import torch

# Settings
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "shakespeare.txt")

# --- The Idea ---

def build_vocab(text):
    # Find all unique characters in the text
    chars = sorted(set(text))

    # Dictionaries to translate between characters and their integer IDs
    char_to_id = {ch: i for i, ch in enumerate(chars)}
    id_to_char = {i: ch for i, ch in enumerate(chars)}

    return chars, char_to_id, id_to_char

def encode(text, char_to_id):
    # Convert a string into a list of integer IDs
    return [char_to_id[ch] for ch in text]

def decode(ids, id_to_char):
    # Convert a list of integer IDs back into a string
    return "".join([id_to_char[i] for i in ids])

# --- Demo ---
if __name__ == "__main__":
    print("Chapter 4: Tokenization\n")

    if not os.path.exists(DATA_PATH):
        print("ERROR: shakespeare.txt not found.")
        print("Please run:  python src/utils/download_data.py")
        sys.exit(1)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Loaded {len(text):,} characters from shakespeare.txt")
    print(f"First 100 chars: {repr(text[:100])}")

    print("\n--- Building the vocabulary ---")
    chars, char_to_id, id_to_char = build_vocab(text)
    vocab_size = len(chars)
    print(f"Unique characters ({vocab_size} total):\n  " + "".join(chars))

    print("\n--- Testing encode / decode ---")
    sample = "Hello, World!"
    encoded = encode(sample, char_to_id)
    decoded = decode(encoded, id_to_char)

    print(f"Original : {repr(sample)}")
    print(f"Encoded  : {encoded}")
    print(f"Decoded  : {repr(decoded)}")
    print(f"Round-trip matches: {sample == decoded}")

    print("\n--- Encoding the full dataset ---")
    all_ids = encode(text, char_to_id)

    # Store IDs in a PyTorch tensor for efficient model training
    data = torch.tensor(all_ids, dtype=torch.long)

    print(f"Full dataset as tensor: shape={data.shape}, dtype={data.dtype}")
    print(f"First 20 token IDs: {data[:20].tolist()}")
    print(f"Decoded back: {repr(decode(data[:20].tolist(), id_to_char))}")

    print("\n--- Splitting into train and validation sets ---")
    # Use 90% for training and 10% for validation to evaluate performance
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data   = data[split:]

    print(f"Training tokens  : {len(train_data):,}")
    print(f"Validation tokens: {len(val_data):,}")

    print("\nTokenizer ready! Ready for Chapter 5.")
