"""
download_data.py - Downloads the Tiny Shakespeare dataset.

Run this once before starting the tutorial:
    python src/utils/download_data.py

The dataset (~1MB) will be saved to src/data/shakespeare.txt.
It contains the complete works of Shakespeare as plain text -
a classic dataset for character-level language model tutorials.
"""

import os
import requests

# URL of the Tiny Shakespeare dataset (hosted on GitHub, ~1 MB)
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# Where to save the file (relative to the project root)
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "shakespeare.txt")
DATA_PATH = os.path.normpath(DATA_PATH)


def download_shakespeare():
    """Download shakespeare.txt if it doesn't already exist."""

    # Check if already downloaded
    if os.path.exists(DATA_PATH):
        size_kb = os.path.getsize(DATA_PATH) // 1024
        print(f"Dataset already exists at: {DATA_PATH} ({size_kb} KB)")
        return DATA_PATH

    # Create the data directory if needed
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    print(f"Downloading Tiny Shakespeare dataset...")
    print(f"  Source : {DATA_URL}")
    print(f"  Saving : {DATA_PATH}")

    response = requests.get(DATA_URL)
    response.raise_for_status()  # Raise an error if download failed

    with open(DATA_PATH, "w", encoding="utf-8") as f:
        f.write(response.text)

    size_kb = os.path.getsize(DATA_PATH) // 1024
    print(f"  Done! ({size_kb} KB downloaded)")
    return DATA_PATH


if __name__ == "__main__":
    path = download_shakespeare()

    # Show a quick preview of the data
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"\nDataset statistics:")
    print(f"  Total characters : {len(text):,}")
    print(f"  Unique characters: {len(set(text))}")
    print(f"\nFirst 200 characters:")
    print("-" * 40)
    print(text[:200])
    print("-" * 40)
