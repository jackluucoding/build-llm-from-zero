"""
Get the Tiny Shakespeare dataset.
This file belongs to Chapter 1.
Run: python src/utils/download_data.py

The text ships with this repository, so cloning it is enough and nothing is
downloaded. If the file is missing it is fetched from this repository's own
copy, and checked against a known fingerprint, so a truncated or altered
download cannot pass unnoticed.

The text was assembled by Andrej Karpathy for the char-rnn project
(https://github.com/karpathy/char-rnn); the plays themselves are public domain.
"""
import hashlib
import os

import requests

# Settings
DATA_URL = ("https://raw.githubusercontent.com/jackluucoding/"
            "build-llm-from-zero/main/src/data/shakespeare.txt")
SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
DATA_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "shakespeare.txt")
)

# --- The Idea ---

def shown(path):
    # The path as the reader sees it from the project folder
    root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.relpath(path, root).replace(os.sep, "/")

def fingerprint(path):
    # A short, fixed summary of the file's bytes. Line endings are normalized
    # first, so a copy that Git checked out with Windows line endings still passes.
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()

def download_shakespeare():
    # Already here and unchanged? Nothing to do.
    if os.path.exists(DATA_PATH) and fingerprint(DATA_PATH) == SHA256:
        print(f"Dataset ready: {shown(DATA_PATH)} ({os.path.getsize(DATA_PATH) // 1024} KB)")
        return DATA_PATH

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    print(f"Fetching the dataset...")
    print(f"  Source : {DATA_URL}")
    print(f"  Saving : {shown(DATA_PATH)}")

    response = requests.get(DATA_URL, timeout=60)
    response.raise_for_status()
    with open(DATA_PATH, "wb") as f:
        f.write(response.content)

    if fingerprint(DATA_PATH) != SHA256:
        raise SystemExit("The downloaded file does not match the expected fingerprint.")

    print(f"  Done! ({os.path.getsize(DATA_PATH) // 1024} KB)")
    return DATA_PATH

# --- Demo ---
if __name__ == "__main__":
    path = download_shakespeare()

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"\nDataset statistics:")
    print(f"  Total characters : {len(text):,}")
    print(f"  Unique characters: {len(set(text))}")
    print(f"\nFirst 200 characters:")
    print("-" * 40)
    print(text[:200])
    print("-" * 40)
