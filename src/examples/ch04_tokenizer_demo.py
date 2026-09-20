"""Shows how different text inputs map to token IDs."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.ch03_tokenizer import build_vocab, encode

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "shakespeare.txt")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

chars, char_to_id, _ = build_vocab(text)

print("Demo: Encoding short business phrases")
phrases = [
    "invoice",
    "URGENT!",
    "Hello."
]

for p in phrases:
    # Notice how spaces and punctuation get their own IDs
    print(f"{repr(p):<15} -> {encode(p, char_to_id)}")
