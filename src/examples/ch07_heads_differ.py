"""
Show that the trained model's attention heads look at different characters.
This file belongs to Chapter 7, and needs the model trained in Chapter 13.
Run: python src/examples/ch07_heads_differ.py
"""
import os
import sys

import torch
import torch.nn.functional as F

# Let Python find the book's own modules in src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.ch03_tokenizer import build_vocab, encode
from src.ch09_gpt_model import GPT
from src.utils.config import GPTConfig

PROMPT = "JULIET: O Romeo"
CHECKPOINT = "checkpoints/model.pt"

if not os.path.exists(CHECKPOINT):
    sys.exit("No trained model yet. Train one first: python src/ch12_train.py")

text = open("src/data/shakespeare.txt", encoding="utf-8").read()
_, char_to_id, _ = build_vocab(text)

model = GPT(GPTConfig())
# weights_only=False because the checkpoint also stores the config object
checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# What the first block sees: token vectors plus position vectors, normalized
ids = torch.tensor([encode(PROMPT, char_to_id)])
x = model.blocks[0].ln1(model.token_emb(ids) + model.pos_emb(torch.arange(ids.shape[1])))

print(f'Prompt: "{PROMPT}"\n')
print("Attention paid by the last character, one row per head:")
print("          " + "".join(f"{c if c != ' ' else '_':>5}" for c in PROMPT))

sharpest = []
for h, head in enumerate(model.blocks[0].attn.heads):
    with torch.no_grad():
        scores = head.query(x) @ head.key(x).transpose(-2, -1) * head.head_size ** -0.5
        weights = F.softmax(scores, dim=-1)[0, -1]     # the last character's row
    print(f"  head {h}: " + "".join(f"{w:5.2f}" for w in weights.tolist()))
    sharpest.append(f"{h}->'{PROMPT[weights.argmax()]}' ({weights.max():.2f})")

print("\nSharpest focus per head: " + "  ".join(sharpest))
