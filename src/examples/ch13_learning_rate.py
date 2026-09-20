"""
Train the same model three times with three learning rates, and compare.
This file belongs to Chapter 13.
Run: python src/examples/ch13_learning_rate.py
"""
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Let Python find the book's own modules in src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.ch09_gpt_model import GPT
from src.ch11_dataloader import train_dataset
from src.utils.config import GPTConfig, TrainConfig

STEPS = 150

for lr in (3e-3, 3e-4, 3e-5):
    torch.manual_seed(42)                       # same starting weights every time
    model = GPT(GPTConfig())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = iter(DataLoader(train_dataset, batch_size=TrainConfig().batch_size,
                             shuffle=True))

    first = last = None
    for step in range(STEPS):
        x, y = next(loader)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, GPTConfig().vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if first is None:
            first = loss.item()
        last = loss.item()

    print(f"lr={lr:<8g} start {first:.2f}  after {STEPS} steps {last:.2f}")
