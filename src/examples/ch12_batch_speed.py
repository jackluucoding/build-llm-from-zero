"""
Time 32 sequences run one at a time against the same 32 run as one batch.
This file belongs to Chapter 12.
Run: python src/examples/ch12_batch_speed.py
"""
import os
import sys
import time

import torch

# Let Python find the book's own modules in src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.ch09_gpt_model import GPT
from src.utils.config import GPTConfig

cfg = GPTConfig()
torch.manual_seed(42)
model = GPT(cfg)
batch = torch.randint(0, cfg.vocab_size, (32, cfg.block_size))

with torch.no_grad():
    start = time.perf_counter()
    for row in batch:                       # one sequence at a time
        model(row.unsqueeze(0))
    one_at_a_time = time.perf_counter() - start

    start = time.perf_counter()
    model(batch)                            # all 32 together
    as_one_batch = time.perf_counter() - start

print(f"32 sequences, one at a time : {one_at_a_time:.3f} s")
print(f"the same 32 as one batch    : {as_one_batch:.3f} s")
print(f"batching is {one_at_a_time / as_one_batch:.1f}x faster on this machine")
