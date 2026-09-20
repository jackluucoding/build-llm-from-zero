"""
Count the numbers inside the heads: four small heads against one big one.
This file belongs to Chapter 7.
Run: python src/examples/ch07_head_budget.py
"""
C, n_heads = 128, 4                      # channels, and how many heads we split them into
head_size = C // n_heads

# Every head holds three filters (query, key, value), each C by head_size
four_heads = n_heads * 3 * C * head_size
one_head = 3 * C * C

print(f"{n_heads} heads of {head_size}: {four_heads:,} numbers")
print(f"1 head of {C}  : {one_head:,} numbers")
print(f"Same budget: {four_heads == one_head}")
