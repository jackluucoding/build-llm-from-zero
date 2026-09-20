"""Shows what character follows the string 'the ' in Shakespeare."""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Read the dataset
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "shakespeare.txt")
with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

# Find what follows "the "
prefix = "the "
next_chars = []
for i in range(len(text) - len(prefix)):
    if text[i:i+len(prefix)].lower() == prefix:
        next_chars.append(text[i+len(prefix)])

# Count and print the top 5
counts = Counter(next_chars)
print(f"What follows '{prefix}' in Shakespeare?")
for char, count in counts.most_common(5):
    display_char = repr(char) if char == ' ' or char == '\n' else char
    print(f"'{display_char}': {count} times")
