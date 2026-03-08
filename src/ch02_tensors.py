"""
ch02_tensors.py - Tensors and PyTorch Basics

A tensor is just a multi-dimensional array of numbers.
Think of it like a spreadsheet:
  - A 1D tensor is a single row of numbers: [1, 2, 3]
  - A 2D tensor is a table (rows x columns): like a matrix
  - A 3D tensor is a stack of tables

PyTorch tensors are like Python lists, but much faster for math
because they can run many operations in parallel.

Run this file:
    python src/ch02_tensors.py
"""

import torch

print("=" * 50)
print("Chapter 2: Tensors and PyTorch Basics")
print("=" * 50)

# ------------------------------------------------------------------
# 1. Creating tensors
# ------------------------------------------------------------------
print("\n--- 1. Creating tensors ---")

# A 1D tensor (like a list of numbers)
a = torch.tensor([1.0, 2.0, 3.0, 4.0])
print(f"1D tensor: {a}")
print(f"  shape: {a.shape}")   # shape tells us the size in each dimension

# A 2D tensor (like a table with 2 rows and 3 columns)
b = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
print(f"\n2D tensor:\n{b}")
print(f"  shape: {b.shape}")   # (2, 3) means 2 rows, 3 columns

# A 2D tensor filled with zeros
zeros = torch.zeros(3, 4)
print(f"\nZeros (3x4):\n{zeros}")

# A 2D tensor filled with random numbers
rand = torch.randn(2, 3)        # randn = random values from a bell curve
print(f"\nRandom (2x3):\n{rand}")

# ------------------------------------------------------------------
# 2. Shapes and dimensions
# ------------------------------------------------------------------
print("\n--- 2. Shapes and dimensions ---")

# In this tutorial we use three dimensions a lot: (B, T, C)
#   B = Batch size   - how many examples we process at once
#   T = Time (tokens)- how many tokens in a sequence
#   C = Channels     - how many numbers represent each token

B, T, C = 2, 5, 8   # a tiny example: 2 sequences, 5 tokens each, 8 numbers per token
x = torch.randn(B, T, C)
print(f"x shape: {x.shape}  - (batch={B}, time={T}, channels={C})")
print(f"x[0] is the first sequence, shape: {x[0].shape}")
print(f"x[0, 2] is the 3rd token of the 1st sequence, shape: {x[0, 2].shape}")

# ------------------------------------------------------------------
# 3. Reshaping
# ------------------------------------------------------------------
print("\n--- 3. Reshaping ---")

# Reshape changes how the numbers are grouped, without changing them.
# It is like re-folding a piece of paper.

flat = torch.arange(12, dtype=torch.float)  # [0, 1, 2, ..., 11]
print(f"Flat (12 numbers): {flat}")

grid = flat.reshape(3, 4)   # re-arrange into 3 rows of 4
print(f"\nReshaped to (3, 4):\n{grid}")

back_to_flat = grid.reshape(-1)  # -1 means "figure out this dimension automatically"
print(f"\nBack to flat: {back_to_flat}")

# ------------------------------------------------------------------
# 4. Matrix multiplication
# ------------------------------------------------------------------
print("\n--- 4. Matrix multiplication ---")

# Matrix multiplication (matmul) is the core operation in neural networks.
# Think of it as: each row of A "interacts" with each column of B.
# Result shape: (rows of A) x (columns of B)

A = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])   # shape (2, 2)

B_mat = torch.tensor([[5.0, 6.0],
                      [7.0, 8.0]])   # shape (2, 2)

result = A @ B_mat               # @ is the matrix multiply operator in Python
print(f"A:\n{A}")
print(f"B:\n{B_mat}")
print(f"A @ B:\n{result}")
print(f"  shape: {result.shape}")

# In 3D tensors (B, T, C), @ automatically applies to the last two dims.
x = torch.randn(2, 5, 8)   # (B=2, T=5, C=8)
W = torch.randn(8, 4)       # weight matrix - projects C=8 down to 4
y = x @ W                   # (2, 5, 8) @ (8, 4) = (2, 5, 4)
print(f"\n3D matmul: {x.shape} @ {W.shape} = {y.shape}")

# ------------------------------------------------------------------
# 5. Key operations used in transformers
# ------------------------------------------------------------------
print("\n--- 5. Key operations used in transformers ---")

# Softmax: converts a list of any numbers into probabilities (sum to 1).
# Think of it as "which option am I most confident about?"
logits = torch.tensor([1.0, 2.0, 3.0])
probs = torch.softmax(logits, dim=0)
print(f"Softmax({logits.tolist()}) = {probs.tolist()}")
print(f"  Sum = {probs.sum():.4f}")   # always sums to 1.0

# Transpose: swap two dimensions (like rotating a table)
mat = torch.randn(3, 5)    # shape (3, 5)
mat_T = mat.transpose(-2, -1)  # swap last two dims -> shape (5, 3)
print(f"\nTranspose: {mat.shape} -> {mat_T.shape}")

print("\nAll tensor basics covered! Ready for Chapter 3.")
