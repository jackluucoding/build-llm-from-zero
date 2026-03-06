# Chapter 3: Tensors and PyTorch Basics

> **Code file**: `src/ch02_tensors.py`
> **Run it**: `python src/ch02_tensors.py`

---

## Theory

### What Is PyTorch?

PyTorch is a Python library for doing math on large arrays of numbers, really, really fast.

You might ask: "Can't we just use regular Python lists?" Yes, technically. But consider: a single 1000x1000 matrix multiply requires 1 billion individual multiplications (1000 x 1000 x 1000). Pure Python would perform these one at a time, taking several seconds per operation. PyTorch uses optimized C++ code under the hood that performs thousands of multiplications simultaneously, completing the same operation in microseconds. Our model does millions of such operations during training. Without PyTorch, training would take weeks instead of minutes.

Think of PyTorch as a turbo-charged calculator.

---

### What Is a Tensor?

A **tensor** is the fundamental data type in PyTorch. It's a multi-dimensional array of numbers.

If that sounds scary, don't worry. You already know tensors:

- A **0D tensor** (scalar) is just one number: `3.14`
- A **1D tensor** (vector) is a list of numbers: `[1, 2, 3, 4]`
- A **2D tensor** (matrix) is a table of numbers: like a spreadsheet
- A **3D tensor** is a stack of tables: like a spreadsheet with multiple sheets

```
1D: [1, 2, 3]                    ← a row

2D: [[1, 2, 3],                  ← a table
     [4, 5, 6]]

3D: [[[1, 2], [3, 4]],           ← a stack of tables
     [[5, 6], [7, 8]]]
```

In this tutorial, we work with 3D tensors a lot. Their dimensions represent:
- **B** = Batch size (how many text sequences we process at once)
- **T** = Time (how many tokens in each sequence)
- **C** = Channel size (embedding dimension: how many numbers we use to represent each token)

We write shapes like `(B, T, C)`, for example, `(32, 128, 128)` means "32 sequences, each 128 tokens long, each token described by 128 numbers."

---

### Key Operations

Here are the operations that appear throughout the tutorial:

**Creating tensors:**
```python
import torch
a = torch.tensor([1.0, 2.0, 3.0])       # from a Python list
b = torch.zeros(3, 4)                     # all zeros, shape (3, 4)
c = torch.randn(2, 5)                     # random values (bell curve)
```

**Checking the shape:**
```python
x = torch.randn(2, 5, 8)
print(x.shape)    # torch.Size([2, 5, 8])
```

**Reshaping:**
```python
x = torch.arange(12.0)   # [0, 1, 2, ..., 11]
y = x.reshape(3, 4)       # re-arrange into 3 rows of 4
# It's like re-folding a piece of paper: same numbers, different shape.
```

**Matrix multiplication (`@`):**
```python
A = torch.randn(3, 4)    # shape (3, 4)
B = torch.randn(4, 5)    # shape (4, 5)
C = A @ B                # shape (3, 5)
# Rule: (m, n) @ (n, p) = (m, p)
# The inner dimensions must match!
```

Why must the inner dimensions match? Think of (3, 4) as "3 students each with 4 test scores" and (4, 5) as "4 test scores each mapped to 5 skill ratings". The 4 scores in the first matrix match the 4 scores in the second - that's what lets you combine them. You couldn't combine (3, 4) and (5, 6) because there's a mismatch: 4 scores don't connect to 5 scores.

**Softmax**, converts any numbers into probabilities (they sum to 1):
```python
logits = torch.tensor([1.0, 2.0, 3.0])
probs  = torch.softmax(logits, dim=0)
# Output: [0.09, 0.24, 0.67]  ← sum = 1.0
# The biggest input (3.0) gets the biggest probability.
```

---

### Why Does Softmax Sum to 1?

Softmax uses the formula:

```
P(i) = exp(x_i) / sum(exp(x_j) for all j)
```

`exp` means e^x, where e is a special mathematical constant (approximately 2.718). Why use it? Two reasons:
1. `exp(x)` is always positive, so we never get negative probabilities.
2. `exp(x)` amplifies differences: `exp(3) = 20` and `exp(1) = 2.7`, so 3 becomes about 7x more dominant than 1 after exp. This means the highest-scoring option gets a meaningfully higher probability.

By dividing each `exp(x_i)` by the sum of all `exp(x_j)`, we guarantee everything sums to 1. This turns raw "scores" (called **logits**) into a probability distribution.

Think of it like: you and your friends vote on pizza toppings. Each person's enthusiasm is a logit. Softmax converts "enthusiasm scores" into "probability each topping wins."

---

## Code

> **File**: `src/ch02_tensors.py`
> **Run it**: `python src/ch02_tensors.py`

This file demonstrates:
1. Creating 1D, 2D, and 3D tensors
2. The `(B, T, C)` shape convention used throughout the tutorial
3. Reshaping
4. Matrix multiplication
5. Softmax and transpose

Run it and read the output carefully. Every operation here will appear again in the attention mechanism (Chapter 5).

### What to look for

When you run it, pay attention to the shape annotations. For example:

```
3D matmul: torch.Size([2, 5, 8]) @ torch.Size([8, 4]) = torch.Size([2, 5, 4])
```

PyTorch automatically handles the batch dimension (2) when doing matrix multiplication on 3D tensors. This is called **broadcasting**: PyTorch expands the smaller tensor's shape to match the larger one so the operation works without writing loops.

You could write a loop: `for i in range(batch_size): result[i] = sequences[i] @ weights`. But that's slow - Python executes each iteration one at a time. Broadcasting tells PyTorch to apply the operation to all 32 sequences at once using optimized parallel code. Same result, much faster.

---

## Key Takeaways

- A tensor is a multi-dimensional array of numbers.
- Shape `(B, T, C)` = batch x time x channel size, the standard convention in this tutorial.
- `@` is matrix multiplication. Inner dimensions must match.
- Softmax turns any numbers into probabilities that sum to 1.
- PyTorch handles batches automatically, no for-loops needed.

---

*Next up: [Chapter 4, Tokenization](ch04-tokenization.md)*
