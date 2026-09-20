"""
Introduce basic tensor operations and shapes.
This file belongs to Chapter 3.
Run: python src/ch02_tensors.py
"""
import torch

if __name__ == "__main__":
    torch.manual_seed(42)
    print("Chapter 3: Tensors and PyTorch Basics\n")

    print("--- 1. Creating tensors ---")
    # A 1D tensor is simply a list of numbers
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    print(f"1D tensor: {a}\n  shape: {a.shape}")

    # A 2D tensor represents a table or matrix of numbers
    b = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(f"\n2D tensor:\n{b}\n  shape: {b.shape}")

    zeros = torch.zeros(3, 4)
    print(f"\nZeros (3x4):\n{zeros}")

    rand = torch.randn(2, 3)
    print(f"\nRandom (2x3):\n{rand}")

    print("\n--- 2. Shapes and dimensions ---")
    B, T, C = 2, 5, 8
    x = torch.randn(B, T, C)
    print(f"x shape: {x.shape}  - (batch={B}, time={T}, channels={C})")
    print(f"x[0] is the first sequence, shape: {x[0].shape}")
    print(
        f"x[0, 2] is the 3rd token of the 1st sequence, shape: {x[0, 2].shape}"
    )

    print("\n--- 3. Reshaping ---")
    flat = torch.arange(12, dtype=torch.float)
    print(f"Flat (12 numbers): {flat}")

    # Reshape changes the layout but keeps the total number of items the same
    grid = flat.reshape(3, 4)
    print(f"\nReshaped to (3, 4):\n{grid}")

    back_to_flat = grid.reshape(-1)
    print(f"\nBack to flat: {back_to_flat}")

    print("\n--- 4. Matrix multiplication ---")
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    B_mat = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    # The @ symbol performs matrix multiplication
    result = A @ B_mat
    print(f"A:\n{A}\nB:\n{B_mat}\nA @ B:\n{result}\n  shape: {result.shape}")

    x = torch.randn(2, 5, 8)
    W = torch.randn(8, 4)
    y = x @ W
    print(f"\n3D matmul: {x.shape} @ {W.shape} = {y.shape}")

    print("\n--- 5. Key operations used in transformers ---")
    logits = torch.tensor([1.0, 2.0, 3.0])

    # Softmax turns raw scores into probabilities that sum to 1
    probs = torch.softmax(logits, dim=0)
    probs_str = "[" + ", ".join(f"{p:.4f}" for p in probs) + "]"
    print(f"Softmax({logits.tolist()}) = {probs_str}")
    print(f"  Sum = {probs.sum():.4f}")

    mat = torch.randn(3, 5)
    
    # Transpose flips the axes, turning rows into columns and vice-versa
    mat_T = mat.transpose(-2, -1)
    print(f"\nTranspose: {mat.shape} -> {mat_T.shape}")

    print("\nReady for Chapter 4.")
