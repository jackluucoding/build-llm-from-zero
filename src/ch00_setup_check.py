"""
ch00_setup_check.py - Environment verification for "Building an LLM from Zero"

Run this after completing the Chapter 0 setup to confirm everything is installed
and working correctly before starting Chapter 1.

Run it:
    python src/ch00_setup_check.py
"""

import sys
import os


def check(label, fn):
    """Run fn(), print [OK] with result or [FAIL] with error message."""
    try:
        result = fn()
        suffix = f": {result}" if result else ""
        print(f"[OK] {label}{suffix}")
        return True
    except Exception as e:
        print(f"[FAIL] {label}: {e}")
        return False


def check_python():
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 10):
        raise RuntimeError(f"Python 3.10+ required, found {v.major}.{v.minor}")
    return f"{v.major}.{v.minor}.{v.micro}"


def check_torch():
    import torch
    t = torch.tensor([1.0, 2.0, 3.0])
    assert t.sum().item() == 6.0, "Basic tensor math failed"
    return torch.__version__


def check_numpy():
    import numpy as np
    return np.__version__


def check_requests():
    import requests
    return requests.__version__


def check_dataset():
    # Try both relative and absolute paths (works whether run from project root or src/)
    candidates = [
        os.path.join("src", "data", "shakespeare.txt"),
        os.path.join(os.path.dirname(__file__), "data", "shakespeare.txt"),
    ]
    path = None
    for c in candidates:
        if os.path.exists(c):
            path = c
            break
    if path is None:
        raise FileNotFoundError(
            "shakespeare.txt not found. Run: python src/utils/download_data.py"
        )
    with open(path, encoding="utf-8") as f:
        chars = len(f.read())
    return f"{chars:,} characters"


def check_tensor_math():
    import torch
    # A quick sanity check: create a small matrix and verify matmul shape
    a = torch.randn(3, 4)
    b = torch.randn(4, 5)
    c = a @ b
    assert c.shape == (3, 5), f"Expected shape (3,5), got {c.shape}"
    return None  # no version string, just confirm it passed


def main():
    print("Checking your environment...\n")

    all_ok = True
    all_ok &= check("Python", check_python)
    all_ok &= check("PyTorch", check_torch)
    all_ok &= check("NumPy", check_numpy)
    all_ok &= check("Requests", check_requests)
    all_ok &= check("Shakespeare dataset found", check_dataset)
    all_ok &= check("Quick tensor test passed", check_tensor_math)

    print()
    if all_ok:
        print("All checks passed! You are ready to start Chapter 1.")
    else:
        print("Some checks failed. See the messages above and re-run the relevant setup steps.")
        print("Chapter 0 (book/section-1-foundations/ch00-environment-setup.md) has troubleshooting tips.")


if __name__ == "__main__":
    main()
