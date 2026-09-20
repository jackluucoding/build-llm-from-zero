"""Show how a sliding window creates multiple overlapping examples."""
import torch

data = torch.tensor([10, 20, 30, 40, 50, 60, 70])
block_size = 3

print(f"Data: {data.tolist()}")
print(f"Block size: {block_size}\n")

# A simple loop to show the sliding window
for i in range(len(data) - block_size):
    x = data[i : i + block_size]
    y = data[i + 1 : i + block_size + 1]
    print(f"Example {i+1}:")
    print(f"  Input  : {x.tolist()}")
    print(f"  Target : {y.tolist()}")
