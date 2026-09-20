# Chapter 12: Dataset and DataLoader: Exercises

Read the chapter online: https://jackluu.io/book/section-4-training/ch12-dataset-and-dataloader/

## Try It

We can see the sliding window in action with a tiny dataset.

```python
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
```

Lines 12 and 13 slice the array to create overlapping sequences for the input and target.

```console
$ python src/examples/ch12_batching_demo.py
Data: [10, 20, 30, 40, 50, 60, 70]
Block size: 3

Example 1:
  Input  : [10, 20, 30]
  Target : [20, 30, 40]
Example 2:
  Input  : [20, 30, 40]
  Target : [30, 40, 50]
Example 3:
  Input  : [30, 40, 50]
  Target : [40, 50, 60]
Example 4:
  Input  : [40, 50, 60]
  Target : [50, 60, 70]

```

Open `src/examples/ch12_batching_demo.py`. Change `block_size` to 4 and run it again. Notice how the number of available examples decreases.

**In Business**
Imagine your house-style assistant needs to learn from a massive archive of 100,000 corporate documents. You wouldn't train it by showing it one word at a time. Processing data in parallel batches is like having the assistant review 32 different emails simultaneously, learning from all of them at once. It is the key to training efficiently at scale.

**Watch Out**
Be careful with the `__len__` of your dataset. If you have 100 characters and a `block_size` of 10, you can only create 90 starting positions because you need 11 characters (10 for input, 1 extra for the target) for a valid example. That is why the code uses `len(self.data) - self.block_size`.

## Check Your Understanding

1. If you have a sequence of 1000 tokens and a block size of 100, how many examples can a sliding window extract?
2. Why is batching important for training speed?
3. Why do we shuffle the training data?
