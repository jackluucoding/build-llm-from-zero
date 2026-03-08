# Chapter 12: Dataset and DataLoader

> **Code file**: `src/ch11_dataloader.py`
> **Run it**: `python src/ch11_dataloader.py`

---

## Theory

### The Dishwasher Problem

Imagine washing 1 million dishes. You could wash them one at a time, but that's slow. Or you could load the dishwasher with 32 at a time, much faster, because the machine handles them in parallel.

Training a neural network is the same. Instead of processing one example at a time, we process a **batch** of 32 examples simultaneously. PyTorch runs all 32 through the model at once using parallelized matrix operations.

This is why our model works on tensors of shape `(B, T, C)`, the `B` dimension is the batch size.

---

### What Is a Training Example?

For our language model, one training example = one text window:
- **Input** `x`: a sequence of `block_size = 128` token IDs
- **Target** `y`: the same sequence, shifted right by 1

```
Text  : "To be or not to be, that is the question"
x     : "To be or not to be, that is the questio"   ← tokens 0..127
y     : "o be or not to be, that is the question"   ← tokens 1..128
```

At each of the 128 positions, `x[i]` is the input context and `y[i]` is what we're trying to predict.

---

### Sliding Windows

We create training examples by sliding a window of size `block_size` across the full text:

```
Position 0:   "To be or not to be..."  (chars 0-127)
Position 1:   "o be or not to be,..."  (chars 1-128)
Position 2:   " be or not to be, t..."  (chars 2-129)
...
```

This gives us over **1 million** overlapping training examples from the ~1.1 million characters of Shakespeare. There's plenty of data!

---

### The PyTorch Dataset and DataLoader

PyTorch provides two abstractions:

**`Dataset`**: An object that knows how many examples exist and can return any one of them by index.

The two methods below use Python's special "dunder" (double underscore) naming: `__len__` and `__getitem__`. Python has dozens of these special methods - they are automatically called behind the scenes when you use built-in operations:
- `len(my_dataset)` automatically calls `my_dataset.__len__()`
- `my_dataset[42]` automatically calls `my_dataset.__getitem__(42)`

By defining these two methods, our `TextDataset` class works with all Python and PyTorch code that expects a sequence-like object.

```python
class TextDataset(Dataset):
    def __len__(self):
        return len(self.data) - self.block_size   # ~1 million examples
    def __getitem__(self, idx):
        x = self.data[idx   : idx + block_size]
        y = self.data[idx+1 : idx + block_size + 1]
        return x, y
```

**`DataLoader`**: Wraps a Dataset and automatically:
- Groups examples into batches of size `batch_size = 32`
- Shuffles the data each epoch (one epoch = one full pass through all training examples; after seeing every example once, we shuffle and start again)
- Handles edge cases

```python
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

Each call to `next(iter(loader))` gives us one batch: `(x, y)` of shape `(32, 128)`.

Note: these are raw token IDs, so the shape is `(B, T)` not `(B, T, C)`. When the model receives this batch in Chapter 12, the embedding layer is the first thing it runs - that is what expands the shape from `(B, T)` to `(B, T, C)`. Here is what that expansion means: we start with (32, 128) - 32 sequences of 128 token IDs, each token represented as a single integer. The embedding layer replaces each integer with a full 128-dimensional vector (learned in Chapter 4). Now each token is described by 128 numbers instead of 1. Result: shape (32, 128, 128) = 32 sequences, 128 tokens each, each token a 128-number vector.

---

### Why Shuffle?

If examples always appear in the same order, the model might learn "character X tends to follow Y in this dataset" based on the dataset's sequential structure rather than actual language patterns. Shuffling forces the model to rely only on genuine patterns within each 128-token window, not on where that window happens to sit in the file.

---

## Code

> **File**: `src/ch11_dataloader.py`
> **Run it**: `python src/ch11_dataloader.py`

This file:
1. Loads and tokenizes Shakespeare
2. Implements `TextDataset`
3. Creates train and validation `DataLoader`s
4. Inspects one batch and decodes it back to text

### Sample output

```
x (input)  : "d the time seems thirty unto me,\nBeing all this time aband..."
y (target) : " the time seems thirty unto me,\nBeing all this time abando..."
(y is x shifted by 1 character)
```

Notice: `y` is just `x` with the first character removed and one new character appended at the end.

---

## Key Takeaways

- A training batch = 32 text windows, each 128 tokens long.
- Input `x` and target `y` are the same window, shifted by 1.
- `Dataset` defines how to get one example; `DataLoader` batches and shuffles.
- Shuffling prevents the model from memorizing order.
- ~1 million training examples from 1 million characters of Shakespeare.

---

*Next up: [Chapter 13, The Training Loop](ch13-training-loop.md)*
