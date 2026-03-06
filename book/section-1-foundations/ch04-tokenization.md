# Chapter 4: Tokenization

> **Code file**: `src/ch03_tokenizer.py`
> **Run it**: `python src/ch03_tokenizer.py`

---

## Theory

### The Problem: Neural Networks Are Number Machines

Neural networks can only work with numbers. They can't eat text directly, it's like trying to feed a dog a photograph of a steak. Technically related, completely useless.

So we need to convert text into numbers. That process is called **tokenization**.

---

### Our Approach: Character-Level Tokenization

The simplest tokenizer imaginable:
1. Find every unique character in the text.
2. Assign each character a unique integer.
3. To encode text, replace each character with its integer.
4. To decode, do the reverse.

For example, with a tiny vocabulary `{a: 0, b: 1, c: 2}`:
```
encode("cab") = [2, 0, 1]
decode([2, 0, 1]) = "cab"
```

That's literally it.

---

### Building the Vocabulary

The Shakespeare dataset contains exactly **65 unique characters**:
- Lowercase letters: `a` through `z` (26)
- Uppercase letters: `A` through `Z` (26)
- Digits: `3` (only one digit appears in all of Shakespeare - most numbers were written as words back then)
- Punctuation: ` `, `!`, `$`, `&`, `'`, `,`, `-`, `.`, `:`, `;`, `?`
- Newlines

We sort them to get a consistent ordering:
```python
chars = sorted(set(text))   # ['\\n', ' ', '!', '$', '&', "'", ',', '-', '.', ...]
stoi  = {ch: i for i, ch in enumerate(chars)}   # string to integer
itos  = {i: ch for i, ch in enumerate(chars)}   # integer to string
```

---

### Why Character-Level?

Other tokenization approaches exist:

**Word-level**: Each word is one token. Problem: huge vocabulary (50,000+ words), rare words not handled well, punctuation messiness.

**Byte-Pair Encoding (BPE)**: Groups frequently co-occurring characters into single tokens. For example, `th`, `ing`, and `tion` each become one token instead of 2-4 characters. GPT-4 uses this. Problem: requires a separate training pass, complex to implement.

**Character-level**: Simple, transparent, zero dependencies. Vocabulary of only 65. Perfect for learning.

The downside: longer sequences. "Hello" = 5 tokens character-level, but often 1 token with BPE. With our `block_size=128`, we can only "see" 128 characters at once instead of perhaps 128 words.

For a tutorial laptop model, that's totally fine.

---

### The Vocabulary Is Learned From the Data

This is subtle but important: our tokenizer is *built from the training data*. We don't import a dictionary, we look at the actual text and extract the vocabulary.

If a character never appears in Shakespeare, it has no token ID. Our tokenizer only knows the 65 characters in the training data. Feed it an emoji or a Chinese character and it tries to look the character up in the `stoi` dictionary. It finds nothing, so Python raises a `KeyError: <character not found>` and your program crashes.

Production systems handle this with a special `<UNK>` (unknown) token as a fallback - any unseen character maps to `<UNK>` instead of crashing. But for our tutorial, sticking to Shakespeare keeps it simple.

---

## Code

> **File**: `src/ch03_tokenizer.py`
> **Run it**: `python src/ch03_tokenizer.py`

This file:
1. Loads `shakespeare.txt`
2. Builds the 65-character vocabulary
3. Creates `encode()` and `decode()` functions
4. Demonstrates a round-trip: text → integers → text
5. Encodes the entire dataset as a PyTorch tensor
6. Splits it into training (90%) and validation (10%) sets

### Sample output

```
Original : 'Hello, World!'
Encoded  : [20, 43, 50, 50, 53, 6, 1, 35, 53, 56, 50, 42, 2]
Decoded  : 'Hello, World!'
Round-trip matches: True
```

The exact numbers will match yours since we use `sorted()`, the vocabulary ordering is deterministic.

---

### Training vs Validation Split

We split the data 90%/10%:
- **Training set**: The model learns from this (sees it during training).
- **Validation set**: We test on this to check if the model is really learning patterns, or just memorizing.

If training loss goes down but validation loss stays high, the model is **overfitting** (memorizing rather than learning). For example: training loss 1.3, validation loss 2.5 means the model has memorized specific character sequences from the training text, but fails on new text it has never seen before.

Imagine memorizing every answer to a practice test without understanding the material. You'd ace the practice test but fail the real exam. That's overfitting. Validation loss is our surprise quiz on material the model has never studied.

---

## Key Takeaways

- Tokenization converts text to integers so neural networks can process it.
- We use character-level tokenization: each character gets a unique integer ID.
- `encode(text)` converts string to list of ints; `decode(ids)` does the reverse.
- The vocabulary (65 characters) is extracted directly from the dataset.
- We split data 90/10 into train and validation sets.

---

*Next up: [Chapter 5, Embeddings](ch05-embeddings.md)*
