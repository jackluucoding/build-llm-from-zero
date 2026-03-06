# Chapter 5: Embeddings

> **Code file**: `src/ch04_embeddings.py`
> **Run it**: `python src/ch04_embeddings.py`

---

## Theory

### Why Can't We Just Use the Token IDs?

After tokenization, each character is an integer: `A=33`, `B=34`, `a=39`, etc.

But using these raw integers as input to a neural network is a problem. The number `39` for `a` doesn't mean `a` is "more" than `A` (number 33). There's no mathematical relationship between these integers that mirrors the relationships between characters.

Here is what goes wrong: neural networks learn by multiplying numbers together. If 'A' is token 33 and 'Z' is token 64, the network sees 'Z' as a bigger number than 'A' - almost twice as big. It might learn that 'Z' is more important than 'A' just because of this accident. But token IDs are assigned arbitrarily - 'Z' could have been token 1 instead. The ordering is meaningless.

Feeding raw token IDs to a neural network is like telling someone the Dewey Decimal numbers of books and expecting them to understand literature. Or, if you've never used a library catalog: it's like texting a friend "I want to talk about item number 39" without saying what item 39 is. The number alone carries no meaning.

---

### The Embedding Table

The solution: a **lookup table** that maps each token ID to a vector of floating-point numbers.

```
token ID 33  →  [0.12, -0.45, 0.83, ...]   ← 128 numbers
token ID 34  →  [-0.22, 0.71, -0.34, ...]  ← 128 numbers
token ID 39  →  [0.55, 0.03, -0.89, ...]   ← 128 numbers
```

These 128 numbers (the "embedding vector") are what the model actually works with. They're initialized randomly but **learned during training**: the model adjusts them using gradient descent, the same mechanism that trains everything else. Tokens that appear in similar contexts will gradually develop similar vectors.

In PyTorch:
```python
embedding = nn.Embedding(vocab_size=65, embedding_dim=128)
```

This creates a table of shape `(65, 128)`, 65 rows (one per character), 128 columns (one per embedding dimension).

When we pass token IDs, PyTorch simply looks up the corresponding rows:
```
input:  [33, 39]     ← token IDs for 'Aa'
output: [[0.12, -0.45, ...],   ← row 33 from the table
         [0.55, 0.03, ...]]    ← row 39 from the table
```

Shape change: `(B, T)` → `(B, T, C)` where `C = n_embd = 128`.

---

### What Do Embeddings Learn?

After training, similar characters tend to have similar embeddings. Characters that appear in similar contexts end up with nearby vectors in the 128-dimensional space.

For words (not characters), this is even more interesting. Trained models learn that `king - man + woman` is very close to `queen`. Here is why: 'king' and 'queen' appear in similar contexts (palaces, thrones, crowns), so their embedding vectors point in similar directions. 'King' and 'man' appear in different patterns ('a man walked' vs 'a king ruled'), so their vectors differ in a specific way. Subtracting that difference and adding 'woman' bridges from 'king' to 'queen' in the vector space. Our character model learns simpler patterns (e.g., capital and lowercase versions of the same letter tend to cluster together).

Think of embeddings as assigning each character a "personality profile" of 128 numbers. The model learns these profiles during training.

---

### Positional Embeddings

Here's a problem: the transformer doesn't inherently know the **order** of tokens.

Unlike reading a sentence word-by-word from left to right, the transformer processes all tokens at the same time, in parallel. It looks at the entire set of tokens at once. Without position information, "cat eats dog" and "dog eats cat" look identical: just three tokens, shuffled. Order matters, so we need to tell the model where each token sits.

We fix this with **positional embeddings**: a second lookup table indexed by *position* rather than token ID.

```python
pos_embedding = nn.Embedding(block_size=128, embedding_dim=128)
```

Position 0 gets its own 128-number vector, position 1 gets a different 128-number vector, and so on - up to position 127.

We **add** the positional embedding to the token embedding (rather than, say, concatenating them) because addition keeps the same 128 dimensions while mixing both pieces of information together. The model learns to use those shared 128 numbers to represent both 'what token is this?' and 'where does it appear?'.

```
Final token representation = token_embedding + position_embedding
```

Shape: `(B, T, C)` + `(T, C)` broadcast = `(B, T, C)`

---

### Learned vs. Fixed Positional Encodings

There are two approaches:

**Fixed**: Use a mathematical formula to assign each position a unique fingerprint - position 0 gets one pattern, position 1 gets a different pattern, position 2 another, and so on. These fingerprints never change during training. The original Transformer paper used this approach.

**Learned**: Treat positions like tokens - give each position an ID (0, 1, 2, ...) and let the model learn a lookup table of position embeddings, just like it learns token embeddings. GPT-2 uses this. We use this too.

For our small model, the difference is negligible. Learned embeddings are simpler to implement.

---

## Code

> **File**: `src/ch04_embeddings.py`
> **Run it**: `python src/ch04_embeddings.py`

This file:
1. Creates a token embedding table `(65, 128)`
2. Demonstrates lookup: `(B, T)` → `(B, T, C)`
3. Creates a position embedding table `(128, 128)`
4. Combines token + position embeddings
5. Prints a parameter count for the embedding layers

### Key line to understand

```python
x = token_embeddings + position_embeddings   # (B, T, C)
```

This `x` is the **input to the transformer blocks**. Every chapter after this operates on this 3D tensor.

---

## Key Takeaways

- Embeddings are learned lookup tables that map token IDs to float vectors.
- Shape change: `(B, T)` → `(B, T, C)` where `C = n_embd = 128`.
- Position embeddings encode the position of each token (0, 1, 2, ...).
- The final input to the transformer = token embedding + position embedding.
- Both embedding tables are learned from scratch during training.

---

*Next up: [Chapter 6, Self-Attention](../section-2-attention/ch06-self-attention.md)*
