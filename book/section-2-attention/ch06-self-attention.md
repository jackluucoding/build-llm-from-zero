# Chapter 6: Self-Attention (Single Head)

> **Code file**: `src/ch05_self_attention.py`
> **Run it**: `python src/ch05_self_attention.py`

---

## Theory

### The Problem: Context Matters

Read this sentence: *"The trophy didn't fit in the suitcase because it was too big."*

What does "it" refer to, the trophy or the suitcase?

You figured it out in milliseconds. But how? You read the whole sentence and connected the word "it" to "trophy" because of the word "big", only the trophy could be too big to fit.

Simple word-by-word processing can't do this. Each word needs to "look at" other words and decide which ones are relevant to understanding its meaning. That's exactly what **self-attention** does.

---

### The Library Analogy: Q, K, V

Self-attention uses three concepts:

**Query (Q)**: What am I looking for?
**Key (K)**: What do I offer?
**Value (V)**: What information do I actually contain?

Imagine a library:
- You walk in with a question: "Books about ocean ecology." That's your **Query**.
- Every book has an index card describing what it's about. That's each book's **Key**.
- The actual text content of each book is its **Value**.

You compare your Query against every Key to find the best matches, then read (take a weighted mix of) the Values of the matching books.

In self-attention, every token simultaneously acts as a Query ("what do I need?"), a Key ("what do I have?"), and a Value ("what's my content?"), at the same time!

Let's make this concrete. Suppose we have the sequence: `[cat, eats, the, big, dog]`. When the model processes "eats":
- Its **Q** (query) asks: "What context do I need to understand my role here?"
- Its **K** (key) announces: "I'm a verb, an action word."
- Its **V** (value) holds: "Full details about me - a past-tense verb, an action."

The model then compares "eats" Q against every token's K:
- eats-Q vs. dog-K: "A noun is very relevant to a verb." High attention score.
- eats-Q vs. the-K: "An article is less relevant." Lower attention score.

So "eats" ends up borrowing more information from "dog" than from "the". That's attention in action.

---

### The Math, Step by Step

One key term before we start: **head_size** is the dimension of each attention head. In our model, we split the 128-dimensional embedding across 4 heads: 128 / 4 = 32 per head. Chapter 6 explains this split. For now, just know that head_size = 32.

Given input `x` of shape `(B, T, C)`:

**Step 1: Project to Q, K, V**

`W_q`, `W_k`, and `W_v` are learnable weight matrices (shape: C x head_size). Think of them as filters: `W_q` multiplies each token's 128-number embedding to extract only the "query-relevant" aspects - out of 128 numbers, which ones matter for asking questions? `W_k` filters for "key" aspects. `W_v` filters for "value" aspects. These filters are learned automatically during training.

```python
q = x @ W_q   # "What am I looking for?" shape: (B, T, head_size)
k = x @ W_k   # "What do I offer?"       shape: (B, T, head_size)
v = x @ W_v   # "My actual content"       shape: (B, T, head_size)
```

**Step 2: Compute attention scores**

We want to score every pair of tokens: "how much should token i pay attention to token j?" To do this using matrix multiply, we need shape (T, head_size) @ (head_size, T) = (T, T), which gives one score for every pair. The transpose flips K's last two dimensions to make this possible.

`k.transpose(-2, -1)` swaps K's last two dimensions, turning shape `(B, T, head_size)` into `(B, head_size, T)`. This lets us compute `(B, T, head_size) @ (B, head_size, T) = (B, T, T)` - a score for every token pair.

```python
scores = q @ k.transpose(-2, -1)   # (B, T, T)
scores = scores / sqrt(head_size)   # scale down to avoid huge values
```

Token `i`'s scores at position `[i, j]` = "how much should token `i` pay attention to token `j`?"

**Step 3: Apply the causal mask**
```python
scores = scores.masked_fill(future_positions, float("-inf"))
```

This sets scores for all future tokens to `-infinity`. Softmax will convert these to 0, so token `i` cannot "see" tokens `j > i`.

Why? If the model can see all future tokens during training, it learns a cheat: just look at position t+1 and copy whatever is there. It gets perfect training accuracy but learns absolutely nothing about language. And during generation (Chapter 14), future tokens literally do not exist yet - the model is supposed to create them. The causal mask forces the model to learn real patterns from context rather than cheating.

**Step 4: Softmax → attention weights**
```python
weights = softmax(scores)   # (B, T, T), each row sums to 1
```

Now `weights[i, j]` = "what fraction of attention does token `i` give to token `j`?"

**Step 5: Weighted sum of values**
```python
output = weights @ v   # (B, T, T) @ (B, T, head_size) = (B, T, head_size)
```

Each token's output = a mix of all values, weighted by how relevant each was.

---

### Why Divide by sqrt(head_size)?

Here is the problem without scaling: a dot product of two 32-dimensional vectors (head_size = 32) can easily reach large numbers - imagine summing 32 multiplications, each potentially 1 or 2, the total could be 30-50 or higher.

Feed those large scores into softmax:
- `softmax([50, 45, 30])` gives roughly `[99%, 1%, 0%]` - the model is overconfident, nearly certain about one token and ignoring the rest.
- The model stops learning because it already thinks it knows the answer with certainty.

After dividing by `sqrt(32) = 5.66`:
- `softmax([8.8, 7.9, 5.3])` gives roughly `[70%, 28%, 2%]` - still prefers one token but spreads attention across others.
- Now the model can keep learning from getting the distribution wrong.

This division is a simple numerical trick to keep the model learning well. It is called "scaled dot-product attention."

---

### The Causal Mask: Visualized

For a sequence of 5 tokens, the mask looks like this:

```
        look at:  0   1   2   3   4
token 0:          OK  -   -   -   -    (can only see itself)
token 1:          OK  OK  -   -   -    (can see 0 and 1)
token 2:          OK  OK  OK  -   -
token 3:          OK  OK  OK  OK  -
token 4:          OK  OK  OK  OK  OK   (can see all previous)
```

`-` means the score is set to `-inf` before softmax, so the weight becomes 0.

Why this specific lower-triangular pattern? It mirrors how you read: when you are at word 0, you have only read word 0. When you are at word 4, you have read words 0 through 4. Token 4 can see tokens 0-4. Token 0 can only see itself.

This lower-triangular mask is what makes attention **causal**, each token can only see its past, not its future.

---

## Code

> **File**: `src/ch05_self_attention.py`
> **Run it**: `python src/ch05_self_attention.py`

This file implements `SingleHeadAttention` as a standalone `nn.Module` with all 5 steps visible in the `forward()` method. Each line has a shape comment.

### Sample output

```
Token 5 attends to tokens 0..5 (future tokens masked):
  token 0: 0.149  ####
  token 1: 0.152  ####
  token 2: 0.146  ####
  token 3: 0.175  #####
  token 4: 0.209  ######
  token 5: 0.168  #####
  token 6: 0.000
  token 7: 0.000
  token 8: 0.000
  token 9: 0.000
```

Notice: tokens 6-9 have weight 0.000, the causal mask is working.

---

## Key Takeaways

- Self-attention lets each token look at all other tokens and decide what's relevant.
- Q = "what I'm looking for", K = "what I offer", V = "my content".
- Scores = Q x K^T, scaled by sqrt(head_size), then softmax gives weights.
- The causal mask prevents tokens from seeing the future.
- Output = weighted sum of V vectors.

---

*Next up: [Chapter 7, Multi-Head Attention](ch07-multi-head-attention.md)*
