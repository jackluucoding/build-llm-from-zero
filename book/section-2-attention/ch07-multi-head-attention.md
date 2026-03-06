# Chapter 7: Multi-Head Attention

> **Code file**: `src/ch06_multihead_attention.py`
> **Run it**: `python src/ch06_multihead_attention.py`

---

## Theory

### One Reader Isn't Enough

In Chapter 5, we built one attention head, one "reader" that scans the text and decides what's important.

But when you read a sentence, you're tracking multiple things at once:
- *Who* is the subject?
- *What action* is happening?
- *When* is this taking place?
- *What's the tone*, serious? sarcastic?

A single attention head can only "look" for one kind of pattern at a time. **Multi-head attention** fixes this by running N heads in parallel, each potentially learning to track a different aspect of the text.

Think of it as having a team of readers, each highlighting different things, then combining all their notes.

---

### How It Works

Multi-head attention is actually very simple conceptually:

1. Run N independent `SingleHeadAttention` modules on the same input `x`.
2. Each head produces an output of shape `(B, T, head_size)`.
3. Concatenate all outputs along the last dimension: `(B, T, N * head_size)`.
4. Apply a final linear projection to get back to shape `(B, T, C)`.

```python
# For each of the 4 heads, run it on input x and collect all 4 outputs into a list.
# This runs all heads on the same data, giving 4 different "perspectives".
head_outputs = [head(x) for head in self.heads]      # N x (B, T, head_size)
concatenated = torch.cat(head_outputs, dim=-1)        # (B, T, N * head_size)
output       = self.projection(concatenated)          # (B, T, C)
```

In our model: N=4 heads, head_size=32, so `4 × 32 = 128 = C`. The output is the same shape as the input, perfect.

---

### Do the Heads Really Learn Different Things?

Yes - by architectural diversity. Each head has its own independent Q, K, V weight matrices, so they start from different random initializations and naturally learn different patterns.

Here is the intuition: all 4 heads see the same input text, but each starts with different random weights and makes different guesses during early training. As training adjusts weights to reduce loss, each head finds a different "angle" that helps. It is like a group study session where everyone reads the same chapter but each person ends up focusing on different details - one tracks the main argument, another notices examples, a third follows the logic structure.

In well-trained large models, researchers have found that different attention heads specialize:
- Some heads track subject-verb agreement across long distances.
- Some focus on local context (neighboring words).
- Some track coreference ("it" to "the trophy").

Our small model won't show this clearly. With only 4 heads sharing 128 dimensions, there is not much room for specialization. Larger models with 96 heads and 12,288 embedding dimensions show this dramatically - researchers have identified heads that reliably track specific grammatical relationships. But the underlying mechanism is identical to what we built.

---

### The Output Projection

After concatenating the heads, we apply one final linear layer:

```python
self.proj = nn.Linear(n_embd, n_embd)
```

Why? Two reasons:
1. It combines the heads intelligently. After concatenation, we have 4 x 32 = 128 numbers from 4 different heads. But the heads might have redundant or conflicting information. The projection is a (128 x 128) weight matrix that learns: "for this kind of token, head 1's insight is most important, head 3 is redundant, downweight it." It blends the heads based on what actually helps.
2. It gives the model extra capacity to transform the combined multi-head representation before passing it on.

---

## Code

> **File**: `src/ch06_multihead_attention.py`
> **Run it**: `python src/ch06_multihead_attention.py`

This file imports `SingleHeadAttention` from Chapter 5 and wraps N of them in a `MultiHeadAttention` module.

### Key output to notice

```
Multi-head output has 4x more channels  -  it sees 4 perspectives at once.
```

The output shape is still `(B, T, C)`, same as the input. Multi-head attention doesn't change the shape, it just enriches the content.

### Parameter count

```
MultiHeadAttention total parameters: 65,664
  From 4 heads: 49,152
  From output proj : 16,512
```

About 50K parameters just for the attention layer, and that's per transformer block!

---

## Key Takeaways

- Multi-head attention = N single-head attention modules running in parallel.
- Each head can specialize in detecting different patterns.
- Outputs are concatenated, then projected back to the original shape.
- Output shape: `(B, T, C)`, same as input.
- 4 heads × 32 head_size = 128 = n_embd. It all fits together.

---

*Next up: [Chapter 8, Feed-Forward Layers and Layer Norm](ch08-feedforward-and-norms.md)*
