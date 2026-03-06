# Chapter 9: The Transformer Block

> **Code file**: `src/ch08_transformer_block.py`
> **Run it**: `python src/ch08_transformer_block.py`

---

## Theory

### Putting the Pieces Together

We now have all the ingredients:
- Multi-head attention (Chapter 6), tokens talk to each other
- Feed-forward layer (Chapter 7), tokens think for themselves
- Layer normalization (Chapter 7), keeps training stable

One **transformer block** combines all of these into a single reusable unit. We'll stack multiple copies of this block to build the full model.

---

### The Transformer Block: Forward Pass

The full forward pass of one block is just two lines of code:

```python
x = x + attention(LayerNorm(x))    # "listen to others, then add it to what I know"
x = x + feedforward(LayerNorm(x))  # "think about it, then update what I know"
```

That's it. Let's unpack each piece.

---

### Trick 1: Residual Connections

Notice the `x + ...` pattern. Instead of:
```python
x = attention(x)   # REPLACE x with attention's output
```

We do:
```python
x = x + attention(x)   # ADD attention's output to x
```

This is called a **residual connection** (or skip connection).

Why does it matter? Imagine you're in a game of telephone. Each person translates the message and passes it on. After 10 rounds, the original message is usually unrecognizable.

Residual connections are like also passing the original message separately alongside the game of telephone. Even if the translated message gets mangled, the original is still there.

In practice: gradients flow more easily during training. Deep networks without residual connections are very hard to train; with them, you can stack dozens of layers.

Here is why it matters for training: during training, "feedback" signals travel backward through all layers to adjust weights. Without shortcuts, this feedback must pass through every transformation layer in sequence - like a chain of telephone. If each layer distorts the signal slightly, after 4+ layers the signal either grows too large (exploding gradients) or shrinks to nearly nothing (vanishing gradients). Either way, the model can't learn effectively from deep layers.

With residual connections, the `+x` shortcut provides a direct highway for the feedback signal to bypass each block entirely. Even if a block's transformations get messy, the original signal travels home cleanly. This is why deep networks (many layers) became practical only after residual connections were invented.

---

### Trick 2: Pre-Layer Norm

We apply LayerNorm *before* each sub-layer (the "pre-norm" style):

```
x  ->  LayerNorm  ->  Attention  ->  + x  ->  LayerNorm  ->  FFN  ->  + x
```

This is the modern convention (used in GPT-2 and all later models). It stabilizes training, especially in the early stages.

---

### Stacking Blocks

One transformer block isn't enough. We stack `n_layers = 4` of them in sequence:

```python
# Create a list of 4 transformer blocks, then chain them so data flows 1 -> 2 -> 3 -> 4.
# nn.Sequential chains modules in order.
# The * unpacks the list of 4 blocks into 4 separate arguments.
blocks = nn.Sequential(*[TransformerBlock() for _ in range(4)])
```

Each block refines the token representations further. Think of it like reading a book multiple times with a different question each pass. First pass: 'what is happening?' (who are the characters?). Second pass: 'why is it happening?' (motivations). Third pass: 'what does it mean?' (themes). Earlier transformer blocks tend to capture simpler patterns (which characters are nearby?), later blocks capture more abstract ones (what does this scene mean?).

In large, well-studied models, researchers have confirmed this pattern: early layers encode syntax and local context, later layers encode semantics and long-range relationships. Our small model is too tiny to show this clearly, but the mechanism is identical.

This is why deeper models (more layers) generally perform better.

---

### Parameter Count

Each block has ~197,888 parameters. With 4 blocks: ~791,552 parameters, about 96% of our entire model. The attention mechanism and feed-forward layer are where most of the "knowledge" lives.

---

## Code

> **File**: `src/ch08_transformer_block.py`
> **Run it**: `python src/ch08_transformer_block.py`

This file:
1. Implements `TransformerBlock` using the two-line forward pass
2. Shows the parameter breakdown
3. Demonstrates the residual connection
4. Stacks 4 blocks and confirms shapes are unchanged

### Key thing to notice

```
Input  shape: torch.Size([2, 10, 128])
Output shape: torch.Size([2, 10, 128])   (same as input)
```

The transformer block is **shape-preserving**. Input and output have identical shapes. This is what allows us to stack them: the output of block 1 feeds directly into block 2.

---

## Key Takeaways

- A transformer block = LayerNorm + Attention + Residual + LayerNorm + FFN + Residual.
- Residual connections add the input to the output: `x = x + sublayer(x)`.
- Pre-norm: LayerNorm is applied before each sub-layer.
- Blocks are shape-preserving: output shape = input shape.
- We stack 4 blocks; each refines the token representations further.

---

*Next up: [Chapter 10, The Full GPT Architecture](ch10-full-gpt-architecture.md)*
