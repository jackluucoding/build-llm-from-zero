# Chapter 8: Feed-Forward Layers and Layer Normalization

> **Code file**: `src/ch07_feedforward.py`
> **Run it**: `python src/ch07_feedforward.py`

---

## Theory

### After Attention, We Need to "Think"

Attention lets each token gather information from its neighbors: *"Who said what, and how does it relate to me?"*

But gathering information is only half the job. The token still needs to **process** that information and decide what to do with it.

The **feed-forward layer** does that processing. It's applied independently to each token's representation and gives the model capacity to transform what it learned through attention.

Analogy: Attention is listening to your classmates explain their solutions. Feed-forward is going home and thinking it over yourself.

---

### Feed-Forward Network Architecture

The feed-forward layer is a small two-layer neural network:

```
x → Linear(C → 4C) → GELU → Linear(4C → C)
```

In our model (C = 128):
```
x → Linear(128 → 512) → GELU → Linear(512 → 128)
```

**Why 4x?** The 4x expansion gives the model a wide workspace. Think of it like a drafting table: too narrow and you can't spread out your work. The network expands to think, then compresses back to an answer. The original Transformer paper used this ratio, and it has proven to be the sweet spot across many model sizes ever since.

**Important**: this is applied to *each token independently*. Token 5 goes through its own copy of the feed-forward network. Token 6 goes through an identical copy. They don't interact here, interaction only happens in attention.

---

### GELU Activation

Between the two linear layers, we apply **GELU** (Gaussian Error Linear Unit).

Think of it as ReLU with a smoother edge. ReLU is a hard on/off switch:
- Negative input? Output 0. Done.
- Positive input? Output as-is.

GELU is a gradual dimmer:
- Strongly negative input? Output nearly 0.
- Slightly negative input? Output reduced, but not to zero.
- Positive input? Output passes through, almost unchanged.

In practice, you just need to know:
- Strongly negative values: nearly 0
- Values near zero: gently reduced
- Positive values: pass through

Why smooth matters: during training, we use calculus to figure out how much to adjust each weight. When the activation function has a sharp corner (like ReLU's sudden kink at zero), the calculus produces sudden jumps that can destabilize training. GELU's smooth curve makes the math nicer and training more stable. GPT-2 and most modern transformers use GELU instead of ReLU.

---

### Layer Normalization

One more ingredient: **Layer Normalization** (LayerNorm).

During training, numbers flowing through many stacked layers can grow very large or shrink very small - like compound interest, small multiplications add up over 4 layers. Very large numbers cause the weight-update math to blow up (exploding gradients). Near-zero numbers cause the update signals to disappear (vanishing gradients). Both problems make training unstable or impossible.

LayerNorm fixes this by "re-centering" and "re-scaling" the values at each token position:

```
LayerNorm(x) = (x - mean) / std × gamma + beta
```

Where `gamma` and `beta` are learned parameters that let the model scale and shift after normalization.

**Result**: After LayerNorm, the values at each position have mean ≈ 0 and standard deviation ≈ 1.

Think of it like: every time a runner finishes a lap, you reset their stopwatch to 0. It doesn't change who's winning, it just keeps the numbers manageable.

By keeping values near mean=0 and std=1, gradients stay in a stable range as they flow backward through the layers. Without this, signals can either amplify exponentially (exploding gradients) or shrink toward zero (vanishing gradients) as they travel through 4+ stacked blocks.

---

### Where Does LayerNorm Go?

There are two conventions:

**Post-norm** (original Transformer, 2017): Apply LayerNorm *after* attention and feed-forward.

**Pre-norm** (GPT-2 and later): Apply LayerNorm *before* each sub-layer.

We use **pre-norm** because it's more stable during training. Pre-norm: normalize first, then do the hard work - like stretching before exercise. Post-norm: do the hard work, then normalize. Pre-norm is more stable because attention and feed-forward receive well-behaved, normalized inputs rather than potentially messy ones.

In Chapter 8, you'll see:

```python
x = x + attention(LayerNorm(x))    # normalize BEFORE attention
x = x + feedforward(LayerNorm(x)) # normalize BEFORE feedforward
```

---

## Code

> **File**: `src/ch07_feedforward.py`
> **Run it**: `python src/ch07_feedforward.py`

This file:
1. Implements `FeedForward` as a standalone `nn.Module`
2. Demonstrates GELU vs ReLU comparison
3. Demonstrates what LayerNorm does to values

### Output to notice

```
Before LayerNorm: mean=4.63, std=9.57
After  LayerNorm: mean=0.0000, std=1.0039
```

The values before normalization were all over the place. After: tightly controlled at mean=0, std≈1. This is what we want during training.

---

## Key Takeaways

- Feed-forward = two-layer MLP applied independently to each token.
- Architecture: `Linear(C → 4C) → GELU → Linear(4C → C)`.
- GELU is a smooth version of ReLU, better for training.
- LayerNorm re-centers and re-scales values to keep training stable.
- We use pre-norm: LayerNorm is applied *before* each sub-layer.

---

*Next up: [Chapter 9, The Transformer Block](../section-3-the-transformer/ch09-transformer-block.md)*
