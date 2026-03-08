# Chapter 16: Temperature and Top-k Sampling

> **Code file**: `src/ch15_generate_sampling.py`
> **Run it**: `python src/ch15_generate_sampling.py`
> *(Requires a trained checkpoint from `ch12_train.py`)*

---

## Theory

### Plain Sampling Has Problems Too

We saw in Chapter 14 that plain sampling (random choice weighted by probability) gives varied output. But sometimes it picks *really* bad tokens, like a comma in the middle of a word, or a rarely-seen character that breaks the grammar.

The problem: even characters with very low probability (say, 0.1%) occasionally get picked. That's one in a thousand tokens, and with 200 tokens generated, you might expect several weird ones.

We need controls.

---

### Control 1: Temperature

**Temperature** adjusts how "peaked" or "flat" the probability distribution is before sampling.

The implementation is simple: divide the logits by a temperature value `T` before applying softmax.

```python
logits = logits / temperature   # adjust before softmax
probs  = softmax(logits)
```

- **T < 1.0** (e.g., 0.5): Dividing by a small number makes logits *bigger* in magnitude. Quick math: dividing 6 by 0.5 gives 12 (dividing by a fraction multiplies). So logits [1, 2, 3] divided by 0.5 give [2, 4, 6] - the gap between highest and lowest doubled from 2 to 4. After softmax, the top token becomes overwhelmingly more likely. Output is focused and coherent, but tends toward repetition.

- **T = 1.0**: Dividing by 1 changes nothing. Normal sampling - the model's raw learned distribution.

- **T > 1.0** (e.g., 2.0): Dividing by a large number makes logits *smaller*. Logits [1, 2, 3] divided by 2.0 give [0.5, 1.0, 1.5] - the gap shrank from 2 to 1. After softmax, all tokens have more similar probabilities. Output is more random and creative, sometimes nonsensical.

Think of temperature as the "creativity dial":
- Low temperature = accountant mode (safe, predictable)
- High temperature = jazz musician mode (creative, unpredictable)

A sweet spot is usually between 0.7 and 1.0. To give you a feel for the range:
- T=0.5: Very focused and coherent, but tends toward repetition
- T=0.8: Good balance - our recommended default
- T=1.0: The model's raw learned distribution, some surprising choices
- T=1.5+: Gets weird fast - expect sudden topic changes and odd vocabulary

---

### Control 2: Top-k Sampling

**Top-k** limits which tokens can even be considered. Before sampling, we keep only the top-k highest-scoring tokens and set all others to negative infinity (which becomes 0 after softmax).

```python
k = 40
top_k_values = logits.topk(k).values
threshold    = top_k_values[:, -1, None]   # k-th largest value
logits       = logits.masked_fill(logits < threshold, float("-inf"))  # eliminated tokens get prob 0
probs        = softmax(logits)
next_token   = multinomial(probs, 1)
```

`masked_fill` is a PyTorch operation that says: "wherever this condition is True, replace the value with a specified number." Here, the condition is `logits < threshold` (score lower than the k-th best), and the replacement is `-inf` (negative infinity). Why `-inf`? Because softmax computes `e^x` for each value. `e^(-infinity) = 0`. So any token set to `-inf` gets exactly 0 probability after softmax - completely eliminated from sampling.

With `top_k = 40`: only the 40 most likely characters are in the running. We keep the top 40 and zero out the remaining 25 (out of 65).

This prevents the model from ever picking a truly bizarre token. Even if the sampling is random within the top 40, at least we know all 40 are "reasonable" choices given the context.

**Common values**: top_k = 40 or 50 works well. top_k = 1 = greedy decoding.

---

### Combining Temperature and Top-k

Most real applications use both together:

```
1. Compute logits
2. Apply top-k (keep the top-k tokens, zero out the rest)
3. Apply temperature (adjust the spread)
4. Apply softmax (convert to probabilities)
5. Sample
```

Recommended settings: `temperature=0.8, top_k=40`

This combination gives you:
- Creative and varied output (from sampling + temperature)
- With a quality floor (from top-k, no bizarre picks)

---

## Code

> **File**: `src/ch15_generate_sampling.py`
> **Run it**: `python src/ch15_generate_sampling.py`

This file generates text with 5 different configurations for direct comparison:
1. `temp=0.5, top_k=40`, focused/conservative
2. `temp=0.8, top_k=40`, balanced (recommended default)
3. `temp=1.0, top_k=40`, the model's raw learned distribution
4. `temp=1.5, top_k=40`, creative/chaotic
5. `temp=1.0, no top_k`, unrestricted sampling

After running Chapter 12's training, the differences become very obvious.

---

## Key Takeaways

- Temperature < 1: more focused output. Temperature > 1: more random output.
- Implementation: just divide logits by temperature before softmax.
- Top-k: only keep the k highest-scoring tokens as candidates.
- Recommended sweet spot: `temperature=0.8, top_k=40`.
- Both controls are applied *before* softmax and sampling.

---

*Next up: [Chapter 17, Putting It All Together](ch17-putting-it-all-together.md)*
