# Chapter 15: Greedy Decoding and Sampling

> **Code file**: `src/ch14_generate_greedy.py`
> **Run it**: `python src/ch14_generate_greedy.py`
> *(Requires a trained checkpoint from `ch12_train.py`)*

---

## Theory

### From Logits to Text

Our model produces **logits** - 65 raw scores, one per character. (Recall from Chapter 9: logits are the unnormalized scores from the LM head, before any softmax.) To pick the next character, we need a strategy.

We have two basic strategies:

---

### Strategy 1: Greedy Decoding

Always pick the character with the **highest score**.

```python
next_token = logits.argmax()   # argmax = "index of the maximum value"
```

**Pros**: Simple, fast, deterministic (always produces the same output from the same prompt).

**Cons**: Repetitive. Once the model outputs a repetitive pattern, that pattern becomes the context for the next prediction. The same context creates nearly identical logits, which again picks the same token.

Here is a concrete example of how the loop forms. Suppose the model learned:
- Given `[king, is,  ]` (king + is + space), the next character is `t`
- Given `[is,  , t]` (is + space + t), the next character is `h`
- Given `[ , t, h]` (space + t + h), the next character is `e`
- ... and eventually circles back to producing ` king is the king is the...`

The model is not broken - it found a short repeating pattern in its training data and locked into it. This is the failure mode of greedy decoding.

Greedy decoding is like a student who always circles "C" on every multiple choice question - technically safe, usually boring, occasionally loops forever.

Example of greedy going wrong:
```
Input: "The king of England"
Greedy: "The king of England the the the the the the the..."
```

The model got stuck in a loop. This is called **repetition** and is a known failure mode of greedy decoding.

---

### Strategy 2: Sampling

Instead of always picking the top token, we **sample randomly** from the probability distribution.

```python
probs   = softmax(logits)
next_token = torch.multinomial(probs, num_samples=1)
```

`torch.multinomial` picks a token randomly, but tokens with higher probability are more likely to be chosen. It's like a weighted lottery:
- Character "e" has 40% probability → wins the lottery 40% of the time
- Character "x" has 0.1% probability → wins very rarely, but occasionally

**Pros**: Varied output, avoids repetition, more creative-sounding text.

**Cons**: Unpredictable. Sometimes picks weird tokens. Different every run (unless you set a random seed).

---

### Which Is Better?

Neither is universally better, it depends on what you want:

| Situation | Better Strategy |
|-----------|----------------|
| You want consistent, predictable output | Greedy |
| You want creative, varied output | Sampling |
| Filling in a specific factual answer | Greedy |
| Writing a poem or story | Sampling |

In practice, most real applications use sampling with controls (temperature and top-k, see Chapter 15).

---

### The Generation Loop

Both strategies use the same loop:

```python
context = prompt_token_ids    # starting context

for _ in range(max_new_tokens):
    logits   = model(context)[:, -1, :]   # get logits at last position
    next_id  = pick_next_token(logits)    # greedy or sampling
    context  = append(context, next_id)   # grow the sequence
```

The key step: we always use the **logits at the last position** (`[:, -1, :]`). The model output shape is `(batch, sequence_length, vocab_size)`. We index `[:, -1, :]` to take the last sequence position from every batch item - that is where the model has attended to everything before it and is predicting what comes next. Positions earlier in the sequence predict what comes after *them*, not after the whole prompt.

We **crop the context** to `block_size` (128) tokens before each forward pass. Here is why: during training, the model only ever saw sequences of exactly 128 tokens. Position embeddings were learned for positions 0 through 127. If you feed the model 200 tokens, it encounters position 150, which has no embedding - the model has never seen that position during training, so the output is undefined. By always keeping the last 128 tokens (a sliding window), we stay within the model's trained range.

---

## Code

> **File**: `src/ch14_generate_greedy.py`
> **Run it**: `python src/ch14_generate_greedy.py`

This file implements both `generate_greedy()` and `generate_sample()` and runs them with the same prompt for comparison.

### If you haven't trained yet

Run `python src/ch12_train.py` first (20-30 minutes). Then come back.

With an untrained model, both strategies produce gibberish. With a trained model, you'll see a clear difference in style.

---

## Key Takeaways

- Greedy decoding: always pick the highest-probability token. Simple but repetitive.
- Sampling: pick randomly weighted by probability. More varied but unpredictable.
- Both use the same loop: predict → pick → append → repeat.
- We always use the logits at the **last position** to predict the **next** token.

---

*Next up: [Chapter 16, Temperature and Top-k](ch16-temperature-and-topk.md)*
