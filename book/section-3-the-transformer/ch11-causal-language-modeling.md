# Chapter 11: Causal Language Modeling

> **Code file**: `src/ch10_causal_lm.py`
> **Run it**: `python src/ch10_causal_lm.py`

---

## Theory

### The Training Trick: One Forward Pass = Many Examples

How do we train a model to predict the next token?

Naively, you might think: take one token, predict the next, check if you're right, update the weights. Then repeat for the next token.

That's 1 million training examples done one at a time. Slow.

The brilliant insight: **a single forward pass on a sequence of length T gives us T training examples at once!**

Here's why. Given the sequence `"HELLO"`:

| Input context | Correct next token |
|--------------|-------------------|
| `H` | `E` |
| `HE` | `L` |
| `HEL` | `L` |
| `HELL` | `O` |

The model processes the whole sequence at once, and at each position, it predicts the next token. We get 4 training examples from one 4-token sequence.

The causal mask (from Chapter 5) ensures position 2 (`HEL`) can't "cheat" by seeing position 3 (`O`). Each position only sees what came before.

---

### Input and Target: The Shifted Pair

In code, this is implemented by using the *same sequence* twice, shifted by one:

```python
x = sequence[:-1]   # all but the last token  → "HELL"
y = sequence[1:]    # all but the first token → "ELLO"
```

At each position `i`, `x[i]` is the input and `y[i]` is the correct next token.

---

### The Loss Function: Cross-Entropy

Once we have predictions (logits) and correct answers (targets), we measure how wrong we are using **cross-entropy loss**.

Intuitively: for each position, the model predicts a probability distribution over all 65 characters. Cross-entropy measures how much probability the model assigned to the *correct* character.

- If the model predicts 90% probability for the correct character: low loss (good!)
- If the model predicts 1% probability for the correct character: high loss (bad!)

For a random model (no training), the expected loss is `log(65) ≈ 4.17`. Here is where that number comes from: with 65 characters and no knowledge, the model assigns equal probability to each (1/65 ≈ 1.5%). Cross-entropy loss for a uniform distribution over 65 choices equals log(65) ≈ 4.17. Think of it as the "complete ignorance" baseline - this is the worst possible score a model can get while still trying. After training to 3000 steps, we expect the loss to drop to around `1.3-1.5`. At that point the model assigns roughly 4-5x higher probability to the correct character than a random guesser would. A much better model would hit `1.0` or below. For reference: a human reading Shakespeare might score around 0.8-1.0 on this exact task.

---

### Autoregressive Generation

Once trained, we use the model to *generate* new text:

1. Start with a prompt: `"ROMEO:\n"` (encoded as token IDs)
2. Feed the prompt to the model → get logits at the last position
3. Convert logits to probabilities (softmax)
4. Sample one token from the probability distribution
5. Append that token to the sequence
6. Repeat from step 2 until you have enough text

Each new token is added to the context, which influences what comes next. This is why it's called **autoregressive**: "auto" means self, and "regressive" means going back to previous data. So autoregressive means the model uses its own previous outputs as inputs for the next step - like writing a sentence one word at a time, where each word you choose influences what word you write next.

---

### "No Cheating": Why the Causal Mask Matters

Without the causal mask, at training position `t` the model could peek at the input token at position `t+1` and trivially copy it to the output. Perfect training accuracy, zero actual language understanding - the model learned to cheat, not to predict.

The mask prevents this technically: inside the attention calculation (Chapter 5), it sets attention weights to zero for all future positions. Position 2 literally cannot receive any information from positions 3, 4, 5, etc. - they are blocked, not just discouraged. The model has no choice but to predict from context.

The mask forces the model to actually learn: "Given everything I've seen so far, what's the most likely next character?"

This is the same constraint you'd impose on a student during an exam: cover up the answers and actually *think*.

---

## Code

> **File**: `src/ch10_causal_lm.py`
> **Run it**: `python src/ch10_causal_lm.py`

This file:
1. Shows how input/target pairs are constructed
2. Computes cross-entropy loss on a random (untrained) model
3. Implements the `generate()` function
4. Demonstrates generation on an untrained model (random output, but correct mechanics)

### Important output

```
Loss (random model): 4.3253
Expected loss for random: 4.1744
```

These are close, the model is essentially guessing randomly, as expected. After training, this loss will drop significantly.

---

## Key Takeaways

- One forward pass on a T-length sequence = T training examples (huge efficiency win!).
- Input `x` and target `y` are the same sequence, shifted by one position.
- Cross-entropy loss measures how wrong the predictions are. Lower = better.
- Random model loss ≈ log(vocab_size) = log(65) ≈ 4.17.
- Generation = repeatedly predicting and appending the next token.

---

*Next up: [Chapter 12, Dataset and DataLoader](../section-4-training/ch12-dataset-and-dataloader.md)*
