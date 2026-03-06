# Chapter 13: The Training Loop

> **Code file**: `src/ch12_train.py`
> **Run it**: `python src/ch12_train.py`
> **Expected time**: ~20-30 minutes on CPU

---

## Theory

### The Most Important Four Lines in Machine Learning

The training loop reduces to four operations, repeated thousands of times:

```python
logits = model(x)              # 1. Forward pass: make predictions
loss   = cross_entropy(logits, y)  # 2. Compute loss: how wrong are we?
loss.backward()                    # 3. Backward pass: who's responsible?
optimizer.step()                   # 4. Update: fix the responsible parameters
```

That's it. Everything else is just bookkeeping.

---

### Step 1: Forward Pass

We feed a batch of 32 text windows through the model. For each of the 32 sequences, at each of the 128 positions, the model outputs 65 scores (one per character). The overall logits shape is `(32, 128, 65)`.

---

### Step 2: Compute Loss

Cross-entropy loss compares our predictions to the correct answers. Lower loss = better predictions.

On the first step, loss ≈ 4.17 (random guessing). After 3000 steps, loss drops to around 1.3-1.5. The model went from "no idea" to "pretty decent at Shakespeare."

---

### Step 3: Backward Pass (Backpropagation)

`loss.backward()` computes the **gradient** of the loss with respect to every parameter in the model.

Here is how it works: during the forward pass, PyTorch secretly records every mathematical operation it performs (multiply, add, softmax, etc.). When you call `loss.backward()`, PyTorch traces backward through all those recorded steps and figures out: "If I had changed this weight by a tiny amount, how would the loss have changed?"

A gradient answers exactly that question for every single weight in the model:
- Positive gradient: increasing this weight makes loss go up, so we should decrease it.
- Negative gradient: increasing this weight makes loss go down, so we should increase it.
- Big gradient: this weight has a strong effect on loss, adjust it more.
- Small gradient: this weight barely matters right now.

Think of it like: you're lost in a hilly landscape (loss landscape) and you want to find the lowest valley. The gradient tells you which direction is downhill from where you're standing.

**Important**: Before each backward pass, we call `optimizer.zero_grad()`. Here is why: PyTorch does not replace gradient values when you call `backward()` - it *adds* the new gradients on top of whatever was already there. So if you skip `zero_grad()`, you accumulate gradients from batch 1, batch 2, batch 3 all added together, which is nonsense. You want only fresh gradients from this batch. Zero it first, then compute fresh.

Note: `optimizer.zero_grad()` must come *before* `loss.backward()`, not after.

---

### Step 4: Optimizer Step (Adam)

`optimizer.step()` uses the gradients to update all parameters:

```
new_value = old_value - learning_rate × gradient
```

This is called **gradient descent**. The learning rate (`3e-4 = 0.0003`) controls how big each step is.

We use the **Adam optimizer**, which is smarter than basic gradient descent.

Basic gradient descent takes the same step size for every weight, every time. Adam is smarter: it keeps a memory of the last many gradient updates for each weight and uses that history to choose a better step size. Weights that keep getting large gradients get smaller steps (they are in steep terrain, be careful). Weights that get small or inconsistent gradients get bigger steps (they need more nudging). It is like a smart hiker who adjusts stride length based on how the terrain has been - shorter strides on steep cliffs, longer strides on flat ground.

Adam usually converges faster and more reliably than plain gradient descent.

---

### The Validation Loss

Every 300 steps, we compute the loss on the *validation set* (text the model hasn't trained on). This is our honest measure of how well the model has learned.

If training loss drops but validation loss doesn't: the model is **overfitting** (memorizing, not learning). For example: training loss 1.3, validation loss 2.5 means the model has memorized specific sequences from the training text but cannot apply what it learned to new text it has never seen. It learned the answers, not the concept.

If both drop together: the model is genuinely learning patterns that work on new text too.

---

### What to Expect

```
step     1 | train loss: 4.2827 | val loss: 4.2xxx
step   300 | train loss: 2.xxxx | val loss: 2.xxxx
step   600 | train loss: 2.xxxx | val loss: 2.xxxx
...
step  3000 | train loss: 1.3xxx | val loss: 1.5xxx
```

Why does loss start near 4.17? On step 0, the model has learned nothing - it is essentially guessing at random among 65 possible characters. With equal probability for all 65 characters, the cross-entropy loss formula gives log(65) = 4.17. Think of it as the "complete ignorance" baseline.

Loss drops fastest in the early steps because there is a lot of low-hanging fruit - the model quickly learns obvious patterns like 'spaces are common' and 'e follows many letters'. As training progresses, only harder patterns remain, and each step yields smaller gains. The model approaches the limits of what 825K parameters trained on 1MB of text can learn.

The validation loss being slightly higher than training loss is normal and expected. The model has seen training examples many times and adapted to them specifically. Validation examples are fresh - like the difference between a familiar practice test and a surprise quiz on the same material.

---

## Code

> **File**: `src/ch12_train.py`
> **Run it**: `python src/ch12_train.py`

This file is the most important in the project. Read through it carefully, every line has a comment explaining its purpose.

Key things to notice:
- `optimizer.zero_grad()` comes *before* `loss.backward()`.
- We call `model.eval()` before validation (disables dropout) and `model.train()` after.
- A checkpoint is saved at the end.

### After training

Once training completes, you'll have:
- A checkpoint file at `checkpoints/model.pt`
- Printed loss values showing the model improving
- A model ready for generation in Chapters 14-16!

---

## Key Takeaways

- Training loop = forward pass → loss → backward pass → optimizer step. Repeat.
- Gradient = "which direction should we adjust each parameter?"
- Adam optimizer adapts the step size for each parameter automatically.
- Validation loss = honest measure of generalization (not memorization).
- ~20-30 min on a CPU-only laptop for 3000 steps.

---

*Next up: [Chapter 14, Checkpointing](ch14-checkpointing.md)*
