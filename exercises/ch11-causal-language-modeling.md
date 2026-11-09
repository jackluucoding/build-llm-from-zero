# Chapter 11: Causal Language Modeling: Exercises

Read the chapter online: https://jackluu.io/book/section-3-the-transformer/ch11-causal-language-modeling/

## Try It

To see how confidence affects the loss, we can simulate different prediction scenarios manually.

```python
"""Calculate cross-entropy loss for different confidence levels."""
import torch
import torch.nn.functional as F

# Two possible words: 'A' (index 0) or 'B' (index 1)
target = torch.tensor([0]) # The correct answer is 'A'

print("Scenario 1: Guessing blindly (50% / 50%)")
logits_blind = torch.tensor([[0.0, 0.0]])
loss_blind = F.cross_entropy(logits_blind, target)
print(f"Loss: {loss_blind.item():.4f}")

print("\nScenario 2: Confident and right (88% for 'A')")
logits_right = torch.tensor([[2.0, 0.0]])
loss_right = F.cross_entropy(logits_right, target)
print(f"Loss: {loss_right.item():.4f}")

print("\nScenario 3: Confident and wrong (88% for 'B')")
logits_wrong = torch.tensor([[0.0, 2.0]])
loss_wrong = F.cross_entropy(logits_wrong, target)
print(f"Loss: {loss_wrong.item():.4f}")
```

Lines 14 and 15 calculate the loss when the model is confident and correct, which gives a much lower loss.

```console
$ python src/examples/ch11_loss_demo.py
Scenario 1: Guessing blindly (50% / 50%)
Loss: 0.6931

Scenario 2: Confident and right (88% for 'A')
Loss: 0.1269

Scenario 3: Confident and wrong (88% for 'B')
Loss: 2.1269

```

Open `src/examples/ch11_loss_demo.py` and change `logits_right` to `[[5.0, 0.0]]` to make the model even more confident. Run the script and see how close to zero the loss gets.

**In Business**
Let's say you want to teach your assistant to generate content that conforms to your organization's house style. Causal language modeling allows you to do this by inputting target sequences, which would be all the previous emails or documents that your company has created. It then learns what words tend to follow each other in that environment. With the loss, you can see how close it is to your company's history.

**Watch Out**
Remember that `F.cross_entropy()` wants logits to already be flattened out to one row of scores per token. `F.cross_entropy()` wants a 2D table of size `[Total Tokens, VOCAB_SIZE]`, not the model's 3D stack of tables `[BATCH, TIME, VOCAB_SIZE]`. That's what `logits.view(B * T_len, config.vocab_size)` does in that code. This line is easy to forget, which makes it a common bug.

## Check Your Understanding

1. If your sequence is "DATA", what is the input sequence `x` and the target sequence `y`?
2. Why do we need softmax before we can interpret the model's output as percentages?
3. If a model is perfectly confident and perfectly correct, what should its cross-entropy loss be?

## Assignment

**Assignment 11: One number, two lists**

**About 15 minutes. You change one number and write no new code.** Work in this chapter's Colab notebook or on your own laptop.

1. **Run it.** Run `src/ch10_causal_lm.py` as it is. Copy the first three lines of numbers it prints: `Sequence`, `Input x` and `Target y`.
2. **Change one number.** Find the line `example_ids = torch.tensor([20, 17, 30, 30, 33, 1, 35, 53, 56, 30])`. These ten numbers are the token IDs of one short sequence. Change the `1` in the middle to `50` and run the script again.
3. **Compare.** Your new number shows up in `Input x` and in `Target y`. Is it in the same place in both lists? Which number of `Input x` sits directly above your new number in `Target y`? Which end of the sequence is missing from `Input x`, and which end is missing from `Target y`?
4. **Explain it.** In two or three sentences a manager could follow, say how a single piece of text supplies both the questions and the correct answers for training.

**Hand in:** both sets of three lines, your answers to step 3, and your sentences.
