# Chapter 8: Feed-Forward and Norms: Exercises

Read the chapter online: https://jackluu.io/book/section-2-attention/ch08-feedforward-and-norms/

## Try It

Open `src/ch07_feedforward.py` and modify the LayerNorm input to have a very large spread: `x_single = torch.randn(config.n_embd) * 1000 + 500`. Execute the script again. Note that LayerNorm still normalizes it back to mean 0 with standard deviation of 1.

**In Business**
Suppose you were building some kind of ML program to generate emails for you at work, written in your company's house style. If your model is poor, it could output garbage. You can think of LayerNorm as ensuring values don't explode before heading to the next neuron layer.

## Check Your Understanding

1. Why does the feed-forward network expand its input size by 4?
2. What happens to a strongly negative number when it passes through GELU?
3. What is the average value of a token's numbers immediately after LayerNorm?

## Assignment

**Assignment 8: Dimmer or switch**

**About 15 minutes. You change one number and write no new code.** Work in this chapter's Colab notebook or on your own laptop.

1. **Run it.** Run `src/ch07_feedforward.py` as it is. Copy the three lines it prints under `GELU activation`: the `Input`, `GELU` and `ReLU` lines.
2. **Change one number.** Find the line `sample = torch.tensor([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0])`. These are the eight test numbers the script feeds to GELU and to ReLU. Change the first one, `-3.0`, to `-2.0` and run the script again. The list now holds `-2.0` and `2.0`, two numbers of the same size with opposite signs.
3. **Compare.** What does GELU print for `-2.0`, and what does it print for `2.0`? What does ReLU print for the same two numbers? Which of the two, GELU or ReLU, gives `-3.0` and `-2.0` exactly the same result?
4. **Explain it.** In two or three sentences a manager could follow, say what GELU does to a positive number and to a negative number of the same size, and why the chapter calls GELU a dimmer and ReLU a switch.

**Hand in:** both sets of three lines, your answers to step 3, and your sentences.
