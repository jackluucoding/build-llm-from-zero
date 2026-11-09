# Chapter 3: Tensors and PyTorch: Exercises

Read the chapter online: https://jackluu.io/book/section-1-foundations/ch03-tensors-and-pytorch/

## Check Your Understanding

1. If an output has shape `(32, 128, 128)`, what does that mean in our `(B, T, C)` notation?
2. Why should the inner dimensions match when doing matrix multiplication?
3. What's the difference between logits and probabilities?

## Assignment

**Assignment 3: One number, new shape**

**About 15 minutes. You change one number and write no new code.** Work in this chapter's Colab notebook or on your own laptop.

1. **Run it.** Run `src/ch02_tensors.py` without changing anything. Copy the line of output that begins `3D matmul:`.
2. **Change one number.** Locate the line `W = torch.randn(8, 4)`. `W` stores the transformation's weights. It takes in 8 numbers for each token and outputs 4. Replace the `4` with a `6` and run the script again.
3. **Compare.** Which numbers on the `3D matmul:` line were different? Which were the same? How many numbers are used to describe each token now?
4. **Explain it.** In two or three sentences understandable by a manager, describe what the multiplication changed about the data and what it didn't change.

**Hand in:** both `3D matmul:` lines, your answers to step 3, and your sentences.
