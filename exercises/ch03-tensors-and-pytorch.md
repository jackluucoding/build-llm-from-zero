# Chapter 3: Tensors and PyTorch: Exercises

Read the chapter online: https://jackluu.io/book/section-1-foundations/ch03-tensors-and-pytorch/

## Check Your Understanding

1. What does the shape `(32, 128, 128)` represent in our `(B, T, C)` format?
2. Why do the inner dimensions need to match in matrix multiplication?
3. What is the difference between logits and probabilities?

## Assignment

**Assignment 3: One number, new shape**

**About 15 minutes. You change one number and write no new code.** Work in this chapter's Colab notebook or on your own laptop.

1. **Run it.** Run `src/ch02_tensors.py` as it is. Copy the line of output that starts with `3D matmul:`.
2. **Change one number.** Find the line `W = torch.randn(8, 4)`. `W` holds the transformation weights. It takes in 8 numbers for each token and gives back 4. Change the `4` to `6` and run the script again.
3. **Compare.** Which numbers in the `3D matmul:` line changed? Which stayed the same? How many numbers describe each token now?
4. **Explain it.** In two or three sentences a manager could follow, say what the multiplication changed about the data and what it left alone.

**Hand in:** both `3D matmul:` lines, your answers to step 3, and your sentences.
