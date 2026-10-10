# Chapter 5: Embeddings: Exercises

Read the chapter online: https://jackluu.io/book/section-1-foundations/ch05-embeddings/

## Try It

In `src/ch04_embeddings.py`, add one more token ID to each row of `token_ids` (any number from 0 to 64) and change `T` from 5 to 6. Run the script again. The output shape grows from `[2, 5, 128]` to `[2, 6, 128]`, but the parameter counts stay exactly the same. The lookup tables do not care how much text you process at once.

**Watch Out**
A common mistake is trying to look up a token ID that is not in the vocabulary. If your `vocab_size` is 65, the valid IDs are 0 to 64. Passing an ID of 65 will crash the program with an "index out of bounds" error.

## Check Your Understanding

1. Why do neural networks struggle with raw token IDs?
2. How many numbers make up a single character's embedding vector in our model?
3. Why do we need positional embeddings in addition to token embeddings?
4. Does the size of the embedding table depend on the batch size?

## Assignment

**Assignment 5: Bigger ID, bigger numbers?**

**About 15 minutes. You change one number and write no new code.** Work in this chapter's Colab notebook or on your own laptop.

1. **Run it.** Run `src/ch04_embeddings.py` as it is. Copy the line that starts with `First token's embedding`. It shows the first 8 of the 128 numbers for the first token.
2. **Change one number.** Find the line `token_ids = torch.tensor([[3, 14, 7, 2, 50], [10, 22, 45, 1, 8]])`. These are ten token IDs. Change the first one, `3`, to `38`, so the line starts with `token_ids = torch.tensor([[38, 14,`. Run the script again.
3. **Compare.** Did the 8 numbers change? Are they bigger, now that the ID is bigger? Did any shape, or any count under `--- Summary ---`, change?
4. **Explain it.** In two or three sentences a manager could follow, say what the token ID is used for here, and whether the size of the ID matters.

**Hand in:** both `First token's embedding` lines, your answers to step 3, and your sentences.
