# Chapter 5: Embeddings: Exercises

Read the chapter online: https://jackluu.io/book/section-1-foundations/ch05-embeddings/

## Try It

Modify `src/ch04_embeddings.py`. Add one additional token ID to each row of `token_ids` (it can be any integer from 0-64), and update `T` from 5 to 6. Then run the script again. You should see that the output shape increases from `[2, 5, 128]` to `[2, 6, 128]`, but your parameter counts remain precisely the same. Lookup tables do not care how much text you run through them at once.

**Watch Out**
A mistake that trips people up is trying to look up a token ID that isn't in your vocabulary. So if your `vocab_size` is 65, you have tokens that range from 0-64. If you were to enter a token ID of 65, you would get an "index out of bounds" error.

## Check Your Understanding

1. Why do neural networks struggle with raw token IDs?
2. What is the length of the embedding vector for each character in our model?
3. Why do we need positional embeddings besides the token embeddings?
4. Is the embedding table size affected by the batch size?

## Assignment

**Assignment 5: Bigger ID, bigger numbers?**

**About 15 minutes. You change one number and write no new code.** Work in this chapter's Colab notebook or on your own laptop.

1. **Run it.** Run `src/ch04_embeddings.py` unchanged. Copy the line beginning with `First token's embedding`. This prints out the first 8 of 128 numbers from the embedding for the first token.
2. **Change one number.** Locate the line that says `token_ids = torch.tensor([[3, 14, 7, 2, 50], [10, 22, 45, 1, 8]])`. This line contains ten token IDs. Replace the first `3` with `38`, so that the line begins `token_ids = torch.tensor([[38, 14,`. Run the script again.
3. **Compare.** Did the eight numbers change? Are they larger, now that we used a larger ID? Did any of the shapes change? Did any of the counts under `--- Summary ---` change?
4. **Explain it.** In two or three sentences that your manager could understand, explain what we're using the token ID for here. And say if the size of the ID matters.

**Hand in:** both `First token's embedding` lines, your answers to step 3, and your sentences.
