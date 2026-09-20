# Chapter 5: Embeddings: Exercises

Read the chapter online: https://jackluu.io/book/section-1-foundations/ch05-embeddings/

## Try It

Change the batch size `B` or the sequence length `T` in `src/ch04_embeddings.py`. Run the script again. Notice that the output shapes scale automatically, but the parameter counts stay exactly the same. The lookup tables do not care how much text you process at once.

**Watch Out**
A common mistake is trying to look up a token ID that is larger than the vocabulary size. If your `vocab_size` is 65, the valid IDs are 0 to 64. Passing an ID of 65 will crash the program with an "index out of bounds" error.

## Check Your Understanding

1. Why do neural networks struggle with raw token IDs?
2. How many numbers make up a single character's embedding vector in our model?
3. Why do we need positional embeddings in addition to token embeddings?
4. Does the size of the embedding table depend on the batch size?
