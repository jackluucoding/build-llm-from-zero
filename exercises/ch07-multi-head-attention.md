# Chapter 7: Multi-Head Attention: Exercises

Read the chapter online: https://jackluu.io/book/section-2-attention/ch07-multi-head-attention/

## Try It

Change the number of heads in `src/utils/config.py`. Set `n_heads = 8` instead of 4. Run the script again. Notice how the `head_size` automatically drops to 16, so the final concatenated size is still 128 (8 × 16 = 128). The model can have more perspectives, but each one has less detail.

**Watch Out**
For multi-head attention to work cleanly, your embedding dimension (`n_embd`) must be perfectly divisible by your number of heads (`n_heads`). If you try `n_embd = 128` and `n_heads = 5`, the program will crash because it cannot divide the channels equally.

## Check Your Understanding

1. Why is one attention head not enough to understand complex text?
2. If `n_embd = 256` and `n_heads = 8`, what is the `head_size`?
3. Four heads of 32 hold the same number of weights as one head of 128. Why does splitting cost nothing?
4. Why do we concatenate the heads rather than average them?
5. In the run above, head 2's attention was almost flat. What does that tell you about the claim that each head learns its own linguistic role?
6. What is the purpose of the final projection layer?
