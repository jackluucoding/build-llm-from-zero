# Chapter 6: Self-Attention (Single Head): Exercises

Read the chapter online: https://jackluu.io/book/section-2-attention/ch06-self-attention/

## Try It

Modify the demo at the bottom of `src/ch05_self_attention.py` to remove the division by `scale`. Change `scale = HEAD_SIZE ** -0.5` to `scale = 1.0`. Execute it again. The weights become extreme: token 4 increases from 0.214 to 0.383 while tokens 2 and 3 plummet to around 0.015. Without scaling, most of the weight accrues to only a few tokens.

```console
$ python src/ch05_self_attention.py
...
  token 0: 0.257  #######
  token 1: 0.293  ########
  token 2: 0.016
  token 3: 0.015
  token 4: 0.383  ###########
  token 5: 0.036  #
...
```

**Watch Out**
Take note of the `transpose` step. We only want to swap the final two dimensions (Time and `head_size`) for the purposes of computing the dot product. Using `.T` could also flip your Batch dimension and break your shape math, so be sure to use `.transpose(-2, -1)`.

```console
$ python src/ch05_self_attention.py
...
RuntimeError: The size of tensor a (2) must match the size of tensor b (32) at non-singleton dimension 0
```

## Check Your Understanding

1. What is the difference between a Query and a Key?
2. Why do we divide the attention scores by the square root of the head size?
3. What happens to a score of negative infinity when passed through softmax?
4. Why is the causal mask necessary for a language model?

## Assignment

**Assignment 6: New numbers, same mask**

**About 15 minutes. You change one number and write no new code.** Work in this chapter's Colab notebook or on your own laptop.

1. **Run it.** Execute `src/ch05_self_attention.py` as is. Copy the ten weights printed for token 5.
2. **Change one number.** Find the line `torch.manual_seed(42)` near the bottom of `src/ch05_self_attention.py`. The seed makes sure the starting random numbers are always the same, which ensures we print the same weights each time. Replace the `42` with any whole number of your choosing and execute the script again.
3. **Compare.** Which of the ten weights changed? Which weights stayed the same? What do the weights across tokens 0-5 sum to now?
4. **Explain it.** In two or three sentences suitable for a manager to understand, explain why four of the rows print the same value regardless of seed.

**Hand in:** both sets of ten weights, your answers to step 3, and your sentences.
