# Chapter 6: Self-Attention (Single Head): Exercises

Read the chapter online: https://jackluu.io/book/section-2-attention/ch06-self-attention/

## Try It

Remove the `scale` division in the demo at the bottom of `src/ch05_self_attention.py` by changing `scale = HEAD_SIZE ** -0.5` to `scale = 1.0`. Run it again. Notice how the weights become much more extreme: token 4 rises from 0.214 to 0.383, and tokens 2 and 3 fall to about 0.015. The scaling is what keeps the model flexible and learning smoothly.

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
Be careful with the `transpose` step, because we only want to swap the last two dimensions (Time and `head_size`) to compute the dot product properly. If you use `.T`, it might flip the Batch dimension too, crashing your shape calculations, so always use `.transpose(-2, -1)`.

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

1. **Run it.** Run `src/ch05_self_attention.py` as it is. Copy the ten weights it prints for token 5.
2. **Change one number.** Near the bottom of the file, find `torch.manual_seed(42)`. The seed fixes the random starting numbers, so every run prints the same weights. Change `42` to any whole number you like and run the script again.
3. **Compare.** Which of the ten weights changed? Which ones did not? What do the weights for tokens 0 to 5 add up to now?
4. **Explain it.** In two or three sentences a manager could follow, say why four of the rows print the same value under any seed.

**Hand in:** both sets of ten weights, your answers to step 3, and your sentences.
