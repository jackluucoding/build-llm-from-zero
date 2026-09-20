# Chapter 6: Self-Attention (Single Head): Exercises

Read the chapter online: https://jackluu.io/book/section-2-attention/ch06-self-attention/

## Try It

Remove the `scale` division in `src/ch05_self_attention.py` by changing it to `scale = 1.0`. Run it again. Notice how the weights become much more extreme (some very close to 1.0, others 0.0). The scaling is critical to keep the model flexible and learning smoothly.

**Watch Out**
Be careful with the `transpose` step. We only want to swap the last two dimensions (Time and `head_size`) to compute the dot product properly. If you use `.T`, it might flip the Batch dimension too, crashing your shape calculations. Always use `.transpose(-2, -1)`.

## Check Your Understanding

1. What is the difference between a Query and a Key?
2. Why do we divide the attention scores by the square root of the head size?
3. What happens to a score of negative infinity when passed through softmax?
4. Why is the causal mask necessary for a language model?
