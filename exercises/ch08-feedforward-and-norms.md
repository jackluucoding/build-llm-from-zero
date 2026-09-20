# Chapter 8: Feed-Forward and Norms: Exercises

Read the chapter online: https://jackluu.io/book/section-2-attention/ch08-feedforward-and-norms/

## Try It

Open `src/ch07_feedforward.py` and change the LayerNorm input to have a massive spread: `x_single = torch.randn(config.n_embd) * 1000 + 500`. Run the script again. You will see that LayerNorm still perfectly tames it back to a mean of 0 and a standard spread of 1.

**In Business**
Imagine you are building an assistant to draft company emails in your house style. If the system is unstable, it might output gibberish. LayerNorm acts like a manager double-checking work between steps, ensuring the data never drifts too far off track before passing it to the next team.

## Check Your Understanding

1. Why does the feed-forward network expand its input size by 4?
2. What happens to a strongly negative number when it passes through GELU?
3. What is the average value of a token's numbers immediately after LayerNorm?
