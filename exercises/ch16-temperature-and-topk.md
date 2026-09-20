# Chapter 16: Temperature and Top-k: Exercises

Read the chapter online: https://jackluu.io/book/section-5-generation/ch16-temperature-and-topk/

## Try It

Open the existing script `src/examples/ch16_explore_temp.py` and run it. It uses a very low temperature (`0.1`). You will notice it behaves almost exactly like greedy decoding, picking the safe top character every time!

```console
$ python src/examples/ch16_explore_temp.py
--- Prompt: 'JULIET:\n' ---
Generating with T=0.1...
JULIET:
I will the shall the shall be the son th...

```

## Check Your Understanding

1. If you set the temperature to 0.1, what happens to the gap between the highest and lowest scores?
2. Why do we replace the eliminated scores in top-k with negative infinity (`-inf`) instead of `0`?
3. Which step must happen first in the code: applying top-k or calculating softmax probabilities?
