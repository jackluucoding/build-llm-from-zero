# Chapter 16: Temperature and Top-k: Exercises

Read the chapter online: https://jackluu.io/book/section-5-generation/ch16-temperature-and-topk/

## Try It

Open the existing script `src/examples/ch16_explore_temp.py` and run it. It uses a very low temperature (`0.1`), and you will notice that it behaves almost exactly like greedy decoding, picking the safe top character every time.

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

## Assignment

**Assignment 16: Turn up the heat**

**About 15 minutes. You change one number and write no new code.** Work in this chapter's Colab notebook or on your own laptop.

1. **Run it.** Run `src/examples/ch16_explore_temp.py` as it is. Copy the text the model writes after `JULIET:`.
2. **Change one number.** Near the bottom of the file, find `temperature=0.1`. Temperature is the number the scores are divided by before the model draws a character. Change `0.1` to `2.0` and run the script again. The line `Generating with T=0.1...` is a fixed label, so it will still say 0.1.
3. **Compare.** Which text keeps repeating the same words? Which one is full of made-up words? Which one is closer to real English?
4. **Explain it.** In two or three sentences a manager could follow, say why neither setting would suit an assistant that writes to customers, and which way you would turn the dial from each one.

**Hand in:** both texts, your answers to step 3, and your sentences.
