# Chapter 7: Multi-Head Attention: Exercises

Read the chapter online: https://jackluu.io/book/section-2-attention/ch07-multi-head-attention/

## Try It

Change the number of heads in `src/utils/config.py` by setting `n_heads = 8` instead of 4. Run the script again and notice how the `head_size` automatically drops to 16, so the final concatenated size is still 128 (8 × 16 = 128). The model can have more perspectives, but each one has less detail.

**Watch Out**
For multi-head attention to work cleanly, your embedding dimension (`n_embd`) must be divisible by your number of heads (`n_heads`). If you try `n_embd = 128` and `n_heads = 5`, the program will crash because it cannot divide the channels equally.

## Check Your Understanding

1. Why is one attention head not enough to understand complex text?
2. If `n_embd = 256` and `n_heads = 8`, what is the `head_size`?
3. Four heads of 32 hold the same number of weights as one head of 128. Why does splitting cost nothing?
4. Why do we concatenate the heads rather than average them?
5. In the run above, head 2's attention was almost flat. What does that tell you about the claim that each head learns its own linguistic role?
6. What is the purpose of the final projection layer?

## Assignment

**Assignment 7: Same heads, new word**

**About 15 minutes. You change one word and write no new code.** Work in this chapter's Colab notebook or on your own laptop.

1. **Run it.** Run `src/examples/ch07_heads_differ.py` as it is. Copy the last line it prints, the one that starts `Sharpest focus per head`.
2. **Change one word.** Near the top of the file, find `PROMPT = "JULIET: O Romeo"`. The prompt is the line of text the model reads. Change `Romeo` to `love`, keep the quotation marks, and run the script again.
3. **Compare.** Which character does each head focus on most now? Which heads pick the character just before the last one in both runs? Which two heads have the smallest numbers in both runs?
4. **Explain it.** In two or three sentences a manager could follow, say how the four heads differ, and why it is worth testing a second prompt before you say what job a head does.

**Hand in:** both `Sharpest focus per head` lines, your answers to step 3, and your sentences.
