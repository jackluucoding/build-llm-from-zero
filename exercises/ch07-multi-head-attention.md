# Chapter 7: Multi-Head Attention: Exercises

Read the chapter online: https://jackluu.io/book/section-2-attention/ch07-multi-head-attention/

## Try It

Edit `src/utils/config.py` to use 8 heads instead of 4. Set `n_heads = 8`. Run the script again. Pay attention to how `head_size` is now automatically set to 16 to keep the final concatenated size at 128 (8 × 16 = 128). We can have more viewpoints, but with less granularity within each one.

**Watch Out**
Your embedding dimension (`n_embd`) has to be divisible by your number of heads (`n_heads`) for multi-head attention to operate cleanly. Setting `n_embd = 128` and `n_heads = 5` will crash the program, as it won't be able to cleanly divide the channels.

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

1. **Run it.** Run `src/examples/ch07_heads_differ.py` unaltered. Copy the very last line printed by the program, the one that begins `Sharpest focus per head`.
2. **Change one word.** Find the line near the top of the program that reads `PROMPT = "JULIET: O Romeo"`. This is the line of text that we prompt the model with. Edit `Romeo` to read `love` instead, preserving the quotation marks, and run the program again.
3. **Compare.** Which character is each head focused on most now? Of all the heads, which two heads had the smallest numbers for both runs? For both runs, which heads focused on the character just before the last character?
4. **Explain it.** In two or three sentences that a manager could understand, describe how the four heads differ, and why experimenting with a second prompt is worthwhile before you say what job each head does.

**Hand in:** both `Sharpest focus per head` lines, your answers to step 3, and your sentences.
