# Chapter 2: What Is an LLM?: Exercises

Read the chapter online: https://jackluu.io/book/section-1-foundations/ch02-what-is-an-llm/

## Check Your Understanding

1. What does a language model predict?
2. Why is it called "autoregressive"?
3. How does self-supervised learning differ from having a teacher grade the work?

## Assignment

**Assignment 2: New word, new guess**

**About 15 minutes. You change one word and write no new code.** Work in this chapter's Colab notebook or on your own laptop.

1. **Run it.** Run `src/examples/ch02_next_char.py` as it is. Copy the five letters and their counts.
2. **Change one word.** Find the line `prefix = "the "`. The prefix is the piece of text the script looks for. Change `the` to `my` and keep the space before the closing quotation mark, so the line reads `prefix = "my "`. Run the script again.
3. **Compare.** Which letter comes first now? Which letters are in both lists? Name one word Shakespeare could write after `my` that starts with the new first letter.
4. **Explain it.** In two or three sentences a manager could follow, say why the most likely next letter changed when you changed one word.

**Hand in:** both lists of five letters and counts, your answers to step 3, and your sentences.
