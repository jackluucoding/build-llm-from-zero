# Chapter 10: The Full GPT Architecture: Exercises

Read the chapter online: https://jackluu.io/book/section-3-the-transformer/ch10-full-gpt-architecture/

## Try It

Open `src/ch09_gpt_model.py`. Locate the line `config = GPTConfig()` and insert a new line underneath it: `config.n_layers = 6`. Execute the script once more. The parameter count grows, and most of the parameters are in the blocks.

**In Business**
Think of building our final GPT model as building a pipeline out of Lego bricks. You've got your data reader (embeddings), processor (your blocks) and output (the LM head). You can upgrade your house-style GPT's processor by stacking more layers on without having to replace any other parts of your pipeline.

## Check Your Understanding

1. If the input sequence has 10 tokens, how many predictions does the model make?
2. Why does the LM head output exactly 65 numbers per token?
3. What is a logit?

## Assignment

**Assignment 10: Shorter text, same model**

**About 15 minutes. You change one number and write no new code.** Work in this chapter's Colab notebook or on your own laptop.

1. **Run it.** Run `src/ch09_gpt_model.py` as it is. Copy three lines: the `TOTAL` line and the two shape lines, `Input shape` and `Output shape`.
2. **Change one number.** Near the bottom of the file, find the line `B, T = 2, 10`. `B` is how many test sequences the script feeds in, and `T` is how many tokens each one holds. Change `10` to `5`, the length of `ROMEO`, and run the script again.
3. **Compare.** Which numbers in the two shape lines changed, and which did not? Did the `TOTAL` line change? How many sets of 65 scores does the model now give for each sequence? The five scores at the bottom shift too. They come from an untrained model, so set them aside.
4. **Explain it.** In two or three sentences a manager could follow, say what the model hands back when it reads a five-character input such as `ROMEO`, and why a shorter input does not make the model itself any smaller.

**Hand in:** both sets of three lines, your answers to step 3, and your sentences.
