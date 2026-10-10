# Chapter 10: The Full GPT Architecture: Exercises

Read the chapter online: https://jackluu.io/book/section-3-the-transformer/ch10-full-gpt-architecture/

## Try It

Open `src/ch09_gpt_model.py`, find the line `config = GPTConfig()` and add a new line under it: `config.n_layers = 6`. Run the script again. Watch how the parameter count increases. The blocks contain the vast majority of the model's "brain".

**In Business**
Building the final GPT architecture is like assembling a complete production pipeline from modular components. In our house-style writing assistant, we combine a data reader (embeddings), an analysis engine (the blocks), and an output formatter (the LM head). Because it is modular, you can upgrade the engine (add more layers) without rewriting the rest of the pipeline.

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
