# Chapter 9: The Transformer Block: Exercises

Read the chapter online: https://jackluu.io/book/section-3-the-transformer/ch09-transformer-block/

## Try It

Edit `src/ch08_transformer_block.py` and modify the number of layers in the stack from `config.n_layers` to `12`. Execute the script once more. Keep track of how the total number of parameters increases. The output shape stays precisely the same.

**In Business**
Software should be built with building blocks. A Transformer block is such a building block. Your house-style writing assistant isn't good enough? Add more Transformer blocks and train it longer. Don't try to reinvent the wheel.

## Check Your Understanding

1. Why do we add the attention output to the original input (`x + attention`)?
2. What does "pre-norm" mean?
3. Why does the Transformer block output exactly the same shape it took in?

## Assignment

**Assignment 9: New numbers, same highway**

**About 15 minutes. You change one number and write no new code.** Work in this chapter's Colab notebook or on your own laptop.

1. **Run it.** Run `src/ch08_transformer_block.py` as it is. Copy four lines: `Original token norm`, `Attention output norm`, `After residual norm` and `Total params`. Here a norm is one number that says how large a token's list of numbers is overall.
2. **Change one number.** Find the line `torch.manual_seed(42)`. The seed fixes the random starting numbers, so every run prints the same values. Change `42` to any whole number you like and run the script again.
3. **Compare.** Which of the four lines changed, and which did not? In your new run, which is larger, the original token or the attention output? Is `After residual norm` closer to the original token or to the attention output?
4. **Explain it.** In two or three sentences a manager could follow, say what the three norm lines show about a residual connection: what the block keeps and what it adds.

**Hand in:** both sets of four lines, your answers to step 3, and your sentences.
