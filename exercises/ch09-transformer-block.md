# Chapter 9: The Transformer Block: Exercises

Read the chapter online: https://jackluu.io/book/section-3-the-transformer/ch09-transformer-block/

## Try It

Open `src/ch08_transformer_block.py` and change the number of layers in the stack from `config.n_layers` to `12`. Run the script again. Notice how the total parameter count grows, but the output shape remains exactly the same. 

**In Business**
When building software systems (like a house-style writing assistant), you want modular, scalable processes. A Transformer block is the ultimate modular unit. If your assistant is not smart enough, you do not have to invent a new architecture; you just stack more blocks and train it longer.

## Check Your Understanding

1. Why do we add the attention output to the original input (`x + attention`)?
2. What does "pre-norm" mean?
3. Why does the Transformer block output exactly the same shape it took in?
