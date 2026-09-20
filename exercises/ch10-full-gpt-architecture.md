# Chapter 10: The Full GPT Architecture: Exercises

Read the chapter online: https://jackluu.io/book/section-3-the-transformer/ch10-full-gpt-architecture/

## Try It

Open `src/ch09_gpt_model.py` and modify `config.n_layers = 6` just to test. Run the script again. Watch how the parameter count increases. The blocks contain the vast majority of the model's "brain".

**In Business**
Building the final GPT architecture is like assembling a complete production pipeline from modular components. In our house-style writing assistant, we combine a data reader (embeddings), an analysis engine (the blocks), and an output formatter (the LM head). Because it is modular, you can upgrade the engine (add more layers) without rewriting the rest of the pipeline.

## Check Your Understanding

1. If the input sequence has 10 tokens, how many predictions does the model make?
2. Why does the LM head output exactly 65 numbers per token?
3. What is a logit?
