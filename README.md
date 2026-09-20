# Building an LLM from Zero: Code and Exercises

Companion code for the book **Building an LLM from Zero: Look Inside the Black Box**
by Truong (Jack) Luu. The book builds a small GPT-style language model from
scratch in Python and PyTorch, in 17 chapters and 3 modules, and trains it on an
ordinary laptop CPU.

- Read the book online: https://jackluu.io/book/
- Download the PDF: https://jackluu.io/files/building-an-llm-from-zero.pdf
- Download the EPUB: https://jackluu.io/files/building-an-llm-from-zero.epub

This repository holds the runnable code for every chapter, the short "Try It"
examples, the exercises, and a notebook per chapter. The book explains each file
line by line.

## Run it in your browser

Every chapter has a notebook in `notebooks/`. They need nothing installed: open one
in Colab and the first cell fetches the code, installs PyTorch and downloads the text.

- Chapter 1: [notebook](notebooks/ch01-environment-setup.ipynb) | [open in Colab](https://colab.research.google.com/github/jackluucoding/build-llm-from-zero/blob/main/notebooks/ch01-environment-setup.ipynb)
- Chapter 2: [notebook](notebooks/ch02-what-is-an-llm.ipynb) | [open in Colab](https://colab.research.google.com/github/jackluucoding/build-llm-from-zero/blob/main/notebooks/ch02-what-is-an-llm.ipynb)
- Chapter 3: [notebook](notebooks/ch03-tensors-and-pytorch.ipynb) | [open in Colab](https://colab.research.google.com/github/jackluucoding/build-llm-from-zero/blob/main/notebooks/ch03-tensors-and-pytorch.ipynb)
- Chapter 4: [notebook](notebooks/ch04-tokenization.ipynb) | [open in Colab](https://colab.research.google.com/github/jackluucoding/build-llm-from-zero/blob/main/notebooks/ch04-tokenization.ipynb)
- Chapter 5: [notebook](notebooks/ch05-embeddings.ipynb) | [open in Colab](https://colab.research.google.com/github/jackluucoding/build-llm-from-zero/blob/main/notebooks/ch05-embeddings.ipynb)
- Chapter 6: [notebook](notebooks/ch06-self-attention.ipynb) | [open in Colab](https://colab.research.google.com/github/jackluucoding/build-llm-from-zero/blob/main/notebooks/ch06-self-attention.ipynb)
- Chapter 7: [notebook](notebooks/ch07-multi-head-attention.ipynb) | [open in Colab](https://colab.research.google.com/github/jackluucoding/build-llm-from-zero/blob/main/notebooks/ch07-multi-head-attention.ipynb)
- Chapter 8: [notebook](notebooks/ch08-feedforward-and-norms.ipynb) | [open in Colab](https://colab.research.google.com/github/jackluucoding/build-llm-from-zero/blob/main/notebooks/ch08-feedforward-and-norms.ipynb)
- Chapter 9: [notebook](notebooks/ch09-transformer-block.ipynb) | [open in Colab](https://colab.research.google.com/github/jackluucoding/build-llm-from-zero/blob/main/notebooks/ch09-transformer-block.ipynb)
- Chapter 10: [notebook](notebooks/ch10-full-gpt-architecture.ipynb) | [open in Colab](https://colab.research.google.com/github/jackluucoding/build-llm-from-zero/blob/main/notebooks/ch10-full-gpt-architecture.ipynb)
- Chapter 11: [notebook](notebooks/ch11-causal-language-modeling.ipynb) | [open in Colab](https://colab.research.google.com/github/jackluucoding/build-llm-from-zero/blob/main/notebooks/ch11-causal-language-modeling.ipynb)
- Chapter 12: [notebook](notebooks/ch12-dataset-and-dataloader.ipynb) | [open in Colab](https://colab.research.google.com/github/jackluucoding/build-llm-from-zero/blob/main/notebooks/ch12-dataset-and-dataloader.ipynb)
- Chapter 13: [notebook](notebooks/ch13-training-loop.ipynb) | [open in Colab](https://colab.research.google.com/github/jackluucoding/build-llm-from-zero/blob/main/notebooks/ch13-training-loop.ipynb)
- Chapter 14: [notebook](notebooks/ch14-checkpointing.ipynb) | [open in Colab](https://colab.research.google.com/github/jackluucoding/build-llm-from-zero/blob/main/notebooks/ch14-checkpointing.ipynb)
- Chapter 15: [notebook](notebooks/ch15-greedy-and-sampling.ipynb) | [open in Colab](https://colab.research.google.com/github/jackluucoding/build-llm-from-zero/blob/main/notebooks/ch15-greedy-and-sampling.ipynb)
- Chapter 16: [notebook](notebooks/ch16-temperature-and-topk.ipynb) | [open in Colab](https://colab.research.google.com/github/jackluucoding/build-llm-from-zero/blob/main/notebooks/ch16-temperature-and-topk.ipynb)
- Chapter 17: [notebook](notebooks/ch17-putting-it-all-together.ipynb) | [open in Colab](https://colab.research.google.com/github/jackluucoding/build-llm-from-zero/blob/main/notebooks/ch17-putting-it-all-together.ipynb)

## Setup

Chapter 1 walks through every step. In short, with Python 3.11 or newer:

```bash
git clone https://github.com/jackluucoding/build-llm-from-zero.git
cd build-llm-from-zero
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install numpy requests matplotlib
python src/utils/download_data.py
python src/ch00_setup_check.py
```

Run every file from the repository folder, for example `python src/ch03_tokenizer.py`.
File numbers are one lower than chapter numbers because the setup check is `ch00`.

## Chapters and files

| Ch | Chapter | Main file | Try It examples |
|---|---|---|---|
| 1 | [Environment Setup](https://jackluu.io/book/section-1-foundations/ch01-environment-setup/) | `src/utils/download_data.py`, `src/ch00_setup_check.py` | - |
| 2 | [What Is an LLM?](https://jackluu.io/book/section-1-foundations/ch02-what-is-an-llm/) | (concepts only) | `src/examples/ch02_next_char.py` |
| 3 | [Tensors and PyTorch](https://jackluu.io/book/section-1-foundations/ch03-tensors-and-pytorch/) | `src/ch02_tensors.py` | - |
| 4 | [Tokenization](https://jackluu.io/book/section-1-foundations/ch04-tokenization/) | `src/ch03_tokenizer.py` | `src/examples/ch04_tokenizer_demo.py` |
| 5 | [Embeddings](https://jackluu.io/book/section-1-foundations/ch05-embeddings/) | `src/ch04_embeddings.py` | - |
| 6 | [Self-Attention (Single Head)](https://jackluu.io/book/section-2-attention/ch06-self-attention/) | `src/ch05_self_attention.py` | - |
| 7 | [Multi-Head Attention](https://jackluu.io/book/section-2-attention/ch07-multi-head-attention/) | `src/ch06_multihead_attention.py` | `src/examples/ch07_head_budget.py`, `src/examples/ch07_heads_differ.py` |
| 8 | [Feed-Forward and Norms](https://jackluu.io/book/section-2-attention/ch08-feedforward-and-norms/) | `src/ch07_feedforward.py` | - |
| 9 | [The Transformer Block](https://jackluu.io/book/section-3-the-transformer/ch09-transformer-block/) | `src/ch08_transformer_block.py` | - |
| 10 | [The Full GPT Architecture](https://jackluu.io/book/section-3-the-transformer/ch10-full-gpt-architecture/) | `src/ch09_gpt_model.py` | - |
| 11 | [Causal Language Modeling](https://jackluu.io/book/section-3-the-transformer/ch11-causal-language-modeling/) | `src/ch10_causal_lm.py` | `src/examples/ch11_loss_demo.py` |
| 12 | [Dataset and DataLoader](https://jackluu.io/book/section-4-training/ch12-dataset-and-dataloader/) | `src/ch11_dataloader.py` | `src/examples/ch12_batch_speed.py`, `src/examples/ch12_batching_demo.py` |
| 13 | [The Training Loop](https://jackluu.io/book/section-4-training/ch13-training-loop/) | `src/ch12_train.py` | `src/examples/ch13_autograd_demo.py`, `src/examples/ch13_learning_rate.py`, `src/examples/ch13_loss_meaning.py` |
| 14 | [Checkpointing](https://jackluu.io/book/section-4-training/ch14-checkpointing/) | `src/ch13_checkpoint.py` | `src/examples/ch14_inspect_checkpoint.py` |
| 15 | [Greedy and Sampling](https://jackluu.io/book/section-5-generation/ch15-greedy-and-sampling/) | `src/ch14_generate_greedy.py` | `src/examples/ch15_different_prompts.py` |
| 16 | [Temperature and Top-k](https://jackluu.io/book/section-5-generation/ch16-temperature-and-topk/) | `src/ch15_generate_sampling.py` | `src/examples/ch16_explore_temp.py`, `src/examples/ch16_temperature.py` |
| 17 | [Putting It All Together](https://jackluu.io/book/section-5-generation/ch17-putting-it-all-together/) | `src/ch16_full_pipeline.py` | `src/examples/ch17_deploy.py` |

## Exercises

Each chapter's hands-on tasks and review questions:

- [Chapter 1](exercises/ch01-environment-setup.md)
- [Chapter 2](exercises/ch02-what-is-an-llm.md)
- [Chapter 3](exercises/ch03-tensors-and-pytorch.md)
- [Chapter 4](exercises/ch04-tokenization.md)
- [Chapter 5](exercises/ch05-embeddings.md)
- [Chapter 6](exercises/ch06-self-attention.md)
- [Chapter 7](exercises/ch07-multi-head-attention.md)
- [Chapter 8](exercises/ch08-feedforward-and-norms.md)
- [Chapter 9](exercises/ch09-transformer-block.md)
- [Chapter 10](exercises/ch10-full-gpt-architecture.md)
- [Chapter 11](exercises/ch11-causal-language-modeling.md)
- [Chapter 12](exercises/ch12-dataset-and-dataloader.md)
- [Chapter 13](exercises/ch13-training-loop.md)
- [Chapter 14](exercises/ch14-checkpointing.md)
- [Chapter 15](exercises/ch15-greedy-and-sampling.md)
- [Chapter 16](exercises/ch16-temperature-and-topk.md)
- [Chapter 17](exercises/ch17-putting-it-all-together.md)

## Data

`src/utils/download_data.py` downloads the Tiny Shakespeare text from Andrej Karpathy's
char-rnn project (https://github.com/karpathy/char-rnn) into `src/data/`.

## License

- Code (`src/`): MIT, see [LICENSE](LICENSE).
- Book text and exercises (`exercises/`): CC BY-NC 4.0, see [LICENSE-BOOK](LICENSE-BOOK).

## Author

Truong (Jack) Luu, Ph.D., Assistant Professor of Information Systems and Analytics,
McCoy College of Business, Texas State University. https://jackluu.io
