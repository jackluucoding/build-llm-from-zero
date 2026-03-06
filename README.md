# Building an LLM from Zero
## So Simple You Can Teach It to Your Kids

A step-by-step tutorial for building a GPT-style language model entirely from scratch using PyTorch. By the end, you will have trained a model that generates Shakespeare-like text, writing every line of code yourself.

**No GPU required.** Designed to run on a regular laptop in about 20-30 minutes.

---

## Who This Is For

- Beginners who learn by building (knows basic Python, no ML background needed)
- Instructors and professors looking for classroom-ready, easy-to-setup demo materials
- IT/IS professionals who want to understand LLMs from the ground up
- Parents who want to teach their kids how AI actually works, hands-on

**You do not need to know** calculus, linear algebra, or any deep learning framework. Every concept is explained from scratch with simple analogies before any math or code is introduced.

---

## What You Will Build

A character-level GPT model (~825,000 parameters) trained on the complete works of Shakespeare. After training, it generates new text in a similar style:

```
ROMEO:
What light through yonder window breaks?
It is the east, and Juliet is the sun.
```

(Approximately, your model's output will vary!)

---

## Hardware Requirements

- Any modern laptop (Windows, Mac, or Linux)
- **8 GB RAM minimum** (16 GB recommended)
- **No GPU required**, designed for CPU-only training
- Tested on: ThinkPad T14 Gen 1, Intel Core i7, 32 GB RAM (released 2020)
- Expected training time: **~20-30 minutes** on the above machine

---

## Prerequisites

- Python 3.10 or newer
- Basic Python: variables, functions, loops, classes
- That's it!

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/yourname/build-llm-from-zero
cd build-llm-from-zero
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies

Install PyTorch (CPU-only, saves ~1.8 GB compared to the GPU version):
```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install numpy requests matplotlib
```

### 4. Download the dataset
```bash
python src/utils/download_data.py
```

### 5. Run the full pipeline (optional, jumps straight to the end)
```bash
python src/ch16_full_pipeline.py
```

### 6. Or follow chapter by chapter
```bash
python src/ch02_tensors.py
python src/ch03_tokenizer.py
# ... and so on
```

---

## Book Structure

The tutorial begins with a Preface, then continues with 6 parts and 17 chapters. Each chapter has:
- A **Theory** page (in `book/`) explaining the concept with analogies
- A **Code** file (in `src/`) that you can run immediately

| Part | Chapters | Topic |
|---------|----------|-------|
| Preface | - | Why I wrote this book |
| Part I: Setup | Ch 01 | Setting up your environment |
| Part II: Foundations | Ch 02–05 | What is an LLM? Tensors, tokenization, embeddings |
| Part III: The Attention Mechanism | Ch 06–08 | Self-attention, multi-head attention, feed-forward layers |
| Part IV: The Transformer | Ch 09–11 | Transformer block, full GPT architecture, causal masking |
| Part V: Training | Ch 12–14 | Dataset, training loop, checkpointing |
| Part VI: Generation | Ch 15–17 | Greedy decoding, sampling, temperature, top-k |

Read the book online: [`book/index.md`](book/index.md)

---

## Repository Structure

```
build-llm-from-zero/
├── README.md               ← You are here
├── requirements.txt        ← Python dependencies
├── book/                   ← Tutorial chapters (Markdown)
│   ├── index.md
│   ├── section-1-foundations/
│   ├── section-2-attention/
│   ├── section-3-the-transformer/
│   ├── section-4-training/
│   └── section-5-generation/
└── src/                    ← Runnable Python source files
    ├── utils/
    │   ├── config.py       ← All hyperparameters
    │   └── download_data.py
    ├── ch02_tensors.py
    ├── ch03_tokenizer.py
    ├── ...
    └── ch16_full_pipeline.py
```

---

## Author

**Truong (Jack) Luu** — [jackluu.io](https://jackluu.io) — [hi@jackluu.io](mailto:hi@jackluu.io)

Questions, feedback, or just want to say your model generated something beautiful? Feel free to reach out.

---

## Acknowledgments

[Claude](https://claude.ai) (by Anthropic) was used as an AI assistant throughout this project to help with writing plans, generating code drafts, and editing prose. All code was reviewed, tested, and validated by the author on a local machine. All mistakes, if any, are my own.

---

## License

This project uses two licenses depending on the content type:

| Content | License | What it means |
|---------|---------|---------------|
| **Source code** (`src/`) | [MIT](LICENSE) | Free to use, modify, and distribute for any purpose including commercial |
| **Book text** (`book/`) | [CC BY-NC 4.0](LICENSE-BOOK) | Free to read and share with attribution, but cannot be sold commercially by others |

The book is free to read online. For commercial licensing of the book text
(print, ebook, or course use), contact [hi@jackluu.io](mailto:hi@jackluu.io).
