# Building an LLM from Zero
### So Simple You Can Teach It to Your Kids

> *"Any sufficiently advanced technology is indistinguishable from magic."*
> (Arthur C. Clarke)
>
> *"Until you build it yourself, then it's just a lot of matrix multiplications."*
> (This Tutorial)

---

Welcome! This is a step-by-step guide to building a GPT-style language model from scratch. By the end, you'll have a working model that generates Shakespeare-like text, and you'll understand every single line of code that makes it happen.

**No GPU. No cloud. No magic.** Just Python, PyTorch, and your laptop.

---

## Who Is This For?

You, if you:
- Know basic Python (variables, functions, loops, classes)
- Are curious about how ChatGPT actually works under the hood
- Have ever wondered: *"Is it really just predicting the next word?"* (Spoiler: yes. Mostly.)

You do **not** need to know calculus, linear algebra, or any machine learning framework. We explain everything from the very beginning.

---

## What You Will Build

A character-level GPT model (~825,000 parameters) trained on the complete works of Shakespeare. After training, it generates text like this:

```
ROMEO:
What is a man, if his chief good and market
Of his time be but to sleep and feed?
A beast, no more.
```

(Approximately. Your model's output will vary, it has its own creative flair.)

---

## Before You Start

Complete the setup in [Chapter 1: Setting Up Your Environment](section-1-foundations/ch01-environment-setup.md) before reading Chapter 2. It walks you through installing Python, Git, PyTorch, and downloading the dataset, step by step.

---

## Table of Contents

[**Preface: Why I Wrote This Book**](preface.md)

### Part I: Setup
*Getting your computer ready.*

| Chapter | Title | Key Idea |
|---------|-------|----------|
| [Ch 01](section-1-foundations/ch01-environment-setup.md) | Setting Up Your Environment | Python, Git, PyTorch |

### Part II: Foundations
*Understanding the ingredients before we cook.*

| Chapter | Title | Key Idea |
|---------|-------|----------|
| [Ch 02](section-1-foundations/ch02-what-is-an-llm.md) | What Is a Language Model? | Next-token prediction |
| [Ch 03](section-1-foundations/ch03-tensors-and-pytorch.md) | Tensors and PyTorch Basics | The "smart array" |
| [Ch 04](section-1-foundations/ch04-tokenization.md) | Tokenization | Text to numbers |
| [Ch 05](section-1-foundations/ch05-embeddings.md) | Embeddings | Numbers to meaning |

### Part III: The Attention Mechanism
*The secret ingredient that makes transformers special.*

| Chapter | Title | Key Idea |
|---------|-------|----------|
| [Ch 06](section-2-attention/ch06-self-attention.md) | Self-Attention | Tokens talking to each other |
| [Ch 07](section-2-attention/ch07-multi-head-attention.md) | Multi-Head Attention | Multiple perspectives |
| [Ch 08](section-2-attention/ch08-feedforward-and-norms.md) | Feed-Forward and Layer Norm | Thinking it over |

### Part IV: The Transformer
*Assembling the engine.*

| Chapter | Title | Key Idea |
|---------|-------|----------|
| [Ch 09](section-3-the-transformer/ch09-transformer-block.md) | The Transformer Block | One complete unit |
| [Ch 10](section-3-the-transformer/ch10-full-gpt-architecture.md) | The Full GPT Architecture | The whole stack |
| [Ch 11](section-3-the-transformer/ch11-causal-language-modeling.md) | Causal Language Modeling | No cheating! |

### Part V: Training
*Teaching the model to write.*

| Chapter | Title | Key Idea |
|---------|-------|----------|
| [Ch 12](section-4-training/ch12-dataset-and-dataloader.md) | Dataset and DataLoader | Loading the dishwasher |
| [Ch 13](section-4-training/ch13-training-loop.md) | The Training Loop | Repeat until smart |
| [Ch 14](section-4-training/ch14-checkpointing.md) | Checkpointing | Saving your game |

### Part VI: Generation
*Making it write.*

| Chapter | Title | Key Idea |
|---------|-------|----------|
| [Ch 15](section-5-generation/ch15-greedy-and-sampling.md) | Greedy and Sampling | Safe vs. risky |
| [Ch 16](section-5-generation/ch16-temperature-and-topk.md) | Temperature and Top-k | The creativity dial |
| [Ch 17](section-5-generation/ch17-putting-it-all-together.md) | Putting It All Together | The full pipeline |

---

*Ready? Start with Chapter 1 to set up your environment, then dive into Chapter 2.*
