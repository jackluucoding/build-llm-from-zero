# Chapter 2: What Is a Language Model?

> No code in this chapter, just ideas. Grab a snack.

---

## Why Did I Write This Book?

There are already a lot of resources out there for learning about AI and large language models. So why write another one?

Because most of them fall into one of three traps, or they push you straight into **tutorial hell**.

**Tutorial hell** is a state of learning paralysis where you endlessly watch coding tutorials and follow along with instructors, feeling productive, but never actually building anything on your own. You run the code, it works, you feel good, but close the laptop and ask yourself *why* any of it worked, and you draw a blank. You've watched the chef cook a hundred times but have never touched the stove yourself.

The three traps that lead there:

- **Too shallow.** Copy-and-paste the code, follow along, done. You produce an output, but you don't understand any of the decisions behind it. The moment something breaks or changes, you're stuck.
- **Too deep.** Research-paper density, prerequisite courses in linear algebra and calculus, written for people who already have a graduate-level background. Most readers bounce off in chapter two.
- **Too demanding on hardware.** Great content, but assumes you have a modern GPU, a cloud computing account, and several dedicated weekends free. That rules out most people.

This book is a different approach.

The goal is a **balance between theory and code**, enough explanation to genuinely understand what you are building and why each piece is there, paired with real, runnable code you can execute right now on the machine in front of you.

Every chapter follows the same structure: first the idea in plain language with an analogy, then the code that implements it, then a short summary. You are never asked to accept something on faith. If a line of code does something, the chapter explains why.

**Right now** means on a regular laptop. No GPU. No cloud. No special hardware.
This entire book was written and tested on a ThinkPad T14 Gen 1 (Intel Core i7, 32 GB RAM, released 2020), a five-year-old business laptop with no dedicated GPU. The full training run completed in about 20-30 minutes. If it runs there, it will run on yours.

Setup is minimal: Python and PyTorch. That's it. No accounts to create, no clusters to configure.

---

### Who Is This Book For?

**Beginners who learn by building.**
You know some Python, you're curious about AI, and you want to actually understand how it works, not just use someone else's model. You want to go from zero to a working language model, line by line, without getting stuck in tutorial hell.

**Instructors and professors.**
You want classroom-ready material: concepts clear enough to teach, code that runs on a student's laptop in a single class session, and a structure that maps cleanly to a lecture. This book is designed to be that.

**IT/IS professionals.**
You work with AI tools every day but want to understand what's happening under the hood. You don't need a PhD, you need a clear explanation of the architecture, a working example, and code you can actually read and modify.

**Parents teaching their kids.**
AI is everywhere. If you want to introduce your child to how it actually works, not just how to use it, this book gives you a concrete, hands-on project you can work through together. Build something real. Ask questions. Break it and fix it. That's how learning sticks.

---

### A Note on AI Assistance

This book was written by [Truong (Jack) Luu](https://jackluu.io/). Claude (by Anthropic) was used as an AI assistant to help with writing plans, generating code drafts, and editing prose. All code was reviewed, edited, and tested by the author on a local machine. Every example in this book runs exactly as shown.

---

## Theory

### The World's Most Useful Party Trick

Imagine you're texting a friend and you type:

> "I'm so hungry, I could eat a  - "

Before you finish, your phone suggests: **horse**. Or **pizza**. Or **whole building**.

Your phone is doing something called **next-word prediction**, it guesses what word is most likely to come next based on everything you've typed so far.

A **Language Model (LM)** does exactly the same thing, just much better.

Given a sequence of words (or characters), it predicts: *"What comes next?"*

That's it. That's the whole idea. Everything else in this tutorial is just *how* to build a machine that can do this really well.

---

### "But ChatGPT Talks Back to Me..."

Yes, but under the hood, it's still just predicting the next token, one at a time.

Here's how a conversation works at the model level:

1. You type: `"What is the capital of France?"`
2. The model predicts the next token: `"Paris"`, most likely.
3. Then it predicts the token after that: `" is"`, still makes sense.
4. Then: `" the"`, then `" capital"`, then `" of"`, then `" France"`, then `"."`, then it decides to stop.

The model never "thinks" about the whole answer at once. It just keeps predicting the next piece, over and over.

This is called **autoregressive generation** (auto = self, regressive = looking back). The model uses its own previous outputs as inputs for the next step.

Here is what that means concretely: when the model outputs "Paris", that word gets added to the input for the next step. Now the input is "What is the capital of France? Paris", and the model predicts what comes after that. Each new word it generates becomes part of the context for the next word. It is like writing by hand - once you write a word, you cannot erase it, you can only add what comes next.

---

### What Is a "Token"?

A token is the smallest unit the model works with. Depending on the model, a token can be:

- **A character**: `H`, `e`, `l`, `l`, `o` (5 tokens for "Hello")
- **A word piece**: `Hello` (1 token)
- **A word**: `Hello` (1 token, different approach)

In this tutorial, we use **character-level tokens**, the simplest option. Every single character (letters, spaces, punctuation) is one token.

Why character-level? Because:
- The vocabulary is tiny (only 65 characters in Shakespeare)
- There's nothing to install or configure
- You can understand it completely in 5 minutes

Real models like GPT-4 use more complex tokenization (called BPE, Byte Pair Encoding), but the *idea* is the same.

---

### What Makes a Language Model "Large"?

The "L" in LLM stands for **Large**, meaning it has a lot of **parameters**.

Parameters are the numbers inside the model that get adjusted during training. Think of them like dials on a radio: training turns them to the right settings so the model produces good output.

| Model | Parameters | Year |
|-------|-----------|------|
| Our model (this tutorial) | ~825,000 | 2024 |
| GPT-2 Small | 117,000,000 | 2019 |
| GPT-3 | 175,000,000,000 | 2020 |
| GPT-4 (estimated) | ~1,800,000,000,000 | 2023 |

Our model is tiny by comparison. But it uses the **exact same architecture** as GPT-2, just with fewer layers and smaller dimensions.

It's like comparing a toy car to a Formula 1 racer. Same basic design, very different scale.

---

### The Training Process (Big Picture)

Building an LLM happens in two phases:

**Phase 1: Architecture Design**
We design the model's structure, how information flows through it, what operations it performs. This is what Parts 1-3 of this tutorial cover.

**Phase 2: Training**
We show the model millions of examples of text and adjust its parameters to make it better at predicting the next token. The model never sees a human label like "this is good writing." It just sees raw text and learns patterns.

This kind of training is called **self-supervised learning**.

In most machine learning, you need labeled training data: humans manually mark "this photo contains a cat" or "this email is spam." Labels = the correct answers.

With language modeling, the labels are built right into the text. If your training text is "Hello world", you get:
- Input: `H` - Label: `e` (the next character)
- Input: `He` - Label: `l`
- Input: `Hel` - Label: `l`
- ... and so on

The text itself is the answer key. No human labelers needed. That is why you can train on any raw text from the internet - billions of unlabeled pages become billions of training examples automatically. That is the superpower of self-supervised learning.

---

### What We're Building

By the end of this tutorial, you'll have:

```
Input  : "To be or not to be, that is the "
Output : "q"  ← untrained
Output : "question"  ← after training!
```

The model goes from outputting gibberish to outputting Shakespeare. Not because anyone told it what Shakespeare sounds like, but because it learned the patterns from the data itself.

Pretty cool, right?

---

## Key Takeaways

- A language model predicts "what comes next" one token at a time.
- Autoregressive generation = using your own outputs as inputs for the next step.
- Tokens are the basic units, in our case, individual characters.
- Parameters are the learnable numbers inside the model.
- Training = adjusting parameters to get better at next-token prediction.

---

*Next up: [Chapter 3, Tensors and PyTorch Basics](ch03-tensors-and-pytorch.md)*
