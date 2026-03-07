# Preface: Why I Wrote This Book

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

## Who Is This Book For?

**Beginners who learn by building.**
You know some Python, you're curious about AI, and you want to actually understand how it works, not just use someone else's model. You want to go from zero to a working language model, line by line, without getting stuck in tutorial hell.

**Instructors and professors.**
You want classroom-ready material: concepts clear enough to teach, code that runs on a student's laptop in a single class session, and a structure that maps cleanly to a lecture. This book is designed to be that.

**IT/IS professionals.**
You work with AI tools every day but want to understand what's happening under the hood. You don't need a PhD, you need a clear explanation of the architecture, a working example, and code you can actually read and modify.

**Parents teaching their kids.**
AI is everywhere. If you want to introduce your child to how it actually works, not just how to use it, this book gives you a concrete, hands-on project you can work through together. Build something real. Ask questions. Break it and fix it. That's how learning sticks.

---

## A Note on AI Assistance

This book was written by [Truong (Jack) Luu](https://jackluu.io/). Claude (by Anthropic) was used as an AI assistant to help with writing plans, generating code drafts, and editing prose. All code was reviewed, edited, and tested by the author on a local machine. Every example in this book runs exactly as shown.

---

*Truong (Jack) Luu*
*[jackluu.io](https://jackluu.io)*

---

*Next up: [Chapter 1, Setting Up Your Environment](section-1-foundations/ch01-environment-setup.md)*
