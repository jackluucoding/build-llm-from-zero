# Chapter 10: The Full GPT Architecture

> **Code file**: `src/ch09_gpt_model.py`
> **Run it**: `python src/ch09_gpt_model.py`

---

## Theory

### The Final Assembly

We now have all the components. The full GPT model is just:

```
Input token IDs
     |
Token Embedding  +  Position Embedding
     |
Transformer Block 1
     |
Transformer Block 2
     |
Transformer Block 3
     |
Transformer Block 4
     |
Final LayerNorm
     |
Linear Layer (LM Head)
     |
Output logits (one score per vocabulary token)
```

Let's trace through a concrete example.

---

### Tracing Through the Model

Suppose our input is the sequence `"ROMEO"` (5 characters = 5 tokens):

**Input token IDs**: `[18, 21, 13, 17, 21]`
Shape: `(1, 5)`, one sequence, 5 tokens

**After token + position embedding**:
Shape: `(1, 5, 128)`, each token now has 128 numbers representing it

**After 4 transformer blocks**:
Shape: `(1, 5, 128)`, same shape, but the values now encode rich contextual meaning
- Token 4 (`O`) now "knows" it's the last letter of "ROMEO", who appears in a play

**After final LayerNorm**:
Shape: `(1, 5, 128)`, normalized, same shape

**After LM head (Linear layer)**:
Shape: `(1, 5, 65)`, 65 scores per position (one score per possible character)

Each position makes its own independent prediction, based only on what it has seen so far (thanks to the causal mask):
- Position 0 (`R`): predicts the next character after `R`
- Position 1 (`O`): predicts the next character after `RO`
- Position 2 (`M`): predicts the next character after `ROM`
- Position 3 (`E`): predicts the next character after `ROME`
- Position 4 (`O`): predicts the next character after the full `ROMEO`

We get 5 independent training targets from a single forward pass. The highest score at position 4 = the model's best guess for what follows "ROMEO".

---

### The LM Head

The final linear layer (called the "Language Model head" or LM head) maps from `n_embd = 128` to `vocab_size = 65`:

```python
self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
```

These 65 output scores are called **logits**. Logit is just a technical term for "raw score" - these numbers can be negative or very large, and they do not add up to 1. They are not probabilities yet.

For text generation (Chapter 14), we convert logits to probabilities using softmax, then sample a character.

For training, we use cross-entropy loss directly on the logits. Why skip the softmax during training? Applying log() and softmax together (which cross-entropy does internally) is more numerically stable than converting to probabilities first and then taking log(). The math is equivalent but the computer arithmetic is safer.

---

### Total Parameter Count

```
Token embedding     :     8,320  (65 × 128)
Position embedding  :    16,384  (128 × 128)
4 Transformer blocks:   791,552  (4 × 197,888)
Final LayerNorm     :       256  (128 × 2 for gamma and beta)
LM Head             :     8,320  (128 × 65)
─────────────────────────────────────────────
TOTAL               :   824,832  (~825K)
```

A note on the position embedding shape (128 x 128): we have 128 possible positions (our max sequence length, called `block_size`), and each position embedding is 128 numbers long (same as the token embedding dimension). So 128 positions x 128 numbers per position = 16,384 parameters. Each position gets its own unique 128-number fingerprint.

For comparison:
- GPT-2 Small: 117,000,000 parameters
- Our model: 824,832 parameters (~142x smaller than GPT-2 Small)
- Same architecture, just fewer layers and narrower dimensions.

---

### Why No Bias in the LM Head?

```python
nn.Linear(n_embd, vocab_size, bias=False)
```

Omitting the bias follows GPT-2 convention. In practice, a bias in the final layer provides no measurable quality improvement. A `nn.Linear` layer normally has two learnable components: weights (the transformation matrix) and a bias (an offset added to every output). Setting `bias=False` means we use only the weights - just pure matrix multiplication, no offset. This is a GPT-2 design choice that works well in practice.

---

## Code

> **File**: `src/ch09_gpt_model.py`
> **Run it**: `python src/ch09_gpt_model.py`

This file:
1. Implements the complete `GPT` class
2. Prints a detailed parameter breakdown
3. Runs a forward pass and explains the output

### Key output

```
TOTAL : 824,832

Input  shape: torch.Size([2, 10])   (B, T)
Output shape: torch.Size([2, 10, 65])  (B, T, vocab_size)
```

The model is complete! The output logits at each position predict the next token.

---

## Key Takeaways

- The GPT model = Embeddings → N Transformer Blocks → LayerNorm → LM Head.
- Input shape: `(B, T)` token IDs. Output shape: `(B, T, vocab_size)` logits.
- Total parameters: ~825K (tiny compared to production models, same architecture).
- Logits at position `t` predict what token comes at position `t+1`.

---

*Next up: [Chapter 11, Causal Language Modeling](ch11-causal-language-modeling.md)*
