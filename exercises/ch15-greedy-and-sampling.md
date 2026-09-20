# Chapter 15: Greedy and Sampling: Exercises

Read the chapter online: https://jackluu.io/book/section-5-generation/ch15-greedy-and-sampling/

## Try It

Different starting prompts create different context for the model. Let's see how sampling handles new prompts.

```python
# ...
encode_fn = lambda s: encode(s, char_to_id)
decode_fn = lambda ids: decode(ids, id_to_char)

print(f"--- Prompt: 'KING:\\n' ---")
print(generate_sample(model, "KING:\n", encode_fn, decode_fn, cfg, 50))

print(f"\n--- Prompt: 'JULIET:\\n' ---")
print(generate_sample(model, "JULIET:\n", encode_fn, decode_fn, cfg, 50))
```

Lines 6 and 9 generate new text starting from two completely different prompts.

```console
$ python src/examples/ch15_different_prompts.py
--- Prompt: 'KING:\n' ---
KING:
My mine gore to kink on Villence,
For my love grin

--- Prompt: 'JULIET:\n' ---
JULIET:
What reason this bagainVain, I will evers,
As do p

```

## Check Your Understanding

1. Why does greedy decoding often get stuck repeating the same phrase?
2. What PyTorch function do we use to randomly pick a token based on its probabilities?
3. Why do we slice the logits tensor with `[:, -1, :]` in the generation loop?
