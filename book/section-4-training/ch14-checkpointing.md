# Chapter 14: Saving and Loading Checkpoints

> **Code file**: `src/ch13_checkpoint.py`
> **Run it**: `python src/ch13_checkpoint.py`

---

## Theory

### Why Checkpoints?

Training on a CPU takes 20-30 minutes. If your laptop crashes, runs out of battery, or you accidentally close the terminal, without checkpoints, you'd lose all that work and start from scratch.

More importantly: once you have a trained model, you want to be able to load it later without retraining. That's what checkpoints are for.

A checkpoint = the model's "save file."

---

### What Is a Checkpoint?

A checkpoint is just a dictionary saved to disk using `torch.save()`:

```python
checkpoint = {
    "model_state": model.state_dict(),   # all the learned weights
    "gpt_cfg"    : config,               # model architecture details
    "step"       : step,                 # how far training had gone
    "val_loss"   : val_loss,             # performance at that point
}
torch.save(checkpoint, "checkpoints/model.pt")
```

`model.state_dict()` is a dictionary of all the model's parameters - every number the model has adjusted during training. This includes: all attention weight matrices (W_q, W_k, W_v for every head in every block), all feed-forward layer weights, all biases, and all layer normalization parameters (gamma and beta). It does NOT include the architecture itself (number of layers, embedding size, etc.) - that is stored separately in `gpt_cfg`. You need both to fully restore a model: the config tells PyTorch how to build the skeleton, and the state_dict fills in all the learned values.

File size: our checkpoint is about **4 MB**. Smaller than a single photo on your phone. Your model's entire brain fits in a thumbnail.

---

### Loading a Checkpoint

To load and use a saved model:

```python
# Load the dictionary from disk
checkpoint = torch.load("checkpoints/model.pt", map_location="cpu")

# Rebuild the model architecture (empty, no weights yet)
config = checkpoint["gpt_cfg"]
model  = GPT(config)

# Fill in the saved weights
model.load_state_dict(checkpoint["model_state"])

# Switch to inference mode (disables dropout)
model.eval()
```

It's like loading a save file in a video game: the game loads the world (architecture) and then restores your character's state (weights).

---

### `model.eval()` vs `model.train()`

Two important mode switches:

**`model.train()`**: Enables dropout (randomly zeroes out ~10% of activations during training). During each forward pass, random neurons get switched off. This forces the model to not rely on any specific neuron always being present - if neuron 42 might disappear, the model has to distribute knowledge across many neurons. This makes the learned features more robust and helps prevent overfitting. Dropout is on by default.

**`model.eval()`**: Disables dropout. All neurons are active. Use this when generating text or evaluating - you want deterministic, full-power outputs.

You can see both in the training loop from Chapter 12: `model.eval()` before computing validation loss, `model.train()` after.

Always switch to `eval()` mode before generation!

---

### Resume Training (Optional)

You can also resume training from a checkpoint:

```python
checkpoint = torch.load("checkpoints/model.pt")
model.load_state_dict(checkpoint["model_state"])
start_step = checkpoint["step"]
# continue training from start_step...
```

For a 3000-step run, this usually isn't needed. But for longer runs on larger models, it's essential.

---

## Code

> **File**: `src/ch13_checkpoint.py`
> **Run it**: `python src/ch13_checkpoint.py`

This file:
1. Checks if a checkpoint exists (creates a dummy one if not)
2. Loads the checkpoint
3. Rebuilds the model
4. Verifies it works with a forward pass
5. Reports the file size

---

## Key Takeaways

- A checkpoint = the model's learned weights + metadata, saved to disk.
- `torch.save()` saves; `torch.load()` + `model.load_state_dict()` restores.
- Always call `model.eval()` before generating text.
- Our checkpoint is ~4 MB, tiny!
- Checkpoints let you use a trained model without retraining.

---

*Next up: [Chapter 15, Greedy Decoding and Sampling](../section-5-generation/ch15-greedy-and-sampling.md)*
