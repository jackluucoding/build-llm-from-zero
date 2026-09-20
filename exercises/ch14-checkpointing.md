# Chapter 14: Checkpointing: Exercises

Read the chapter online: https://jackluu.io/book/section-4-training/ch14-checkpointing/

## Try It

You can inspect the checkpoint directly by writing a small script to load the file.

```python
# ...
checkpoint_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "checkpoints", "model.pt"
)
# ...
checkpoint = torch.load(
    checkpoint_path, map_location="cpu", weights_only=False
)
print("Keys in checkpoint:", list(checkpoint.keys()))
print("Training step:", checkpoint.get("step"))
```

Lines 9 and 10 print the keys and the training step from the loaded checkpoint dictionary.

```console
$ python src/examples/ch14_inspect_checkpoint.py
Keys in checkpoint: ['model_state', 'gpt_cfg', 'step', 'val_loss']
Training step: 3000

```

## Check Your Understanding

1. Why do we need to save the model's configuration in the checkpoint along with the weights?
2. What happens if you forget to call `model.eval()` before generating text?
3. What PyTorch function do we use to apply the saved weights to the newly built model?
