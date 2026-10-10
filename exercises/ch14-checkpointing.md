# Chapter 14: Checkpointing: Exercises

Read the chapter online: https://jackluu.io/book/section-4-training/ch14-checkpointing/

## Try It

You can look inside the checkpoint with a small script that loads the file. The repository already has one:

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

## Assignment

**Assignment 14: Open the save file**

**About 15 minutes. You change one word and write no new code.** Work in this chapter's Colab notebook or on your own laptop.

1. **Run it.** Run `src/examples/ch14_inspect_checkpoint.py` as it is. Copy the two lines it prints.
2. **Change one word.** Find the line `print("Training step:", checkpoint.get("step"))`. The word inside `get(...)` names the item the script takes out of the file. Inside `get(...)`, change `"step"` to `"gpt_cfg"`, another name from the list of keys. Keep the quotation marks, and run the script again.
3. **Compare.** The label still says `Training step:`, but what follows it now? Which of its numbers match the line `Model config : 4 layers, 128 embd, 4 heads` from this chapter's run? Which other numbers do you recognize from earlier chapters?
4. **Explain it.** In two or three sentences a manager could follow, say why the save file must hold this configuration and not only the learned weights.

**Hand in:** both outputs, your answers to step 3, and your sentences.
