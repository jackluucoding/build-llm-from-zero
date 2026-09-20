# Chapter 17: Putting It All Together: Exercises

Read the chapter online: https://jackluu.io/book/section-5-generation/ch17-putting-it-all-together/

## Try It

Try testing the final model like a deployed production service. Create a script to load `model_final.pt` and respond to a user prompt.

```python
# ...
checkpoint_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "checkpoints", "model_final.pt"
)

if not os.path.exists(checkpoint_path):
    # Fall back to model.pt if model_final.pt does not exist
    checkpoint_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "checkpoints", "model.pt"
    )

checkpoint = torch.load(
    checkpoint_path, map_location="cpu", weights_only=False
)
# ...
```

Lines 6 and 8 check if the final checkpoint exists and fall back to a previous save if it doesn't.

```console
$ python src/examples/ch17_deploy.py
--- Prompt: 'USER: What is the news?\n' ---
USER: What is the news?

FROKENTER:
And with is there?

GLOUCEMIO:
All But, there come from the prainion; and every,
As do p

```

## Check Your Understanding

1. What is the difference between the model we built and GPT-2?
2. Why does the model output sometimes contain spelling or logic errors despite being fully trained?
3. What is the purpose of an end-to-end pipeline in a business setting?
