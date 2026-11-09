# Chapter 17: Putting It All Together: Exercises

Read the chapter online: https://jackluu.io/book/section-5-generation/ch17-putting-it-all-together/

## Try It

Test drive the final model like a deployed service would by loading the saved file `model_final.pt` and responding to a user's prompt. You are already provided with a script that accomplishes this in the repo:

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

## Assignment

**Assignment 17: Ask a new question**

**About 15 minutes. You change one word and write no new code.** Work in this chapter's Colab notebook or on your own laptop.

1. **Run it.** Run `src/examples/ch17_deploy.py` as it is. Copy the prompt and everything the model writes after it.
2. **Change one word.** Find the line `prompt = "USER: What is the news?\n"`. The prompt is the text the model continues. Change the word `news` to any other English word, in plain letters only, and leave the rest of the line as it is. Run the script again.
3. **Compare.** Does the model answer either question? What does it write instead? What do the two replies have in common?
4. **Explain it.** In two or three sentences a manager could follow, say why this model continues your text in the shape of a play, and what extra training a real assistant gets so that it answers.

**Hand in:** both prompts with their replies, your answers to step 3, and your sentences.
