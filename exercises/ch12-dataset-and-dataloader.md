# Chapter 12: Dataset and DataLoader: Exercises

Read the chapter online: https://jackluu.io/book/section-4-training/ch12-dataset-and-dataloader/

## Try It

We can see the sliding window in action with a tiny dataset.

```python
"""Show how a sliding window creates multiple overlapping examples."""
import torch

data = torch.tensor([10, 20, 30, 40, 50, 60, 70])
block_size = 3

print(f"Data: {data.tolist()}")
print(f"Block size: {block_size}\n")

# A simple loop to show the sliding window
for i in range(len(data) - block_size):
    x = data[i : i + block_size]
    y = data[i + 1 : i + block_size + 1]
    print(f"Example {i+1}:")
    print(f"  Input  : {x.tolist()}")
    print(f"  Target : {y.tolist()}")
```

Lines 12 and 13 slice the array to create overlapping sequences for the input and target.

```console
$ python src/examples/ch12_batching_demo.py
Data: [10, 20, 30, 40, 50, 60, 70]
Block size: 3

Example 1:
  Input  : [10, 20, 30]
  Target : [20, 30, 40]
Example 2:
  Input  : [20, 30, 40]
  Target : [30, 40, 50]
Example 3:
  Input  : [30, 40, 50]
  Target : [40, 50, 60]
Example 4:
  Input  : [40, 50, 60]
  Target : [50, 60, 70]

```

Open `src/examples/ch12_batching_demo.py`. Change `block_size` to 4, run it again, and notice how the number of available examples decreases.

**In Business**
Imagine wanting your company's house-style assistant to learn from a library of documents. Say we had 100,000 company documents that we wanted to teach our assistant on. Obviously we're not teaching it one word at a time. When learning, you can teach your assistant in groups of 32 at a time. These could be emails or articles or even whole books. That is what makes it so efficient to teach your assistant on these large data sets.

**Watch Out**
Watch your dataset's `__len__`. If you have 100 characters and your `block_size` is 10, you only have 90 starting spots. Why? Because we need 11 characters to create a single example, 10 characters to input and 1 more to predict. That's why there's `len(self.data) - self.block_size` in the code.

## Check Your Understanding

1. If you have a sequence of 1000 tokens and a block size of 100, how many examples can a sliding window extract?
2. Why is batching important for training speed?
3. Why do we shuffle the training data?

## Assignment

**Assignment 12: Follow one number**

**About 15 minutes. You change one number and write no new code.** Work in this chapter's Colab notebook or on your own laptop.

1. **Run it.** Run `src/examples/ch12_batching_demo.py` as it is. Copy the four examples it prints, each with its Input and its Target.
2. **Change one number.** Find the line `data = torch.tensor([10, 20, 30, 40, 50, 60, 70])`. These seven numbers stand in for a text of seven characters. Change the `40` in the middle to any whole number that is not already in the list, and run the script again.
3. **Compare.** In how many examples is your number part of the Input? In how many is it part of the Target? How many examples does the script print now, and did that count change?
4. **Explain it.** In two or three sentences a manager could follow, say how a sliding window turns one text into many training examples, and why one character can appear in several of them.

**Hand in:** both sets of four examples, your answers to step 3, and your sentences.
