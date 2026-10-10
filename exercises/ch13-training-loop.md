# Chapter 13: The Training Loop: Exercises

Read the chapter online: https://jackluu.io/book/section-4-training/ch13-training-loop/

## Try It

We can see the mechanics of PyTorch's automatic gradients on a tiny scale.

```python
"""Show how PyTorch automatically calculates gradients to update a weight."""
import torch

# A single weight starting at 2.0
weight = torch.tensor([2.0], requires_grad=True)

# Our simple 'model' multiplies input by weight
x = torch.tensor([3.0])
target = torch.tensor([12.0]) # We want output to be 12

print(f"Initial weight: {weight.item():.2f}")

for step in range(3):
    # Forward pass
    output = weight * x
    loss = (output - target) ** 2

    # Backward pass (calculate the gradient)
    loss.backward()

    print(f"Step {step+1}: Output={output.item():.2f}, "
          f"Loss={loss.item():.2f}, Gradient={weight.grad.item():.2f}")

    # Update weight (move in opposite direction of gradient)
    with torch.no_grad():
        weight -= 0.05 * weight.grad
        weight.grad.zero_()

print(f"Final weight: {weight.item():.2f}")
```

Lines 19, 26, and 27 show PyTorch computing the gradient and using it to adjust the weight toward the target.

```console
$ python src/examples/ch13_autograd_demo.py
Initial weight: 2.00
Step 1: Output=6.00, Loss=36.00, Gradient=-36.00
Step 2: Output=11.40, Loss=0.36, Gradient=-3.60
Step 3: Output=11.94, Loss=0.00, Gradient=-0.36
Final weight: 4.00

```

Open `src/examples/ch13_autograd_demo.py`. Change the starting `weight` to `10.0` and watch how the gradient pulls it down instead of pushing it up, always aiming for the target output of 12.

**In Business**
Training is where the real investment happens. When your company trains its house-style assistant, it pays for the compute time required to run this loop billions of times across thousands of documents. The loss curve is your main dashboard metric, because as long as it is going down, the assistant is getting better at mimicking your corporate voice.

**Watch Out**
Never forget `optimizer.zero_grad()` before `loss.backward()`. PyTorch accumulates gradients by default (adds them up). If you don't zero them out at the start of the backward pass, your model will take steps based on a mix of the current batch and all previous batches, wandering off in the wrong direction.

## Check Your Understanding

1. What are the four main steps inside the training loop?
2. What does `loss.backward()` actually do?
3. Why is a dropping loss curve a good sign?

## Assignment

**Assignment 13: Shrink the step**

**About 15 minutes. You change one number and write no new code.** Work in this chapter's Colab notebook or on your own laptop.

1. **Run it.** Run `src/examples/ch13_autograd_demo.py` as it is. Copy the five lines it prints: the initial weight, the three steps and the final weight.
2. **Change one number.** Find the line `weight -= 0.05 * weight.grad`. The `0.05` is the learning rate: how far the weight moves on each step. Change `0.05` to `0.0003` and run the script again.
3. **Compare.** Which printed lines are the same in both runs? After three steps, how close does the output get to the target of 12 in each run? Is the loss still falling in the second run?
4. **Explain it.** In two or three sentences a manager could follow, say what a learning rate that is too small for the job costs a company that pays for every training step.

**Hand in:** both sets of five lines, your answers to step 3, and your sentences.
