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
