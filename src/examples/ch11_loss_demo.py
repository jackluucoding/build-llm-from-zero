"""Calculate cross-entropy loss for different confidence levels."""
import torch
import torch.nn.functional as F

# Two possible words: 'A' (index 0) or 'B' (index 1)
target = torch.tensor([0]) # The correct answer is 'A'

print("Scenario 1: Guessing blindly (50% / 50%)")
logits_blind = torch.tensor([[0.0, 0.0]])
loss_blind = F.cross_entropy(logits_blind, target)
print(f"Loss: {loss_blind.item():.4f}")

print("\nScenario 2: Confident and right (88% for 'A')")
logits_right = torch.tensor([[2.0, 0.0]])
loss_right = F.cross_entropy(logits_right, target)
print(f"Loss: {loss_right.item():.4f}")

print("\nScenario 3: Confident and wrong (88% for 'B')")
logits_wrong = torch.tensor([[0.0, 2.0]])
loss_wrong = F.cross_entropy(logits_wrong, target)
print(f"Loss: {loss_wrong.item():.4f}")
