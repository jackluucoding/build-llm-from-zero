"""
Translate a cross-entropy loss into the confidence it stands for.
This file belongs to Chapter 13.
Run: python src/examples/ch13_loss_meaning.py
"""
import math

VOCAB = 65

# Losses from the training run in this chapter
for name, loss in [("random guessing", math.log(VOCAB)),
                   ("step 1", 4.3280),
                   ("step 300", 2.7016),
                   ("step 3000", 1.7357)]:
    # Cross-entropy is the negative log of the probability given to the right answer
    prob = math.exp(-loss)
    print(f"{name:<16} loss {loss:.2f} -> {prob:6.1%} on the right character")
