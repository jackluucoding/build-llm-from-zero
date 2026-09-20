"""Plot the effect of temperature on a probability distribution."""
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Example scores (logits)
    logits = np.array([2.5, 1.8, 0.5, -1.0, -2.5])
    labels = ["A", "B", "C", "D", "E"]
    
    temps = [0.5, 1.0, 2.0]
    
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 3.5), dpi=300)
    fig.patch.set_facecolor('white')
    
    for ax, T in zip(axes, temps):
        # Apply temperature
        scaled = logits / T
        # Softmax
        exp_L = np.exp(scaled - np.max(scaled))
        probs = exp_L / np.sum(exp_L)
        
        ax.bar(labels, probs, color="#00695C")
        ax.set_ylim(0, 1.0)
        ax.set_title(f"T = {T}")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig("ch16_temperature.png")
    print("Saved chart: ch16_temperature.png")

if __name__ == "__main__":
    main()
