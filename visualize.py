"""
Visualization script for semantic alignment experiment results.
Generates plots comparing model performance and latent space properties.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")


def load_results():
    with open(os.path.join(RESULTS_DIR, "training_log.json"), "r") as f:
        return json.load(f)


def plot_training_curves(results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    hist_a = results["results"]["model_a"]["training_history"]
    hist_b = results["results"]["model_b"]["training_history"]
    epochs = range(1, len(hist_a["train_loss"]) + 1)

    # Training loss
    axes[0].plot(epochs, hist_a["train_loss"], "b-o", label="Model A", markersize=4)
    axes[0].plot(epochs, hist_b["train_loss"], "r-s", label="Model B", markersize=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Loss")
    axes[0].set_title("Training Loss Over Epochs")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Test accuracy
    axes[1].plot(epochs, hist_a["test_accuracy"], "b-o", label="Model A", markersize=4)
    axes[1].plot(epochs, hist_b["test_accuracy"], "r-s", label="Model B", markersize=4)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Test Accuracy Over Epochs")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Test loss
    axes[2].plot(epochs, hist_a["test_loss"], "b-o", label="Model A", markersize=4)
    axes[2].plot(epochs, hist_b["test_loss"], "r-s", label="Model B", markersize=4)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Test Loss")
    axes[2].set_title("Test Loss Over Epochs")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "training_curves.png"), dpi=150)
    plt.close()
    print("Saved training_curves.png")


def plot_accuracy_comparison(results):
    acc_a = results["results"]["model_a"]["final_accuracy"]
    acc_b = results["results"]["model_b"]["final_accuracy"]
    acc_ab = results["results"]["cross_models"]["A_encoder_B_decoder"]["accuracy"]
    acc_ba = results["results"]["cross_models"]["B_encoder_A_decoder"]["accuracy"]

    models = ["Model A\n(original)", "Model B\n(original)", "A_enc +\nB_dec", "B_enc +\nA_dec"]
    accuracies = [acc_a, acc_b, acc_ab, acc_ba]
    colors = ["#2196F3", "#F44336", "#FF9800", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, accuracies, color=colors, edgecolor="black", linewidth=0.8)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{acc:.2f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Model Accuracy Comparison\n(Original vs Cross-Combination)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.3)

    # Add horizontal line for reference
    avg_original = (acc_a + acc_b) / 2
    ax.axhline(y=avg_original, color="gray", linestyle="--", alpha=0.5, label=f"Avg Original: {avg_original:.1f}%")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "accuracy_comparison.png"), dpi=150)
    plt.close()
    print("Saved accuracy_comparison.png")


def plot_accuracy_drop(results):
    drop_ab = results["conclusion"]["accuracy_drop_A_enc_B_dec"]
    drop_ba = results["conclusion"]["accuracy_drop_B_enc_A_dec"]

    fig, ax = plt.subplots(figsize=(8, 5))
    models = ["A_enc + B_dec", "B_enc + A_dec"]
    drops = [drop_ab, drop_ba]
    colors = ["#FF9800", "#9C27B0"]

    bars = ax.bar(models, drops, color=colors, edgecolor="black", linewidth=0.8, width=0.5)

    for bar, drop in zip(bars, drops):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"-{drop:.2f}%",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color="red",
        )

    ax.set_ylabel("Accuracy Drop (%)", fontsize=12)
    ax.set_title("Accuracy Degradation from Mismatched Encoder-Decoder", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(drops) + 10)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "accuracy_drop.png"), dpi=150)
    plt.close()
    print("Saved accuracy_drop.png")


def plot_latent_analysis(results):
    latent_a = results["results"]["latent_analysis"]["model_a"]
    latent_b = results["results"]["latent_analysis"]["model_b"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Mean activation comparison
    mean_a = np.array(latent_a["mean"])
    mean_b = np.array(latent_b["mean"])
    x = np.arange(len(mean_a))

    axes[0].bar(x - 0.2, mean_a, 0.4, label="Model A", color="#2196F3", alpha=0.7)
    axes[0].bar(x + 0.2, mean_b, 0.4, label="Model B", color="#F44336", alpha=0.7)
    axes[0].set_xlabel("Latent Dimension Index")
    axes[0].set_ylabel("Mean Activation")
    axes[0].set_title("Latent Space Mean Activation Comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Standard deviation comparison
    std_a = np.array(latent_a["std"])
    std_b = np.array(latent_b["std"])

    axes[1].bar(x - 0.2, std_a, 0.4, label="Model A", color="#2196F3", alpha=0.7)
    axes[1].bar(x + 0.2, std_b, 0.4, label="Model B", color="#F44336", alpha=0.7)
    axes[1].set_xlabel("Latent Dimension Index")
    axes[1].set_ylabel("Standard Deviation")
    axes[1].set_title("Latent Space Std Dev Comparison")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "latent_analysis.png"), dpi=150)
    plt.close()
    print("Saved latent_analysis.png")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    results = load_results()

    plot_training_curves(results)
    plot_accuracy_comparison(results)
    plot_accuracy_drop(results)
    plot_latent_analysis(results)

    print("\nAll plots saved to:", PLOTS_DIR)


if __name__ == "__main__":
    main()
