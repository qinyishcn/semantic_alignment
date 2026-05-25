"""
Unified visualization: all experiments on one chart for comparison.
Uses English labels to avoid CJK font issues on Windows.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
RESULTS_DIR2 = os.path.join(os.path.dirname(__file__), "results_dim_mismatch")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "results", "plots")


def load_all_results():
    with open(os.path.join(RESULTS_DIR, "training_log.json"), "r") as f:
        exp1 = json.load(f)
    with open(os.path.join(RESULTS_DIR2, "dim_mismatch_log.json"), "r") as f:
        exp2 = json.load(f)
    return exp1, exp2


def plot_unified_comparison(exp1, exp2):
    """Single grouped bar chart comparing all configurations."""
    fig, ax = plt.subplots(figsize=(18, 9))

    # ── Data ───────────────────────────────────────────────────────────
    acc_a1 = exp1["results"]["model_a"]["final_accuracy"]
    acc_b1 = exp1["results"]["model_b"]["final_accuracy"]
    acc_ab = exp1["results"]["cross_models"]["A_encoder_B_decoder"]["accuracy"]
    acc_ba = exp1["results"]["cross_models"]["B_encoder_A_decoder"]["accuracy"]
    acc_a2 = exp2["model_a"]["accuracy"]
    acc_c2 = exp2["model_c"]["accuracy"]
    acc_pad = exp2["fix_methods"]["zero_pad_A_enc_32_to_64_C_dec"]["accuracy"]
    acc_trunc = exp2["fix_methods"]["truncate_C_enc_64_to_32_A_dec"]["accuracy"]

    # ── Labels & values ────────────────────────────────────────────────
    labels = [
        "Model A\n(seed=42, dim=32)",
        "Model B\n(seed=123, dim=32)",
        "A_enc + B_dec\n(same dim, cross)",
        "B_enc + A_dec\n(same dim, cross)",
        "",
        "Model A\n(seed=42, dim=32)",
        "Model C\n(seed=999, dim=64)",
        "A_enc(32)+\nC_dec(64)\ndirect concat",
        "C_enc(64)+\nA_dec(32)\ndirect concat",
        "",
        "A_enc(32) -> pad\n-> C_dec(64)",
        "C_enc(64) -> trunc\n-> A_dec(32)",
    ]

    accuracies = [
        acc_a1, acc_b1, acc_ab, acc_ba, 0,
        acc_a2, acc_c2, 0, 0, 0,
        acc_pad, acc_trunc,
    ]

    colors = [
        "#4CAF50", "#4CAF50", "#F44336", "#F44336", "white",
        "#2196F3", "#2196F3", "#9E9E9E", "#9E9E9E", "white",
        "#FF9800", "#9C27B0",
    ]

    error_indices = [7, 8]

    # ── Plot ───────────────────────────────────────────────────────────
    x = np.arange(len(labels))
    bar_width = 0.62

    bars = ax.bar(x, accuracies, bar_width, color=colors,
                  edgecolor="black", linewidth=0.8, alpha=0.88)

    # Annotate values
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        if i in error_indices:
            ax.text(bar.get_x() + bar.get_width() / 2, 4,
                    "ERROR\nShape Mismatch", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color="#C62828",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFCDD2",
                              edgecolor="#C62828", alpha=0.9))
        elif acc > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"{acc:.1f}%", ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold")

    # Reference lines
    ax.axhline(y=95, color="green", linestyle="--", alpha=0.3, linewidth=1)
    ax.text(len(labels) - 0.3, 95.5, "95% baseline", fontsize=8, color="green", alpha=0.5, ha="right")

    # Group separators
    for sep_x in [4.5, 9.5]:
        ax.axvline(x=sep_x, color="gray", linestyle=":", alpha=0.4, linewidth=1.2)

    # Group labels at top
    group_info = [
        (1.5, "Exp 1: Same-Dim Cross", "#4CAF50"),
        (5.5, "Exp 2: Orig (dim=32/64)", "#2196F3"),
        (7.5, "Exp 2: Direct Concat", "#9E9E9E"),
        (10.5, "Exp 2: Fix Methods", "#FF9800"),
    ]
    for gx, gtext, gcolor in group_info:
        ax.text(gx, 106, gtext, ha="center", va="bottom",
                fontsize=10.5, fontweight="bold", color=gcolor,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=gcolor, alpha=0.85, linewidth=1.5))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5, ha="center")
    ax.set_ylabel("Test Accuracy (%)", fontsize=13)
    ax.set_title("Semantic Alignment — Unified Comparison Across All Experiments",
                 fontsize=15, fontweight="bold", pad=25)
    ax.set_ylim(0, 115)
    ax.grid(True, axis="y", alpha=0.3)

    # Legend
    legend_patches = [
        mpatches.Patch(facecolor="#4CAF50", edgecolor="black", label="Original (same dim)"),
        mpatches.Patch(facecolor="#F44336", edgecolor="black", label="Cross-combine (same dim)"),
        mpatches.Patch(facecolor="#2196F3", edgecolor="black", label="Original (dim=32/64)"),
        mpatches.Patch(facecolor="#9E9E9E", edgecolor="black", label="Direct concat (ERROR)"),
        mpatches.Patch(facecolor="#FF9800", edgecolor="black", label="Zero-padding fix"),
        mpatches.Patch(facecolor="#9C27B0", edgecolor="black", label="Truncation fix"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9.5, framealpha=0.92)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "unified_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved unified_comparison.png")


def plot_summary_table(exp1, exp2):
    """A clean summary table as an image."""
    fig, ax = plt.subplots(figsize=(16, 6.5))
    ax.axis("off")

    col_labels = ["Config", "Accuracy", "Type", "Note"]
    cell_data = [
        ["Model A (dim=32, seed=42)", f'{exp1["results"]["model_a"]["final_accuracy"]:.2f}%', "Original", "Exp 1"],
        ["Model B (dim=32, seed=123)", f'{exp1["results"]["model_b"]["final_accuracy"]:.2f}%', "Original", "Exp 1"],
        ["A_enc + B_dec (same dim)", f'{exp1["results"]["cross_models"]["A_encoder_B_decoder"]["accuracy"]:.2f}%', "Cross", "Semantic mismatch"],
        ["B_enc + A_dec (same dim)", f'{exp1["results"]["cross_models"]["B_encoder_A_decoder"]["accuracy"]:.2f}%', "Cross", "Semantic mismatch"],
        ["Model A (dim=32, seed=42)", f'{exp2["model_a"]["accuracy"]:.2f}%', "Original", "Exp 2"],
        ["Model C (dim=64, seed=999)", f'{exp2["model_c"]["accuracy"]:.2f}%', "Original", "Exp 2"],
        ["A_enc(32) + C_dec(64)", "ERROR", "Direct", "Shape mismatch"],
        ["C_enc(64) + A_dec(32)", "ERROR", "Direct", "Shape mismatch"],
        ["A_enc(32->64 pad) + C_dec", f'{exp2["fix_methods"]["zero_pad_A_enc_32_to_64_C_dec"]["accuracy"]:.2f}%', "Pad fix", "Semantic+dim mismatch"],
        ["C_enc(64->32 trunc) + A_dec", f'{exp2["fix_methods"]["truncate_C_enc_64_to_32_A_dec"]["accuracy"]:.2f}%', "Trunc fix", "Semantic+dim mismatch"],
    ]

    cell_colors = []
    color_map = {"Original": "#E8F5E9", "Cross": "#FFEBEE", "Direct": "#F5F5F5", "Pad fix": "#FFF3E0", "Trunc fix": "#F3E5F5"}
    for row in cell_data:
        c = color_map.get(row[2], "white")
        cell_colors.append([c] * 4)

    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellColours=cell_colors,
        colColours=["#E3F2FD"] * 4,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.9)

    for j in range(4):
        table[0, j].set_text_props(fontweight="bold")

    ax.set_title("Semantic Alignment Experiment — All Results Summary",
                 fontsize=15, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "summary_table.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved summary_table.png")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    exp1, exp2 = load_all_results()
    plot_unified_comparison(exp1, exp2)
    plot_summary_table(exp1, exp2)
    print("\nDone!")


if __name__ == "__main__":
    main()
