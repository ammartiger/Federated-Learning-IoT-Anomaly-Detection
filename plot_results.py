import os
import matplotlib.pyplot as plt
import numpy as np

# Styling configuration to match IEEE IoT Journal publication standards
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "lines.linewidth": 1.5,
    "grid.alpha": 0.4,
    "grid.linestyle": "--"
})

WINDOW_SIZES = [1, 5, 10, 15, 20, 30, 40]
METRICS = ["Accuracy", "F1-Score", "Precision", "Recall"]

def plot_gru_comparison(model_name: str, fl_metrics: dict, non_fl_metrics: dict, save_path: str):
    """
    Replicates Figures 5, 6, 7, 8:
    Bar charts of Accuracy, F1-Score, Precision, Recall across 7 window sizes for FL vs Non-FL.
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    fig.suptitle(f"{model_name}: Model evaluation results of proposed FL approach and non-FL approach", y=1.02, fontweight="bold")

    w_sizes = [w for w in WINDOW_SIZES if str(w) in fl_metrics]
    if not w_sizes:
        w_sizes = sorted([int(k) for k in fl_metrics.keys()])
    bar_width = 0.38
    x_indices = np.arange(len(w_sizes))

    metric_keys = {
        "Accuracy": "accuracy",
        "F1-Score": "f1",
        "Precision": "precision",
        "Recall": "recall"
    }

    for ax_idx, metric in enumerate(METRICS):
        ax = axes[ax_idx]
        m_key = metric_keys[metric]
        
        fl_vals = [fl_metrics[str(w)][m_key] * 100 for w in w_sizes]
        non_fl_vals = [non_fl_metrics[str(w)][m_key] * 100 for w in w_sizes]

        bars1 = ax.bar(x_indices - bar_width/2, fl_vals, bar_width, label="FL", color="#2b5c8f", edgecolor="black", linewidth=0.6)
        bars2 = ax.bar(x_indices + bar_width/2, non_fl_vals, bar_width, label="Non-FL", color="#d97724", edgecolor="black", linewidth=0.6)

        # Value annotations
        for bar in bars1:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7, rotation=90)
        for bar in bars2:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7, rotation=90)

        ax.set_title(metric, fontweight="bold")
        ax.set_xlabel("Window Size")
        ax.set_xticks(x_indices)
        ax.set_xticklabels(w_sizes)
        ax.set_ylim(0, 115)
        if ax_idx == 0:
            ax.set_ylabel("Percentage (%)")
        if ax_idx == 3:
            ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {model_name} plot to {save_path}")


def plot_training_times(training_times_dict: dict, save_path: str):
    """
    Replicates Figure 9:
    Training time comparison across window sizes for FL vs Non-FL across GRU models.
    """
    models = [m for m in ["GRU-1", "GRU-2", "GRU-3", "GRU-4"] if m in training_times_dict]
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(4.5 * len(models), 4.5), sharey=True)
    if len(models) == 1:
        axes = [axes]
    fig.suptitle("Fig. 9. Training Time: For different window sizes for proposed FL approach and non-FL approach", y=1.02, fontweight="bold")

    for idx, model_name in enumerate(models):
        ax = axes[idx]
        w_sizes = [w for w in WINDOW_SIZES if str(w) in training_times_dict[model_name]["FL"]]
        if not w_sizes:
            w_sizes = sorted([int(k) for k in training_times_dict[model_name]["FL"].keys()])
        bar_width = 0.38
        x_indices = np.arange(len(w_sizes))

        fl_times = [training_times_dict[model_name]["FL"][str(w)] for w in w_sizes]
        non_fl_times = [training_times_dict[model_name]["Non-FL"][str(w)] for w in w_sizes]

        bars1 = ax.bar(x_indices - bar_width/2, fl_times, bar_width, label="FL", color="#2b5c8f", edgecolor="black", linewidth=0.6)
        bars2 = ax.bar(x_indices + bar_width/2, non_fl_times, bar_width, label="Non-FL", color="#d97724", edgecolor="black", linewidth=0.6)

        for bar in bars1:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7)
        for bar in bars2:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7)

        ax.set_title(f"Model {idx+1} ({model_name})", fontweight="bold")
        ax.set_xlabel("Window Size")
        ax.set_xticks(x_indices)
        ax.set_xticklabels(w_sizes)
        if idx == 0:
            ax.set_ylabel("Training Time (seconds)")
        if idx == len(models) - 1:
            ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Training Time plot to {save_path}")


def plot_ensembler_results(ensembler_metrics: dict, save_path: str):
    """
    Replicates Figure 10:
    Model evaluation results of Random Forest Ensembler across the 7 window sizes.
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    fig.suptitle("Fig. 10. Model Evaluation Results: Ensembler on 7 Global Models", y=1.02, fontweight="bold")

    w_sizes = [w for w in WINDOW_SIZES if str(w) in ensembler_metrics]
    if not w_sizes:
        w_sizes = sorted([int(k) for k in ensembler_metrics.keys()])
    bar_width = 0.55
    x_indices = np.arange(len(w_sizes))

    metric_keys = {
        "Accuracy": "accuracy",
        "F1-Score": "f1",
        "Precision": "precision",
        "Recall": "recall"
    }

    colors = ["#204060", "#2d6898", "#418ab3", "#6baed6"]

    for ax_idx, metric in enumerate(METRICS):
        ax = axes[ax_idx]
        m_key = metric_keys[metric]
        vals = [ensembler_metrics[str(w)][m_key] * 100 for w in w_sizes]

        bars = ax.bar(x_indices, vals, bar_width, color=colors[ax_idx], edgecolor="black", linewidth=0.6)

        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, rotation=90)

        ax.set_title(metric, fontweight="bold")
        ax.set_xlabel("Window Size Subset")
        ax.set_xticks(x_indices)
        ax.set_xticklabels([f"W<={w}" for w in w_sizes])
        ax.set_ylim(0, 115)
        if ax_idx == 0:
            ax.set_ylabel("Percentage (%)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Ensembler plot to {save_path}")


def plot_overall_average(fl_avg_acc: float, non_fl_avg_acc: float, ens_avg_acc: float, save_path: str):
    """
    Replicates Figure 11:
    Model evaluation results: Average of all four models (FL vs Non-FL vs Ensembler).
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle("Fig. 11. Model Evaluation Results: Average of All Four Models", fontweight="bold")

    categories = ["Non-FL", "FL", "Ensembler"]
    values = [non_fl_avg_acc * 100, fl_avg_acc * 100, ens_avg_acc * 100]
    colors = ["#d97724", "#2b5c8f", "#2ca02c"]

    bars = ax.bar(categories, values, width=0.45, color=colors, edgecolor="black", linewidth=0.8)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Average Accuracy (%)")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Overall Average plot to {save_path}")
