import json
import os
import matplotlib.pyplot as plt
import numpy as np

def analyze_and_plot_results(results_path="results/curriculum_experiment_results.json"):
    if not os.path.exists(results_path):
        print(f"❌ Results file '{results_path}' not found.")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    colors = {
        "curriculum": "#1f77b4",       # Blue
        "shuffled": "#2ca02c",         # Green
        "anti_curriculum": "#d62728",   # Red
        "fixed_long": "#ff7f0e"        # Orange
    }

    labels = {
        "curriculum": "Curriculum (32→64→128→256)",
        "shuffled": "Shuffled Control (Random Order)",
        "anti_curriculum": "Anti-Curriculum (256→128→64→32)",
        "fixed_long": "Fixed-Long Baseline (Static 256)"
    }

    # 1. Validation BPC Curve
    ax1 = axes[0, 0]
    for key, arm in data.items():
        steps = [entry["step"] for entry in arm["logs"]]
        bpcs = [entry["val_bpc"] for entry in arm["logs"]]
        ax1.plot(steps, bpcs, label=labels[key], color=colors[key], linewidth=2.0)
    ax1.set_title("1. Validation Normalized Loss (BPC vs Steps)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Optimizer Steps")
    ax1.set_ylabel("Normalized BPC (bits/char)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # 2. Attention Matrix Entropy H_bar
    ax2 = axes[0, 1]
    for key, arm in data.items():
        steps = [entry["step"] for entry in arm["logs"]]
        ents = [entry["mean_entropy"] for entry in arm["logs"]]
        ax2.plot(steps, ents, label=labels[key], color=colors[key], linewidth=2.0)
    ax2.set_title("2. Layer Attention Matrix Entropy (H̄)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Optimizer Steps")
    ax2.set_ylabel("Normalized Attention Entropy (0.0=Sharp, 1.0=Uniform)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    # 3. Effective Representation Rank
    ax3 = axes[1, 0]
    for key, arm in data.items():
        steps = [entry["step"] for entry in arm["logs"]]
        ranks = [entry["mean_rank"] for entry in arm["logs"]]
        ax3.plot(steps, ranks, label=labels[key], color=colors[key], linewidth=2.0)
    ax3.set_title("3. Effective Representation Rank (Rank_eff)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Optimizer Steps")
    ax3.set_ylabel("Effective Rank (exp(-sum p log p))")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend()

    # 4. Context Length Dependency Gain ΔBPC at Step 2500
    ax4 = axes[1, 1]
    ctx_lens = [32, 64, 128, 256]
    x_indices = np.arange(len(ctx_lens))
    width = 0.2

    for i, (key, arm) in enumerate(data.items()):
        final_log = arm["logs"][-1]
        ctx_bpc = final_log.get("context_bpc", {})
        bpc_vals = [ctx_bpc.get(str(c), ctx_bpc.get(c, 0.0)) for c in ctx_lens]
        ax4.bar(x_indices + (i - 1.5) * width, bpc_vals, width, label=labels[key], color=colors[key])

    ax4.set_title("4. Validation BPC under Truncated Context Horizons (Step 2500)", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Evaluation Context Horizon (Tokens)")
    ax4.set_ylabel("Normalized BPC (bits/char)")
    ax4.set_xticks(x_indices)
    ax4.set_xticklabels([f"T={c}" for c in ctx_lens])
    ax4.grid(True, linestyle="--", alpha=0.6)
    ax4.legend()

    plt.tight_layout()
    output_png = "results/curriculum_dynamics_comparison.png"
    plt.savefig(output_png, dpi=300)
    print(f"📊 Visualization plot saved to '{output_png}'.")

    # Print summary table
    print("\n=======================================================")
    print("📋 SUMMARY OF EXPERIMENTAL ARMS (Step 2500)")
    print("=======================================================")
    print(f"{'Arm':<32} | {'Final BPC':<10} | {'Final Rank':<10} | {'Attn Entropy':<12}")
    print("-" * 72)
    for key, arm in data.items():
        final_log = arm["logs"][-1]
        print(f"{labels[key]:<32} | {final_log['val_bpc']:<10.4f} | {final_log['mean_rank']:<10.2f} | {final_log['mean_entropy']:<12.4f}")
    print("-" * 72)

if __name__ == "__main__":
    analyze_and_plot_results()
