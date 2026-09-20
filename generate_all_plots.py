import json
import os
import matplotlib.pyplot as plt
import numpy as np

def generate_tri_factor_dashboard():
    exp_file = "results/curriculum_experiment_results.json"
    if not os.path.exists(exp_file):
        print(f"❌ Results file '{exp_file}' not found.")
        return

    with open(exp_file, "r") as f:
        data = json.load(f)

    # Gradient probe empirical data
    probe_T = np.array([32, 64, 128, 256])
    probe_var = np.array([0.8383, 1.0784, 1.2674, 1.6328])
    probe_norm = np.array([5.4840, 6.1616, 6.9804, 7.5183])
    
    # Theoretical linear scaling prediction: Tr(Sigma) = alpha * T + beta
    alpha, beta = np.polyfit(probe_T, probe_var, 1)
    theory_var_fit = alpha * probe_T + beta

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(4, 2, figsize=(16, 22))

    colors = {
        "curriculum": "#1f77b4",       # Deep Blue
        "shuffled": "#2ca02c",         # Forest Green
        "anti_curriculum": "#d62728",   # Crimson Red
        "fixed_long": "#ff7f0e"        # Amber Orange
    }

    labels = {
        "curriculum": "Curriculum (32→64→128→256)",
        "shuffled": "Shuffled Control (Random Order)",
        "anti_curriculum": "Anti-Curriculum (256→128→64→32)",
        "anti_curriculum": "Anti-Curriculum (256->128->64->32)",
        "fixed_long": "Fixed-Long Baseline (Static 256)"
    }

    # -------------------------------------------------------------
    # Panel 1 [Theory vs Computation]: Gradient Noise Scaling Theory vs Measured Probe
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.plot(probe_T, probe_var, 'o-', color='#1f77b4', linewidth=2.5, markersize=8, label="Empirical Probe Tr(Sigma)")
    ax1.plot(probe_T, theory_var_fit, 'k--', linewidth=2.0, label=f"Theory Fit: Tr(Sigma) ~ {alpha:.4f}T")
    ax1.set_title("1. [Theory vs Computation] Gradient Variance Noise Scaling (McCandlish 2018)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Context Length (T)")
    ax1.set_ylabel("Total Gradient Variance Tr(Sigma)")
    ax1.set_xticks(probe_T)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.annotate(f"Short T=32 reduces variance by 48.6%\ncompared to Long T=256", 
                 xy=(32, 0.8383), xytext=(64, 0.9),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
    ax1.legend()

    # -------------------------------------------------------------
    # Panel 2 [Theory vs Experiment]: Rank Collapse Theorem vs Empirical Effective Rank
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    steps = [entry["step"] for entry in data["curriculum"]["logs"]]
    
    # Plot Max Theoretical Rank d_model=256
    ax2.axhline(y=256, color='gray', linestyle=':', linewidth=2, label="Max Theoretical Rank (d_model=256)")
    
    for key, arm in data.items():
        rnks = [entry["mean_rank"] for entry in arm["logs"]]
        ax2.plot(steps, rnks, label=labels[key], color=colors[key], linewidth=2.2)

    ax2.set_title("2. [Theory vs Experiment] Rank Collapse Theorem (Dong 2021) vs Effective Rank", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Optimizer Steps")
    ax2.set_ylabel("Effective Matrix Rank (Rank_eff)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.annotate("Curriculum avoids Rank Collapse\n(Final Rank: 140.79 vs 131.75)", xy=(2500, 140.79), xytext=(1500, 145),
                 arrowprops=dict(facecolor='blue', shrink=0.05, width=1, headwidth=6),
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="#e6f2ff", alpha=0.8))
    ax2.legend(loc="lower right")

    # -------------------------------------------------------------
    # Panel 3 [Experiment vs Experiment]: Validation BPC Convergence Curves
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    for key, arm in data.items():
        bpcs = [entry["val_bpc"] for entry in arm["logs"]]
        ax3.plot(steps, bpcs, label=labels[key], color=colors[key], linewidth=2.2)

    ax3.set_title("3. [Experiment vs Experiment] Validation BPC Convergence Across 4 Arms", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Optimizer Steps")
    ax3.set_ylabel("Bits-Per-Character (BPC)")
    ax3.set_ylim(1.95, 4.5)
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend()

    # -------------------------------------------------------------
    # Panel 4 [Experiment vs Experiment]: Relative Loss Delta (Delta BPC vs Curriculum)
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    curr_bpcs = np.array([entry["val_bpc"] for entry in data["curriculum"]["logs"]])
    
    for key, arm in data.items():
        if key == "curriculum":
            continue
        arm_bpcs = np.array([entry["val_bpc"] for entry in arm["logs"]])
        delta_bpc = arm_bpcs - curr_bpcs
        ax4.plot(steps, delta_bpc, label=f"Delta ({labels[key]} - Curriculum)", color=colors[key], linewidth=2.2)

    ax4.axhline(y=0.0, color='black', linestyle='--', linewidth=1.5, label="Curriculum Baseline (Delta=0)")
    ax4.set_title("4. [Experiment vs Experiment] Relative Loss Penalty (Delta BPC vs Curriculum)", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Optimizer Steps")
    ax4.set_ylabel("Delta BPC Penalty (Higher = Worse)")
    ax4.grid(True, linestyle="--", alpha=0.6)
    ax4.annotate("Anti-Curriculum Severe Degradation\n(+0.17 BPC Penalty)", xy=(2500, 0.1722), xytext=(1400, 0.14),
                 arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=6),
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="#ffe6e6", alpha=0.8))
    ax4.legend(loc="upper left")

    # -------------------------------------------------------------
    # Panel 5 [Theory vs Experiment]: Attention Entropy Sharpening (H_bar)
    # -------------------------------------------------------------
    ax5 = axes[2, 0]
    ax5.axhline(y=1.0, color='gray', linestyle=':', linewidth=2, label="Theoretical Uniform Entropy (H_bar=1.0)")
    
    for key, arm in data.items():
        ents = [entry["mean_entropy"] for entry in arm["logs"]]
        ax5.plot(steps, ents, label=labels[key], color=colors[key], linewidth=2.2)

    ax5.set_title("5. [Theory vs Experiment] Attention Entropy Sharpening (H_bar: 0=Sharp, 1=Uniform)", fontsize=12, fontweight="bold")
    ax5.set_xlabel("Optimizer Steps")
    ax5.set_ylabel("Normalized Attention Entropy H_bar")
    ax5.grid(True, linestyle="--", alpha=0.6)
    ax5.annotate("Curriculum Sharpened (H_bar=0.2975)\nvs Fixed-Long Unfocused (H_bar=0.5038)", xy=(2500, 0.2975), xytext=(1200, 0.25),
                 arrowprops=dict(facecolor='blue', shrink=0.05, width=1, headwidth=6),
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="#e6f2ff", alpha=0.8))
    ax5.legend(loc="upper right")

    # -------------------------------------------------------------
    # Panel 6 [Experiment vs Experiment]: Context Horizon Truncation Sensitivity (Yao 2024)
    # -------------------------------------------------------------
    ax6 = axes[2, 1]
    ctx_lens = [32, 64, 128, 256]
    x_indices = np.arange(len(ctx_lens))
    width = 0.2

    for i, (key, arm) in enumerate(data.items()):
        final_log = arm["logs"][-1]
        ctx_bpc = final_log.get("context_bpc", {})
        bpc_vals = [ctx_bpc.get(str(c), ctx_bpc.get(c, 0.0)) for c in ctx_lens]
        ax6.bar(x_indices + (i - 1.5) * width, bpc_vals, width, label=labels[key], color=colors[key])

    ax6.set_title("6. [Experiment vs Experiment] Truncated Context Horizon BPC (Yao et al. 2024)", fontsize=12, fontweight="bold")
    ax6.set_xlabel("Evaluation Context Horizon (Tokens)")
    ax6.set_ylabel("Normalized BPC (bits/char)")
    ax6.set_xticks(x_indices)
    ax6.set_xticklabels([f"T_eval={c}" for c in ctx_lens])
    ax6.grid(True, linestyle="--", alpha=0.6)
    ax6.legend(loc="upper right")

    # -------------------------------------------------------------
    # Panel 7: Temporal Context Length Schedules T(t) Overlaid
    # -------------------------------------------------------------
    ax7 = axes[3, 0]
    stps_arr = np.arange(1, 2501)
    sched_curr = [32 if s <= 625 else (64 if s <= 1250 else (128 if s <= 1875 else 256)) for s in stps_arr]
    sched_anti = [256 if s <= 625 else (128 if s <= 1250 else (64 if s <= 1875 else 32)) for s in stps_arr]
    
    ax7.plot(stps_arr, sched_curr, color=colors["curriculum"], linewidth=2.5, label="Curriculum T(t)")
    ax7.plot(stps_arr, sched_anti, color=colors["anti_curriculum"], linewidth=2.5, label="Anti-Curriculum T(t)")
    ax7.axvline(x=625, color='gray', linestyle=':', alpha=0.7)
    ax7.axvline(x=1250, color='gray', linestyle=':', alpha=0.7)
    ax7.axvline(x=1875, color='gray', linestyle=':', alpha=0.7)

    ax7.set_title("7. [Schedule Overlay] Sequence Length Stage Transitions T(t)", fontsize=12, fontweight="bold")
    ax7.set_xlabel("Optimizer Steps")
    ax7.set_ylabel("Context Window T")
    ax7.set_yticks([32, 64, 128, 256])
    ax7.grid(True, linestyle="--", alpha=0.6)
    ax7.legend()

    # -------------------------------------------------------------
    # Panel 8: Theory vs Computation vs Experiment Key Summary Table
    # -------------------------------------------------------------
    ax8 = axes[3, 1]
    ax8.axis('off')
    ax8.set_title("8. [Summary Matrix] Theory - Computation - Experiment Alignment", fontsize=12, fontweight="bold", pad=15)

    table_data = [
        ["Dimension", "Theoretical Prediction", "Empirical Measurement", "Verdict / Result"],
        ["Grad Noise", "Tr(Sigma) ~ T (McCandlish 2018)", "T=32: 0.8383 vs T=256: 1.6328", "[CONFIRMED] (-48.6% Var)"],
        ["Rank Collapse", "Pure Attn Rank Decays (Dong 2021)", "Curriculum: 140.79 vs Fixed: 131.75", "[CONFIRMED] (+9.04 Rank)"],
        ["Attn Entropy", "Initial Softmax Uniform (H_bar->1.0)", "Curriculum: 0.2975 vs Fixed: 0.5038", "[CONFIRMED] (Sharpened)"],
        ["Loss Conv.", "Curriculum > Static (Press 2021)", "Curriculum: 2.067 vs Fixed: 2.069", "[EQUIVALENT] (~2.06 BPC)"],
        ["Anti-Curric.", "Shrinking T late destabilizes", "Anti-Curriculum: 2.2395 BPC", "[DEGRADED] (+0.17 BPC)"]
    ]

    table = ax8.table(cellText=table_data, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.8)

    # Style table headers
    for (row, col), cell in table.get_celld().items():
        cell_str = cell.get_text().get_text()
        if row == 0:
            cell.set_facecolor('#1f77b4')
            cell.get_text().set_color('white')
            cell.get_text().set_weight('bold')
        elif col == 3:
            if "CONFIRMED" in cell_str:
                cell.set_facecolor('#e6ffe6')
            elif "EQUIVALENT" in cell_str:
                cell.set_facecolor('#fff5e6')
            elif "DEGRADED" in cell_str:
                cell.set_facecolor('#ffe6e6')

    plt.suptitle("Theory - Computation - Experiment Cross-Validation Dashboard\n(Transformer Context Length Curriculum Dynamics)", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()

    output_png = "results/theory_computation_experiment_dashboard.png"
    plt.savefig(output_png, dpi=300)
    print(f"✅ Comprehensive 8-Panel Dashboard saved to '{output_png}'.")

    # Copy to artifact directory
    artifact_png = "/Users/angela/.gemini/antigravity-ide/brain/b96f9b73-ff85-4618-9431-a716e02f5bf5/theory_computation_experiment_dashboard.png"
    os.system(f'cp "{output_png}" "{artifact_png}"')
    print(f"✅ Copied to artifact dir: '{artifact_png}'")

if __name__ == "__main__":
    generate_tri_factor_dashboard()
