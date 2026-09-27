"""
Figure and Process-Analysis Contract Exporter
Preregistered specification: PREREGISTRATION_AMENDMENT_001_FIGURES.md
"""

import os
import csv
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARM_COLORS = {
    "ascending": "#1f77b4",     # blue
    "descending": "#ff7f0e",    # orange
    "nonmonotonic": "#2ca02c",  # green
}

HORIZON_COLORS = {
    32: "#dadaeb",   # lightest purple
    64: "#bcbddc",   # medium-light purple
    128: "#9e9ac8",  # medium-dark purple
    256: "#6a51a3",  # darkest purple
}


def export_tidy_csvs(data, active_seeds, checkpoints, pre_step, post_step, figures_dir):
    """
    Export the 5 tidy CSV tables required by Section 2 of Amendment 001.
    """
    data_dir = os.path.join(figures_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. process_horizon_bpc.csv
    f1_path = os.path.join(data_dir, "process_horizon_bpc.csv")
    with open(f1_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed", "arm", "global_step", "recovery_step",
            "current_training_context", "evaluation_horizon", "mean_bpc"
        ])
        for seed in active_seeds:
            for arm in ["ascending", "descending", "nonmonotonic"]:
                for log in data[seed][arm]["logs"]:
                    g_step = log["step"]
                    r_step = max(0, g_step - pre_step)
                    train_T = log.get("T", 256)
                    h_bpc = log.get("horizon_bpc", {})
                    for h, val in h_bpc.items():
                        writer.writerow([seed, arm, g_step, r_step, train_T, h, val])
                        
    # 2. anchor_horizon_bpc.csv
    f2_path = os.path.join(data_dir, "anchor_horizon_bpc.csv")
    with open(f2_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed", "arm", "global_step", "recovery_step",
            "evaluation_horizon", "mean_bpc"
        ])
        for seed in active_seeds:
            for arm in ["ascending", "descending", "nonmonotonic"]:
                for a_log in data[seed][arm].get("anchor_logs", []):
                    g_step = a_log["step"]
                    r_step = max(0, g_step - pre_step)
                    h_bpc = a_log.get("horizon_bpc", {})
                    for h, val in h_bpc.items():
                        writer.writerow([seed, arm, g_step, r_step, h, val])
                        
    # 3. anchor_sequence_bpc.csv
    f3_path = os.path.join(data_dir, "anchor_sequence_bpc.csv")
    with open(f3_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed", "arm", "global_step", "recovery_step",
            "evaluation_horizon", "validation_sequence", "mean_bpc"
        ])
        for seed in active_seeds:
            for arm in ["ascending", "descending", "nonmonotonic"]:
                for a_log in data[seed][arm].get("anchor_logs", []):
                    g_step = a_log["step"]
                    r_step = max(0, g_step - pre_step)
                    h_res = a_log.get("horizon_results", {})
                    for h, res in h_res.items():
                        seq_bpc = res.get("seq_bpc", [])
                        for seq_idx, bpc in enumerate(seq_bpc):
                            writer.writerow([seed, arm, g_step, r_step, h, seq_idx, bpc])
                            
    # 4. recovery_contrasts.csv
    f4_path = os.path.join(data_dir, "recovery_contrasts.csv")
    with open(f4_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed", "global_step", "recovery_step", "evaluation_horizon", "delta_D_A_bpc"
        ])
        for seed in active_seeds:
            desc_logs = {l["step"]: l for l in data[seed]["descending"]["logs"]}
            asc_logs = {l["step"]: l for l in data[seed]["ascending"]["logs"]}
            rec_steps = [s for s in checkpoints if s >= pre_step]
            for s in rec_steps:
                r_step = s - pre_step
                for h in [32, 64, 128, 256]:
                    h_str = str(h)
                    d_val = desc_logs[s]["horizon_bpc"].get(h, desc_logs[s]["horizon_bpc"].get(h_str))
                    a_val = asc_logs[s]["horizon_bpc"].get(h, asc_logs[s]["horizon_bpc"].get(h_str))
                    if d_val is not None and a_val is not None:
                        writer.writerow([seed, s, r_step, h, d_val - a_val])
                        
    # 5. transition_shocks.csv
    f5_path = os.path.join(data_dir, "transition_shocks.csv")
    with open(f5_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed", "arm", "transition", "step_pre", "step_post",
            "evaluation_horizon", "shock_bpc"
        ])
        candidate_pairs = [
            (500, 501), (1000, 1001), (1500, 1501), (2000, 2001),
            (10, 11), (20, 21), (30, 31), (40, 41)
        ]
        valid_pairs = [p for p in candidate_pairs if p[0] in checkpoints and p[1] in checkpoints]
        for s_pre, s_post in valid_pairs:
            pair_label = f"{s_pre}->{s_post}"
            for seed in active_seeds:
                for arm in ["ascending", "descending", "nonmonotonic"]:
                    l_pre = next(l for l in data[seed][arm]["logs"] if l["step"] == s_pre)
                    l_post = next(l for l in data[seed][arm]["logs"] if l["step"] == s_post)
                    for h in [32, 64, 128, 256]:
                        h_str = str(h)
                        v_pre = l_pre["horizon_bpc"].get(h, l_pre["horizon_bpc"].get(h_str))
                        v_post = l_post["horizon_bpc"].get(h, l_post["horizon_bpc"].get(h_str))
                        if v_pre is not None and v_post is not None:
                            writer.writerow([seed, arm, pair_label, s_pre, s_post, h, v_post - v_pre])


def generate_contract_figures(data, active_seeds, checkpoints, pre_step, post_step, figures_dir):
    """
    Generate all 10 figures prespecified in Sections 4-13 of Amendment 001.
    """
    os.makedirs(figures_dir, exist_ok=True)
    arms = ["ascending", "descending", "nonmonotonic"]
    horizons = [32, 64, 128, 256]
    steps_arr = np.array(checkpoints)
    
    # Check if extended to step 3000
    has_extension = any(
        any(l["step"] == 3000 for l in data[s][a].get("anchor_logs", []))
        for s in active_seeds for a in arms
    )
    
    # -------------------------------------------------------------
    # Main Figure 1 — Training trajectories by evaluation horizon
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    for idx, h in enumerate(horizons):
        ax = axes[idx // 2, idx % 2]
        h_str = str(h)
        for arm in arms:
            arm_trajs = []
            for seed in active_seeds:
                bpc_series = [
                    l["horizon_bpc"].get(h, l["horizon_bpc"].get(h_str))
                    for l in data[seed][arm]["logs"]
                ]
                arm_trajs.append(bpc_series)
                ax.plot(steps_arr, bpc_series, color=ARM_COLORS[arm], alpha=0.3, lw=1)
                
            mean_bpc = np.mean(arm_trajs, axis=0)
            ax.plot(steps_arr, mean_bpc, color=ARM_COLORS[arm], lw=2.5, label=arm)
            
        for b_step in [500, 1000, 1500, 2000]:
            if b_step <= steps_arr[-1]:
                ax.axvline(x=b_step, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"Evaluation Horizon T={h}")
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Validation BPC")
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend(loc="upper right")
            
    fig.suptitle("Main Figure 1: Training Trajectories by Evaluation Horizon", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "main_01_training_trajectories_by_horizon.png"), dpi=200)
    plt.close()
    
    # -------------------------------------------------------------
    # Main Figure 2 — Recovery gap by evaluation horizon
    # -------------------------------------------------------------
    rec_steps = [s for s in checkpoints if s >= pre_step]
    if len(rec_steps) >= 2:
        plt.figure(figsize=(10, 6))
        for h in horizons:
            h_str = str(h)
            gap_trajs = []
            for seed in active_seeds:
                d_bpc = [
                    next(l for l in data[seed]["descending"]["logs"] if l["step"] == s)["horizon_bpc"].get(h, 
                    next(l for l in data[seed]["descending"]["logs"] if l["step"] == s)["horizon_bpc"].get(h_str))
                    for s in rec_steps
                ]
                a_bpc = [
                    next(l for l in data[seed]["ascending"]["logs"] if l["step"] == s)["horizon_bpc"].get(h,
                    next(l for l in data[seed]["ascending"]["logs"] if l["step"] == s)["horizon_bpc"].get(h_str))
                    for s in rec_steps
                ]
                diff = np.array(d_bpc) - np.array(a_bpc)
                gap_trajs.append(diff)
                plt.plot(np.array(rec_steps) - pre_step, diff, color=HORIZON_COLORS[h], alpha=0.35, lw=1)
                
            mean_gap = np.mean(gap_trajs, axis=0)
            plt.plot(np.array(rec_steps) - pre_step, mean_gap, color=HORIZON_COLORS[h], lw=2.5, label=f"T={h}")
            
        plt.axhline(y=0.0, color="black", linestyle=":", lw=1)
        # Explicit annotation: threshold applies strictly to primary T=256 anchor endpoint
        plt.axhline(y=0.03, color="red", linestyle=":", label="Primary T=256 Decision Threshold (0.03 BPC)")
        for v_k in [0, 250, 500]:
            if v_k <= (rec_steps[-1] - pre_step):
                plt.axvline(x=v_k, color="gray", linestyle="--", alpha=0.5)
        if has_extension:
            for v_k in [750, 1000]:
                if v_k <= (rec_steps[-1] - pre_step):
                    plt.axvline(x=v_k, color="gray", linestyle="--", alpha=0.5)
        plt.xlabel("Recovery Step k")
        plt.ylabel("Δ_h^P(k) [Descending − Ascending BPC]")
        plt.title("Main Figure 2: Recovery Gap by Evaluation Horizon")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(figures_dir, "main_02_recovery_gap_by_horizon.png"), dpi=200, bbox_inches="tight")
        plt.close()
        
    # -------------------------------------------------------------
    # Main Figure 3 & 4 — Anchor Heatmaps at Steps 2000 and 2500
    # -------------------------------------------------------------
    def get_anchor_matrix(step_target):
        mat = np.zeros((3, 4))
        for r_idx, arm in enumerate(arms):
            for c_idx, h in enumerate(horizons):
                h_str = str(h)
                vals = []
                for seed in active_seeds:
                    a_log = next((l for l in data[seed][arm].get("anchor_logs", []) if l["step"] == step_target), None)
                    if a_log:
                        v = a_log["horizon_bpc"].get(h, a_log["horizon_bpc"].get(h_str))
                        vals.append(v)
                mat[r_idx, c_idx] = np.mean(vals) if vals else 0.0
        return mat
        
    mat_2000 = get_anchor_matrix(pre_step)
    mat_2500 = get_anchor_matrix(post_step)
    
    vmin = min(mat_2000.min(), mat_2500.min())
    vmax = max(mat_2000.max(), mat_2500.max())
    
    for s_step, mat, fname, title in [
        (pre_step, mat_2000, "main_03_anchor_heatmap_step2000.png", f"Main Figure 3: Anchor Heatmap Before Recovery (Step {pre_step})"),
        (post_step, mat_2500, "main_04_anchor_heatmap_step2500.png", f"Main Figure 4: Anchor Heatmap After Recovery (Step {post_step})"),
    ]:
        plt.figure(figsize=(8, 5))
        plt.imshow(mat, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        plt.colorbar(label="Mean Anchor BPC")
        plt.xticks(range(4), [f"T={h}" for h in horizons])
        plt.yticks(range(3), [f"{a} (exploratory)" if a == "nonmonotonic" else a for a in arms])
        for i in range(3):
            for j in range(4):
                plt.text(j, i, f"{mat[i, j]:.4f}", ha="center", va="center", color="white" if mat[i, j] < (vmin + vmax)/2 else "black")
        plt.title(title)
        plt.savefig(os.path.join(figures_dir, fname), dpi=200, bbox_inches="tight")
        plt.close()
        
    # -------------------------------------------------------------
    # Main Figure 5 — Pre-to-Post Change Heatmap
    # -------------------------------------------------------------
    change_mat = mat_2500 - mat_2000
    abs_max = max(abs(change_mat.min()), abs(change_mat.max()), 1e-4)
    plt.figure(figsize=(8, 5))
    plt.imshow(change_mat, cmap="coolwarm", vmin=-abs_max, vmax=abs_max, aspect="auto")
    plt.colorbar(label="Δ BPC (Post − Pre)")
    plt.xticks(range(4), [f"T={h}" for h in horizons])
    plt.yticks(range(3), [f"{a} (exploratory)" if a == "nonmonotonic" else a for a in arms])
    for i in range(3):
        for j in range(4):
            val = change_mat[i, j]
            label = f"{val:+.4f}\n({'imp' if val < 0 else 'deg'})"
            plt.text(j, i, label, ha="center", va="center", color="black")
    plt.title("Main Figure 5: Pre-to-Post Anchor Change (Negative = Improvement)")
    plt.savefig(os.path.join(figures_dir, "main_05_pre_post_change_heatmap.png"), dpi=200, bbox_inches="tight")
    plt.close()
    
    # -------------------------------------------------------------
    # Appendix A1 — Transition-Local Shock
    # -------------------------------------------------------------
    candidate_pairs = [
        (500, 501), (1000, 1001), (1500, 1501), (2000, 2001),
        (10, 11), (20, 21), (30, 31), (40, 41)
    ]
    valid_pairs = [p for p in candidate_pairs if p[0] in checkpoints and p[1] in checkpoints]
    if valid_pairs:
        fig, axes = plt.subplots(1, len(valid_pairs), figsize=(4 * len(valid_pairs), 4), sharey=True)
        if len(valid_pairs) == 1:
            axes = [axes]
        for p_idx, (s_pre, s_post) in enumerate(valid_pairs):
            ax = axes[p_idx]
            bar_w = 0.25
            for a_idx, arm in enumerate(arms):
                mean_shocks = []
                for h_idx, h in enumerate(horizons):
                    h_str = str(h)
                    shocks = []
                    for seed in active_seeds:
                        l1 = next(l for l in data[seed][arm]["logs"] if l["step"] == s_pre)
                        l2 = next(l for l in data[seed][arm]["logs"] if l["step"] == s_post)
                        v1 = l1["horizon_bpc"].get(h, l1["horizon_bpc"].get(h_str))
                        v2 = l2["horizon_bpc"].get(h, l2["horizon_bpc"].get(h_str))
                        shocks.append(v2 - v1)
                    mean_shocks.append(np.mean(shocks))
                    # Plot raw seed points
                    for sk in shocks:
                        ax.scatter(h_idx + (a_idx - 1) * bar_w, sk, color=ARM_COLORS[arm], alpha=0.4, s=15, zorder=3)
                ax.bar(np.arange(4) + (a_idx - 1) * bar_w, mean_shocks, width=bar_w, color=ARM_COLORS[arm], alpha=0.7, label=arm if p_idx == 0 else "")
            ax.set_xticks(range(4))
            ax.set_xticklabels([f"T={h}" for h in horizons])
            ax.set_title(f"Transition {s_pre}→{s_post}")
            ax.axhline(y=0.0, color="black", linestyle=":", lw=1)
            ax.grid(True, alpha=0.25)
        axes[0].set_ylabel("Shock BPC (Post − Pre)")
        fig.suptitle("Appendix Figure A1: Transition-Local Loss Shock", fontsize=13)
        axes[0].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "appendix_A1_transition_shock.png"), dpi=200)
        plt.close()

    # -------------------------------------------------------------
    # Appendix A2 — Context Profiles (Step 2000 vs 2500)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, step_t, mat, title in [(ax1, pre_step, mat_2000, f"Step {pre_step}"), (ax2, post_step, mat_2500, f"Step {post_step}")]:
        for r_idx, arm in enumerate(arms):
            # Raw seed points
            for seed in active_seeds:
                a_log = next((l for l in data[seed][arm].get("anchor_logs", []) if l["step"] == step_t), None)
                if a_log:
                    seed_vals = [a_log["horizon_bpc"].get(h, a_log["horizon_bpc"].get(str(h))) for h in horizons]
                    ax.plot([0, 1, 2, 3], seed_vals, color=ARM_COLORS[arm], alpha=0.3, lw=1)
                    ax.scatter([0, 1, 2, 3], seed_vals, color=ARM_COLORS[arm], alpha=0.5, s=20)
            # 3-seed mean line
            ax.plot([0, 1, 2, 3], mat[r_idx, :], marker="o", color=ARM_COLORS[arm], lw=2.5, label=arm)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels([f"T={h}" for h in horizons])
        ax.set_xlabel("Evaluation Horizon")
        ax.set_ylabel("Anchor BPC")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Appendix Figure A2: Context Profiles Before and After Recovery", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "appendix_A2_context_profiles.png"), dpi=200)
    plt.close()
    
    # -------------------------------------------------------------
    # Appendix A3 — Paired Seed Endpoints
    # -------------------------------------------------------------
    plt.figure(figsize=(7, 5))
    x_ticks = [0, 1]
    x_labels = [f"Pre (Step {pre_step})", f"Post (Step {post_step})"]
    if has_extension:
        x_ticks.append(2)
        x_labels.append("Post-Ext (Step 3000)")
        
    diffs_pre = []
    diffs_post = []
    diffs_ext = []
    
    for seed in active_seeds:
        a_pre = next(l for l in data[seed]["ascending"]["anchor_logs"] if l["step"] == pre_step)["anchor_bpc"]
        d_pre = next(l for l in data[seed]["descending"]["anchor_logs"] if l["step"] == pre_step)["anchor_bpc"]
        a_post = next(l for l in data[seed]["ascending"]["anchor_logs"] if l["step"] == post_step)["anchor_bpc"]
        d_post = next(l for l in data[seed]["descending"]["anchor_logs"] if l["step"] == post_step)["anchor_bpc"]
        
        y_pts = [d_pre - a_pre, d_post - a_post]
        diffs_pre.append(d_pre - a_pre)
        diffs_post.append(d_post - a_post)
        
        if has_extension:
            a_3000 = next((l for l in data[seed]["ascending"]["anchor_logs"] if l["step"] == 3000), None)
            d_3000 = next((l for l in data[seed]["descending"]["anchor_logs"] if l["step"] == 3000), None)
            if a_3000 and d_3000:
                y_pts.append(d_3000["anchor_bpc"] - a_3000["anchor_bpc"])
                diffs_ext.append(d_3000["anchor_bpc"] - a_3000["anchor_bpc"])
                
        plt.plot(x_ticks, y_pts, marker="o", lw=1.5, alpha=0.5, label=f"Seed {seed}")
        
    # Overlay mean marker
    mean_pts = [np.mean(diffs_pre), np.mean(diffs_post)]
    if has_extension and diffs_ext:
        mean_pts.append(np.mean(diffs_ext))
    plt.plot(x_ticks, mean_pts, marker="s", color="black", lw=3, label="3-Seed Mean", zorder=4)
    
    plt.axhline(y=0.0, color="black", linestyle=":", lw=1)
    plt.axhline(y=0.03, color="red", linestyle=":", label="0.03 BPC")
    plt.xticks(x_ticks, x_labels)
    plt.ylabel("Paired Δ^A BPC (Descending − Ascending)")
    plt.title("Appendix Figure A3: Paired Seed Endpoints")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(figures_dir, "appendix_A3_paired_seed_endpoints.png"), dpi=200, bbox_inches="tight")
    plt.close()
    
    # -------------------------------------------------------------
    # Appendix A4 — Context Alignment Contrast
    # -------------------------------------------------------------
    if len(rec_steps) >= 2:
        plt.figure(figsize=(9, 5))
        align_trajs = []
        for seed in active_seeds:
            d_logs = {l["step"]: l for l in data[seed]["descending"]["logs"]}
            a_logs = {l["step"]: l for l in data[seed]["ascending"]["logs"]}
            a_series = []
            for s in rec_steps:
                d_256 = d_logs[s]["horizon_bpc"].get(256, d_logs[s]["horizon_bpc"].get("256"))
                a_256 = a_logs[s]["horizon_bpc"].get(256, a_logs[s]["horizon_bpc"].get("256"))
                d_32 = d_logs[s]["horizon_bpc"].get(32, d_logs[s]["horizon_bpc"].get("32"))
                a_32 = a_logs[s]["horizon_bpc"].get(32, a_logs[s]["horizon_bpc"].get("32"))
                delta_256 = d_256 - a_256
                delta_32 = d_32 - a_32
                a_series.append(delta_256 - delta_32)
            align_trajs.append(a_series)
            plt.plot(np.array(rec_steps) - pre_step, a_series, alpha=0.35, lw=1)
        mean_align = np.mean(align_trajs, axis=0)
        plt.plot(np.array(rec_steps) - pre_step, mean_align, color="#1f77b4", lw=2.5, label="Mean A^P(k)")
        plt.axhline(y=0.0, color="black", linestyle=":", lw=1)
        plt.xlabel("Recovery Step k")
        plt.ylabel("A^P(k) = Δ_256^P(k) − Δ_32^P(k)")
        plt.title("Appendix Figure A4: Context-Alignment Contrast")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(figures_dir, "appendix_A4_context_alignment.png"), dpi=200, bbox_inches="tight")
        plt.close()
        
    # -------------------------------------------------------------
    # Appendix A5 — Nonmonotonic Exploratory Contrasts
    # -------------------------------------------------------------
    if len(rec_steps) >= 2:
        plt.figure(figsize=(10, 6))
        for h in horizons:
            h_str = str(h)
            na_gaps = []
            dn_gaps = []
            for seed in active_seeds:
                n_bpc = [next(l for l in data[seed]["nonmonotonic"]["logs"] if l["step"] == s)["horizon_bpc"].get(h, 
                         next(l for l in data[seed]["nonmonotonic"]["logs"] if l["step"] == s)["horizon_bpc"].get(h_str)) for s in rec_steps]
                a_bpc = [next(l for l in data[seed]["ascending"]["logs"] if l["step"] == s)["horizon_bpc"].get(h, 
                         next(l for l in data[seed]["ascending"]["logs"] if l["step"] == s)["horizon_bpc"].get(h_str)) for s in rec_steps]
                d_bpc = [next(l for l in data[seed]["descending"]["logs"] if l["step"] == s)["horizon_bpc"].get(h, 
                         next(l for l in data[seed]["descending"]["logs"] if l["step"] == s)["horizon_bpc"].get(h_str)) for s in rec_steps]
                na_gaps.append(np.array(n_bpc) - np.array(a_bpc))
                dn_gaps.append(np.array(d_bpc) - np.array(n_bpc))
            plt.plot(np.array(rec_steps) - pre_step, np.mean(na_gaps, axis=0), color=HORIZON_COLORS[h], lw=2, linestyle="-", label=f"N−A (T={h})")
            plt.plot(np.array(rec_steps) - pre_step, np.mean(dn_gaps, axis=0), color=HORIZON_COLORS[h], lw=2, linestyle="--", label=f"D−N (T={h})")
        plt.axhline(y=0.0, color="black", linestyle=":", lw=1)
        plt.xlabel("Recovery Step k")
        plt.ylabel("Δ BPC")
        plt.title("Appendix Figure A5: Nonmonotonic Exploratory Contrasts\n(Exploratory — one prespecified nonmonotonic permutation)")
        plt.legend(loc="upper right", ncol=2)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(figures_dir, "appendix_A5_nonmonotonic_exploratory.png"), dpi=200, bbox_inches="tight")
        plt.close()
        
    # -------------------------------------------------------------
    # Appendix A6 — Extension Anchor Heatmap (Step 3000, if present)
    # -------------------------------------------------------------
    if has_extension:
        mat_3000 = get_anchor_matrix(3000)
        plt.figure(figsize=(8, 5))
        plt.imshow(mat_3000, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        plt.colorbar(label="Mean Anchor BPC")
        plt.xticks(range(4), [f"T={h}" for h in horizons])
        plt.yticks(range(3), [f"{a} (exploratory)" if a == "nonmonotonic" else a for a in arms])
        for i in range(3):
            for j in range(4):
                plt.text(j, i, f"{mat_3000[i, j]:.4f}", ha="center", va="center", color="white" if mat_3000[i, j] < (vmin + vmax)/2 else "black")
        plt.title("Appendix Figure A6: Extension Anchor Heatmap (Step 3000)")
        plt.savefig(os.path.join(figures_dir, "appendix_A6_extension_anchor_step3000.png"), dpi=200, bbox_inches="tight")
        plt.close()
        
    print(f"Generated contract CSVs in {os.path.join(figures_dir, 'data')}")
    print(f"Generated contract figures in {figures_dir}")
