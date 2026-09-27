"""
Figure and Process-Analysis Contract Exporter
Preregistered specification: PREREGISTRATION_AMENDMENT_001_FIGURES.md

Pure tabular plotting layer consuming canonical StudyTables.
"""

import os
from typing import Any, Dict, List
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .core_types import StudyTables
from .config import ARMS, CONTEXT_LENGTHS, THRESHOLD_MIN_PRE_GAP, THRESHOLD_RESIDUAL_PERSISTENT

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


def plot_contract_figures(
    tables: StudyTables,
    summary: Dict[str, Any],
    active_seeds: List[int],
    checkpoints: List[int],
    figures_dir: str,
):
    """
    Renders all 5 Main figures and 6 Appendix figures specified by Amendment 001.
    All data is queried cleanly through canonical StudyTables.
    """
    os.makedirs(figures_dir, exist_ok=True)
    horizons = CONTEXT_LENGTHS
    steps_arr = np.array(checkpoints)
    pre_step = summary["base_pre_step"]
    post_step = summary["base_post_step"]
    has_extension = summary.get("has_complete_extension", False)
    
    # -------------------------------------------------------------
    # Main Figure 1 — Training trajectories by evaluation horizon
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    for idx, h in enumerate(horizons):
        ax = axes[idx // 2, idx % 2]
        for arm in ARMS:
            arm_trajs = []
            for seed in active_seeds:
                bpc_series = [tables.process_bpc(seed, arm, step, h) for step in checkpoints]
                arm_trajs.append(bpc_series)
                ax.plot(steps_arr, bpc_series, color=ARM_COLORS[arm], alpha=0.3, lw=1)
                
            mean_bpc = np.nanmean(arm_trajs, axis=0)
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
    recovery_k = [s - pre_step for s in rec_steps]
    if len(rec_steps) >= 2:
        plt.figure(figsize=(10, 6))
        for h in horizons:
            gap_trajs = []
            for seed in active_seeds:
                gaps = [
                    tables.process_bpc(seed, "descending", s, h) - tables.process_bpc(seed, "ascending", s, h)
                    for s in rec_steps
                ]
                gap_trajs.append(gaps)
                plt.plot(recovery_k, gaps, color=HORIZON_COLORS[h], alpha=0.22, lw=1)
            mean_gaps = np.nanmean(gap_trajs, axis=0)
            plt.plot(recovery_k, mean_gaps, color=HORIZON_COLORS[h], lw=2.5, marker="o", label=f"T={h}")
            
        plt.axhline(y=0.0, color="black", linestyle="-", lw=1, alpha=0.7)

        for k_offset in [0, 250, 500]:
            if k_offset in recovery_k:
                plt.axvline(x=k_offset, color="gray", linestyle=":", alpha=0.6)
                
        plt.title("Main Figure 2: Process Recovery Gap Δ^P(k) by Horizon (T=256 Recovery)")
        plt.xlabel("Recovery Step k")
        plt.ylabel("Gap: Descending − Ascending (BPC)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "main_02_recovery_gap_by_horizon.png"), dpi=200)
        plt.close()

    # -------------------------------------------------------------
    # Main Figures 3 & 4 — Anchor Performance Heatmaps
    # -------------------------------------------------------------
    anchor_matrices = {}
    for step_t in [pre_step, post_step]:
        matrix = np.zeros((len(ARMS), len(horizons)))
        for r_idx, arm in enumerate(ARMS):
            for c_idx, h in enumerate(horizons):
                vals = [tables.anchor_bpc(seed, arm, step_t, h) for seed in active_seeds]
                matrix[r_idx, c_idx] = np.nanmean(vals)
        anchor_matrices[step_t] = matrix

    shared_vmin = min(float(np.nanmin(matrix)) for matrix in anchor_matrices.values())
    shared_vmax = max(float(np.nanmax(matrix)) for matrix in anchor_matrices.values())
    shared_midpoint = (shared_vmin + shared_vmax) / 2.0

    for fig_num, step_t in [(3, pre_step), (4, post_step)]:
        matrix = anchor_matrices[step_t]
        plt.figure(figsize=(7, 4.5))
        plt.imshow(matrix, cmap="viridis", aspect="auto", vmin=shared_vmin, vmax=shared_vmax)
        plt.colorbar(label="Anchor Validation BPC")
        plt.xticks(range(len(horizons)), [f"T={h}" for h in horizons])
        plt.yticks(range(len(ARMS)), ARMS)
        for i in range(len(ARMS)):
            for j in range(len(horizons)):
                color = "white" if matrix[i, j] < shared_midpoint else "black"
                plt.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center", color=color)
        plt.title(f"Main Figure {fig_num}: Anchor BPC Heatmap at Step {step_t}")
        fname = f"main_0{fig_num}_anchor_heatmap_step{step_t}.png"
        plt.savefig(os.path.join(figures_dir, fname), dpi=200, bbox_inches="tight")
        plt.close()

    # -------------------------------------------------------------
    # Main Figure 5 — Pre-to-Post Anchor Change Heatmap
    # -------------------------------------------------------------
    mat_2000 = anchor_matrices[pre_step]
    mat_2500 = anchor_matrices[post_step]
    change_matrix = mat_2500 - mat_2000
    
    plt.figure(figsize=(7, 4.5))
    vmax = max(abs(np.nanmin(change_matrix)), abs(np.nanmax(change_matrix)), 0.05)
    plt.imshow(change_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(label="Δ BPC (Post − Pre)")
    plt.xticks(range(len(horizons)), [f"T={h}" for h in horizons])
    plt.yticks(range(len(ARMS)), ARMS)
    for i in range(len(ARMS)):
        for j in range(len(horizons)):
            val = change_matrix[i, j]
            label = f"{val:+.4f}\n({'imp' if val < 0 else 'deg'})"
            plt.text(j, i, label, ha="center", va="center", color="black")
    plt.title("Main Figure 5: Pre-to-Post Anchor Change (Negative = Improvement)")
    plt.savefig(os.path.join(figures_dir, "main_05_pre_post_change_heatmap.png"), dpi=200, bbox_inches="tight")
    plt.close()
    
    # -------------------------------------------------------------
    # Appendix A1 — Transition-Local Shock
    # -------------------------------------------------------------
    shocks_records = tables.transition_shock_records
    if shocks_records:
        unique_transitions = sorted(list({r["transition"] for r in shocks_records}))
        fig, axes = plt.subplots(1, len(unique_transitions), figsize=(4 * len(unique_transitions), 4), sharey=True)
        if len(unique_transitions) == 1:
            axes = [axes]
        bar_w = 0.25
        for p_idx, trans in enumerate(unique_transitions):
            ax = axes[p_idx]
            for a_idx, arm in enumerate(ARMS):
                mean_shocks = []
                for h_idx, h in enumerate(horizons):
                    recs = [
                        r["shock_bpc"] for r in shocks_records
                        if r["transition"] == trans and r["arm"] == arm and r["evaluation_horizon"] == h
                    ]
                    mean_val = float(np.mean(recs)) if recs else 0.0
                    mean_shocks.append(mean_val)
                    for sk in recs:
                        ax.scatter(h_idx + (a_idx - 1) * bar_w, sk, color=ARM_COLORS[arm], alpha=0.4, s=15, zorder=3)
                ax.bar(np.arange(4) + (a_idx - 1) * bar_w, mean_shocks, width=bar_w, color=ARM_COLORS[arm], alpha=0.7, label=arm if p_idx == 0 else "")
            ax.set_xticks(range(4))
            ax.set_xticklabels([f"T={h}" for h in horizons])
            ax.set_title(f"Transition {trans}")
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
        for r_idx, arm in enumerate(ARMS):
            for seed in active_seeds:
                raw_curve = [tables.anchor_bpc(seed, arm, step_t, h) for h in horizons]
                ax.plot(range(4), raw_curve, color=ARM_COLORS[arm], alpha=0.3, lw=1, linestyle=":")
                ax.scatter(range(4), raw_curve, color=ARM_COLORS[arm], alpha=0.4, s=20)
            ax.plot(range(4), mat[r_idx], color=ARM_COLORS[arm], lw=2.5, marker="o", label=arm)
        ax.set_xticks(range(4))
        ax.set_xticklabels([f"T={h}" for h in horizons])
        ax.set_title(title)
        ax.set_xlabel("Evaluation Horizon")
        ax.grid(True, alpha=0.3)
    ax1.set_ylabel("Anchor Validation BPC")
    ax1.legend()
    fig.suptitle("Appendix Figure A2: Anchor Context Profiles (Mean & Raw Seed Points)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "appendix_A2_context_profiles.png"), dpi=200)
    plt.close()

    # -------------------------------------------------------------
    # Appendix A3 — Paired Seed Endpoints
    # -------------------------------------------------------------
    plt.figure(figsize=(7, 6))
    for seed in active_seeds:
        d_pre = tables.anchor_bpc(seed, "descending", pre_step, 256) - tables.anchor_bpc(seed, "ascending", pre_step, 256)
        d_post = tables.anchor_bpc(seed, "descending", post_step, 256) - tables.anchor_bpc(seed, "ascending", post_step, 256)
        x_pts = [0, 500]
        y_pts = [d_pre, d_post]
        if has_extension:
            d_3000 = tables.anchor_bpc(seed, "descending", 3000, 256) - tables.anchor_bpc(seed, "ascending", 3000, 256)
            x_pts.append(1000)
            y_pts.append(d_3000)
        plt.plot(x_pts, y_pts, marker="o", lw=1.5, alpha=0.5, label=f"Seed {seed}")
        
    mean_pre = summary["stats_pre"]["mean"]
    mean_post = summary["stats_post_500"]["mean"]
    m_x = [0, 500]
    m_y = [mean_pre, mean_post]
    if has_extension and summary.get("extension_results") is not None:
        m_x.append(1000)
        m_y.append(summary["extension_results"]["stats_post_1000"]["mean"])
    plt.plot(m_x, m_y, marker="s", color="black", lw=3.0, label="3-Seed Mean", zorder=4)
    
    plt.axhspan(
        -THRESHOLD_MIN_PRE_GAP,
        THRESHOLD_MIN_PRE_GAP,
        color="gray",
        alpha=0.15,
        label=f"Case B band (|Δ| ≤ {THRESHOLD_MIN_PRE_GAP:.2f})",
    )
    plt.axhline(y=0.0, color="black", linestyle="-", lw=1)
    plt.axhline(
        y=THRESHOLD_RESIDUAL_PERSISTENT,
        color="red",
        linestyle="--",
        label=f"Positive persistence threshold (+{THRESHOLD_RESIDUAL_PERSISTENT:.2f})",
    )
    plt.xticks(m_x, [f"Step {pre_step}\n(k=0)", f"Step {post_step}\n(k=500)"] + ([f"Step 3000\n(k=1000)"] if has_extension else []))
    plt.ylabel("Gap: Descending − Ascending (BPC at T=256)")
    plt.title("Appendix Figure A3: Paired Seed Trajectories & 3-Seed Mean")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "appendix_A3_paired_seed_endpoints.png"), dpi=200)
    plt.close()

    # -------------------------------------------------------------
    # Appendix A4 — Context Alignment Gap
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5.5))
    alignment_trajectories = []
    for seed in active_seeds:
        values = []
        for step in rec_steps:
            delta_256 = tables.process_bpc(seed, "descending", step, 256) - tables.process_bpc(seed, "ascending", step, 256)
            delta_32 = tables.process_bpc(seed, "descending", step, 32) - tables.process_bpc(seed, "ascending", step, 32)
            values.append(delta_256 - delta_32)
        alignment_trajectories.append(values)
        plt.plot(recovery_k, values, color="#4c78a8", alpha=0.3, lw=1.2, marker="o", markersize=3)
    plt.plot(
        recovery_k,
        np.mean(alignment_trajectories, axis=0),
        color="#173f5f",
        lw=3,
        marker="o",
        label="3-seed mean",
    )
    plt.axhline(y=0.0, color="black", linestyle="--", lw=1)
    plt.xlabel("Recovery Step k")
    plt.ylabel(r"$A^P(k)=\Delta_{256}^{P}(k)-\Delta_{32}^{P}(k)$ (BPC)")
    plt.title("Appendix Figure A4: Context-Alignment Contrast During Recovery")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "appendix_A4_context_alignment.png"), dpi=200)
    plt.close()

    # -------------------------------------------------------------
    # Appendix A5 — Nonmonotonic Exploratory Contrasts
    # -------------------------------------------------------------
    if len(rec_steps) >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=False)
        for idx, h in enumerate(horizons):
            ax = axes[idx // 2, idx % 2]
            contrast_means = {}
            for label, first_arm, second_arm, color in [
                ("Nonmonotonic − Ascending", "nonmonotonic", "ascending", "purple"),
                ("Descending − Nonmonotonic", "descending", "nonmonotonic", "teal"),
            ]:
                seed_trajectories = []
                for seed in active_seeds:
                    values = [
                        tables.process_bpc(seed, first_arm, step, h)
                        - tables.process_bpc(seed, second_arm, step, h)
                        for step in rec_steps
                    ]
                    seed_trajectories.append(values)
                    ax.plot(recovery_k, values, color=color, alpha=0.22, lw=1)
                contrast_means[label] = np.mean(seed_trajectories, axis=0)
                ax.plot(recovery_k, contrast_means[label], color=color, lw=2.5, marker="o", label=label)
            ax.axhline(y=0.0, color="black", linestyle=":", lw=1)
            ax.set_title(f"Evaluation Horizon T={h}")
            if idx >= 2:
                ax.set_xlabel("Recovery Step k")
            ax.set_ylabel("Process Contrast (BPC)")
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend()
        fig.suptitle(
            "Appendix Figure A5: Exploratory — one prespecified nonmonotonic permutation",
            fontsize=13,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "appendix_A5_nonmonotonic_exploratory.png"), dpi=200)
        plt.close()

    # -------------------------------------------------------------
    # Appendix A6 — Extension Anchor Heatmap (Step 3000)
    # -------------------------------------------------------------
    if has_extension:
        mat_3000 = np.zeros((len(ARMS), len(horizons)))
        for r_idx, arm in enumerate(ARMS):
            for c_idx, h in enumerate(horizons):
                v_3000 = [tables.anchor_bpc(seed, arm, 3000, h) for seed in active_seeds]
                mat_3000[r_idx, c_idx] = np.nanmean(v_3000)
        plt.figure(figsize=(7, 4.5))
        plt.imshow(mat_3000, cmap="viridis", aspect="auto")
        plt.colorbar(label="Anchor Validation BPC")
        plt.xticks(range(len(horizons)), [f"T={h}" for h in horizons])
        plt.yticks(range(len(ARMS)), ARMS)
        for i in range(len(ARMS)):
            for j in range(len(horizons)):
                plt.text(j, i, f"{mat_3000[i, j]:.4f}", ha="center", va="center", color="white" if mat_3000[i, j] > np.nanmean(mat_3000) else "black")
        plt.title("Appendix Figure A6: Extension Anchor BPC Heatmap at Step 3000")
        plt.savefig(os.path.join(figures_dir, "appendix_A6_extension_anchor_step3000.png"), dpi=200, bbox_inches="tight")
        plt.close()
