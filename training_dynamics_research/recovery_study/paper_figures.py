"""Post-results paper figures built from the frozen recovery-study artifacts.

These figures change presentation, not estimands or decision rules. The
preregistered contract figures remain available in the parent figures folder.
"""

import os
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import CONTEXT_LENGTHS, THRESHOLD_MIN_PRE_GAP
from .core_types import StudyTables


HORIZON_COLORS = {
    32: "#56B4E9",
    64: "#009E73",
    128: "#E69F00",
    256: "#7B2CBF",
}
PRE_COLOR = "#0072B2"
POST_COLOR = "#D55E00"
RAW_ALPHA = 0.24


def _save(fig: plt.Figure, output_dir: str, filename: str) -> None:
    fig.savefig(os.path.join(output_dir, filename), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _recovery_axis(checkpoints: List[int], pre_step: int) -> tuple[List[int], List[int]]:
    steps = [step for step in checkpoints if step >= pre_step]
    return steps, [step - pre_step for step in steps]


def _process_gap(
    tables: StudyTables,
    seed: int,
    steps: List[int],
    horizon: int,
) -> np.ndarray:
    return np.asarray([
        tables.process_bpc(seed, "descending", step, horizon)
        - tables.process_bpc(seed, "ascending", step, horizon)
        for step in steps
    ])


def _style_effect_axis(ax: plt.Axes) -> None:
    ax.axvline(0.0, color="black", lw=1, zorder=0)
    ax.grid(axis="x", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)


def _plot_recovery_dynamics(
    tables: StudyTables,
    active_seeds: List[int],
    recovery_steps: List[int],
    recovery_k: List[int],
    output_dir: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), sharex=True)
    short_horizon_values = []

    for idx, horizon in enumerate(CONTEXT_LENGTHS):
        ax = axes[idx // 2, idx % 2]
        trajectories = []
        for seed in active_seeds:
            values = _process_gap(tables, seed, recovery_steps, horizon)
            trajectories.append(values)
            if horizon != 256:
                short_horizon_values.extend(values.tolist())
            ax.plot(
                recovery_k,
                values,
                color=HORIZON_COLORS[horizon],
                alpha=RAW_ALPHA,
                lw=1,
                marker="o",
                markersize=3,
            )

        mean_values = np.mean(trajectories, axis=0)
        ax.plot(
            recovery_k,
            mean_values,
            color=HORIZON_COLORS[horizon],
            lw=2.8,
            marker="o",
            markersize=6,
            label="3-seed mean",
        )
        ax.axhline(0.0, color="black", lw=1)
        for marker in (0, 250, 500):
            ax.axvline(marker, color="0.72", lw=0.9, ls=":", zorder=0)
        ax.set_title(f"Evaluation horizon T={horizon}")
        ax.set_ylabel("Descending − ascending BPC")
        ax.grid(alpha=0.2)
        if idx >= 2:
            ax.set_xlabel("Recovery step k")
        if idx == 0:
            ax.legend(frameon=False, loc="lower right")

    short_min = min(short_horizon_values) - 0.02
    short_max = max(short_horizon_values) + 0.02
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_ylim(short_min, short_max)
    axes[1, 1].text(
        0.98,
        0.96,
        "Independent y-scale",
        transform=axes[1, 1].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="0.35",
    )
    fig.suptitle(
        "Figure 1 | Recovery dynamics across evaluation horizons\n"
        "Markers are measured checkpoints; faint lines are individual seeds",
        fontsize=14,
    )
    fig.tight_layout()
    _save(fig, output_dir, "figure_01_recovery_dynamics.png")


def _plot_primary_endpoint(summary: Dict[str, Any], output_dir: str) -> None:
    pre = summary["stats_pre"]
    post = summary["stats_post_500"]
    recovered = summary["stats_abs_recovery_500"]

    fig, (ax_pairs, ax_post, ax_recovery) = plt.subplots(
        1,
        3,
        figsize=(14.5, 5.7),
        gridspec_kw={"width_ratios": [1.25, 0.85, 0.85]},
    )

    ax_pairs.axhspan(
        -THRESHOLD_MIN_PRE_GAP,
        THRESHOLD_MIN_PRE_GAP,
        color="0.6",
        alpha=0.16,
        label=f"Practical-removal band (±{THRESHOLD_MIN_PRE_GAP:.2f})",
    )
    for pre_value, post_value in zip(pre["raw_diffs"], post["raw_diffs"]):
        ax_pairs.plot([0, 1], [pre_value, post_value], color="0.65", lw=1.2, zorder=1)
        ax_pairs.scatter(0, pre_value, color=PRE_COLOR, s=54, zorder=2)
        ax_pairs.scatter(1, post_value, color=POST_COLOR, s=54, zorder=2)
    ax_pairs.scatter([0, 1], [pre["mean"], post["mean"]], marker="D", color="black", s=68, zorder=3, label="3-seed mean")
    ax_pairs.axhline(0.0, color="black", lw=1)
    ax_pairs.set_xticks([0, 1], ["Before recovery\n(k=0)", "After recovery\n(k=500)"])
    ax_pairs.set_ylabel(r"Anchor gap $\Delta^A$ at T=256 (BPC)")
    ax_pairs.set_title("A  Paired anchor endpoints only\n(lines show pairing, not intermediate dynamics)", loc="left")
    ax_pairs.grid(axis="y", alpha=0.22)
    ax_pairs.spines[["top", "right"]].set_visible(False)
    ax_pairs.legend(frameon=False, loc="upper right")

    def draw_effect_panel(
        ax: plt.Axes,
        stats: Dict[str, Any],
        color: str,
        title: str,
        xlim: tuple[float, float],
        show_equivalence_band: bool = False,
    ) -> None:
        if show_equivalence_band:
            ax.axvspan(-THRESHOLD_MIN_PRE_GAP, THRESHOLD_MIN_PRE_GAP, color="0.6", alpha=0.16)
        raw = np.asarray(stats["raw_diffs"])
        jitter = np.linspace(-0.07, 0.07, len(raw))
        ax.scatter(raw, jitter, color=color, alpha=0.62, s=38, zorder=2)
        low, high = stats["ci_95"]
        ax.errorbar(
            stats["mean"],
            0,
            xerr=[[stats["mean"] - low], [high - stats["mean"]]],
            fmt="D",
            color="black",
            ecolor=color,
            elinewidth=2.5,
            capsize=5,
            markersize=7,
            zorder=3,
        )
        ax.text(stats["mean"], 0.14, f"mean {stats['mean']:+.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.18, 0.22)
        ax.set_yticks([])
        ax.set_xlabel("Effect size (BPC)\nmean with 95% t interval")
        ax.set_title(title, loc="left")
        _style_effect_axis(ax)

    post_low, post_high = post["ci_95"]
    recovered_low, recovered_high = recovered["ci_95"]
    draw_effect_panel(
        ax_post,
        post,
        POST_COLOR,
        "B  Post-recovery residual",
        (post_low - 0.035, post_high + 0.035),
        show_equivalence_band=True,
    )
    draw_effect_panel(
        ax_recovery,
        recovered,
        "#009E73",
        "C  Pre-to-post reduction",
        (recovered_low - 0.10, recovered_high + 0.10),
    )
    fig.suptitle("Figure 2 | Primary endpoint: the large deficit reversed, but the residual is unresolved", fontsize=14)
    fig.tight_layout()
    _save(fig, output_dir, "figure_02_primary_endpoint_estimation.png")


def _plot_horizon_effects(summary: Dict[str, Any], output_dir: str) -> None:
    horizon_stats = summary["horizon_endpoint_stats"]
    horizons = list(CONTEXT_LENGTHS)
    fig, (ax_pre, ax_post) = plt.subplots(1, 2, figsize=(12.5, 5.8), sharey=True)

    for ax, endpoint, title in [
        (ax_pre, "pre", "A  Before recovery (k=0)"),
        (ax_post, "post_500", "B  After recovery (k=500)"),
    ]:
        for y, horizon in enumerate(horizons):
            stats = horizon_stats[str(horizon)] if str(horizon) in horizon_stats else horizon_stats[horizon]
            values = stats[endpoint]
            raw = np.asarray(values["raw_diffs"])
            jitter = np.linspace(-0.07, 0.07, len(raw))
            ax.scatter(raw, y + jitter, color=HORIZON_COLORS[horizon], alpha=0.55, s=30, zorder=2)
            low, high = values["ci_95"]
            ax.errorbar(
                values["mean"],
                y,
                xerr=[[values["mean"] - low], [high - values["mean"]]],
                fmt="D",
                color="black",
                ecolor=HORIZON_COLORS[horizon],
                elinewidth=2.4,
                capsize=5,
                markersize=6,
                zorder=3,
            )
        ax.set_yticks(range(len(horizons)), [f"T={h}" for h in horizons])
        ax.set_xlabel("Descending − ascending anchor BPC")
        ax.set_title(title, loc="left")
        _style_effect_axis(ax)

    ax_post.axvspan(-THRESHOLD_MIN_PRE_GAP, THRESHOLD_MIN_PRE_GAP, color="0.6", alpha=0.16)
    fig.suptitle(
        "Figure 3 | Horizon-specific paired effects\n"
        "Dots are seeds; diamonds and bars are means with 95% t intervals",
        fontsize=14,
    )
    fig.tight_layout()
    _save(fig, output_dir, "figure_03_horizon_effect_forest.png")


def _alignment_values(
    tables: StudyTables,
    seed: int,
    recovery_steps: List[int],
) -> np.ndarray:
    delta_256 = _process_gap(tables, seed, recovery_steps, 256)
    delta_32 = _process_gap(tables, seed, recovery_steps, 32)
    return delta_256 - delta_32


def _plot_context_alignment(
    tables: StudyTables,
    active_seeds: List[int],
    recovery_steps: List[int],
    recovery_k: List[int],
    output_dir: str,
) -> None:
    trajectories = np.asarray([
        _alignment_values(tables, seed, recovery_steps)
        for seed in active_seeds
    ])
    mean_values = np.mean(trajectories, axis=0)

    fig, (ax_full, ax_early) = plt.subplots(1, 2, figsize=(12.5, 5.3), gridspec_kw={"width_ratios": [1.4, 1]})
    for ax in (ax_full, ax_early):
        for values in trajectories:
            ax.plot(recovery_k, values, color="#56B4E9", alpha=0.32, lw=1.2, marker="o", markersize=3)
        ax.plot(recovery_k, mean_values, color="#005F73", lw=3, marker="o", markersize=6, label="3-seed mean")
        ax.axhline(0.0, color="black", lw=1)
        ax.grid(alpha=0.22)
        ax.set_xlabel("Recovery step k")
        ax.spines[["top", "right"]].set_visible(False)
    ax_full.set_ylabel(r"$A^P(k)=\Delta_{256}^{P}(k)-\Delta_{32}^{P}(k)$ (BPC)")
    ax_full.set_title("A  Full recovery window", loc="left")
    ax_full.legend(frameon=False)
    ax_early.set_xlim(-2, 52)
    ax_early.set_title("B  Early-recovery zoom", loc="left")
    fig.suptitle("Figure 4 | Evaluation-horizon dependence collapses rapidly during recovery", fontsize=14)
    fig.tight_layout()
    _save(fig, output_dir, "figure_04_context_alignment.png")


def plot_paper_figures(
    tables: StudyTables,
    summary: Dict[str, Any],
    active_seeds: List[int],
    checkpoints: List[int],
    figures_dir: str,
) -> None:
    """Render the post-results paper figure set without changing formal results."""
    output_dir = os.path.join(figures_dir, "paper")
    os.makedirs(output_dir, exist_ok=True)
    recovery_steps, recovery_k = _recovery_axis(checkpoints, summary["base_pre_step"])

    _plot_recovery_dynamics(tables, active_seeds, recovery_steps, recovery_k, output_dir)
    _plot_primary_endpoint(summary, output_dir)
    _plot_horizon_effects(summary, output_dir)
    _plot_context_alignment(tables, active_seeds, recovery_steps, recovery_k, output_dir)
