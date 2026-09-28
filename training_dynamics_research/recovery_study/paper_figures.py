"""Publication figures built from the frozen recovery-study artifacts.

This module is a post-results presentation layer. It changes neither the
registered estimands nor the decision rules, and it preserves the registered
contract figures in the parent output directory.
"""

import os
from typing import Any, Dict, List, Sequence, Tuple

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
ALIGNMENT_COLOR = "#006D77"
NONMONOTONIC_COLORS = ("#CC79A7", "#0072B2")
RAW_ALPHA = 0.28


def _set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, output_dir: str, stem: str) -> None:
    """Save a raster preview and a vector manuscript artifact."""
    fig.savefig(os.path.join(output_dir, f"{stem}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)


def _recovery_axis(checkpoints: List[int], pre_step: int) -> Tuple[List[int], List[int]]:
    steps = [step for step in checkpoints if step >= pre_step]
    return steps, [step - pre_step for step in steps]


def _process_gap(
    tables: StudyTables,
    seed: int,
    steps: Sequence[int],
    horizon: int,
) -> np.ndarray:
    return np.asarray(
        [
            tables.process_bpc(seed, "descending", step, horizon)
            - tables.process_bpc(seed, "ascending", step, horizon)
            for step in steps
        ]
    )


def _trajectories(
    tables: StudyTables,
    active_seeds: Sequence[int],
    steps: Sequence[int],
    horizon: int,
) -> np.ndarray:
    return np.asarray([_process_gap(tables, seed, steps, horizon) for seed in active_seeds])


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
    )


def _clean_axis(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.grid(axis=grid_axis, alpha=0.18, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def _draw_trajectory(
    ax: plt.Axes,
    x: Sequence[int],
    values: np.ndarray,
    color: str,
    show_legend: bool = False,
) -> np.ndarray:
    for row in values:
        ax.plot(x, row, color=color, alpha=RAW_ALPHA, lw=0.9, marker="o", markersize=2.5)
    mean_values = np.mean(values, axis=0)
    ax.plot(
        x,
        mean_values,
        color=color,
        lw=2.2,
        marker="o",
        markersize=4.5,
        label="Mean" if show_legend else None,
    )
    ax.axhline(0.0, color="black", lw=0.8)
    _clean_axis(ax)
    return mean_values


def _draw_paired_endpoints(ax: plt.Axes, summary: Dict[str, Any]) -> None:
    pre = summary["stats_pre"]
    post = summary["stats_post_500"]
    ax.axhspan(
        -THRESHOLD_MIN_PRE_GAP,
        THRESHOLD_MIN_PRE_GAP,
        color="0.65",
        alpha=0.18,
        label=rf"Practical-removal band ($\pm${THRESHOLD_MIN_PRE_GAP:.2f})",
    )
    for pre_value, post_value in zip(pre["raw_diffs"], post["raw_diffs"]):
        ax.plot([0, 1], [pre_value, post_value], color="0.60", lw=1.0, zorder=1)
        ax.scatter(0, pre_value, color=PRE_COLOR, s=25, zorder=2)
        ax.scatter(1, post_value, color=POST_COLOR, s=25, zorder=2)
    ax.scatter(
        [0, 1],
        [pre["mean"], post["mean"]],
        marker="D",
        color="black",
        s=35,
        zorder=3,
        label="Mean",
    )
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks([0, 1], ["Before\n$k=0$", "After\n$k=500$"])
    ax.set_ylabel(r"Anchor gap $\Delta^A_{256}$ (BPC)")
    _clean_axis(ax, "y")


def _draw_post_effect(ax: plt.Axes, summary: Dict[str, Any]) -> None:
    stats = summary["stats_post_500"]
    ax.axvspan(
        -THRESHOLD_MIN_PRE_GAP,
        THRESHOLD_MIN_PRE_GAP,
        color="0.65",
        alpha=0.18,
    )
    raw = np.asarray(stats["raw_diffs"])
    y = np.linspace(-0.08, 0.08, len(raw))
    ax.scatter(raw, y, color=POST_COLOR, alpha=0.70, s=24, zorder=2)
    low, high = stats["ci_95"]
    ax.errorbar(
        stats["mean"],
        0,
        xerr=[[stats["mean"] - low], [high - stats["mean"]]],
        fmt="D",
        color="black",
        ecolor=POST_COLOR,
        elinewidth=2.0,
        capsize=4,
        markersize=5,
        zorder=3,
    )
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_xlim(low - 0.035, high + 0.035)
    ax.set_ylim(-0.18, 0.22)
    ax.set_yticks([])
    ax.set_xlabel("Post-recovery effect (BPC)")
    ax.text(
        stats["mean"],
        0.15,
        rf"$\bar{{\Delta}}={stats['mean']:+.3f}$",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    _clean_axis(ax, "x")


def _plot_main_primary(
    tables: StudyTables,
    summary: Dict[str, Any],
    active_seeds: List[int],
    recovery_steps: List[int],
    recovery_k: List[int],
    output_dir: str,
) -> None:
    """Main Figure 1: dynamics, paired endpoints, and residual uncertainty."""
    fig = plt.figure(figsize=(7.2, 5.4))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.55, wspace=0.42)
    ax_dynamics = fig.add_subplot(grid[0, :])
    ax_pairs = fig.add_subplot(grid[1, 0])
    ax_effect = fig.add_subplot(grid[1, 1])

    values = _trajectories(tables, active_seeds, recovery_steps, 256)
    mean_values = _draw_trajectory(
        ax_dynamics,
        recovery_k,
        values,
        HORIZON_COLORS[256],
    )
    ax_dynamics.set_xlabel("Recovery step $k$")
    ax_dynamics.set_ylabel(r"Process gap $\Delta^P_{256}(k)$ (BPC)")
    _panel_label(ax_dynamics, "A")

    for k in (0, 10, 50):
        idx = recovery_k.index(k)
        ax_dynamics.annotate(
            f"{mean_values[idx]:+.2f}",
            (k, mean_values[idx]),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=7.5,
            color=HORIZON_COLORS[256],
        )

    inset = ax_dynamics.inset_axes([0.52, 0.29, 0.43, 0.58])
    early = [idx for idx, k in enumerate(recovery_k) if k <= 50]
    _draw_trajectory(
        inset,
        [recovery_k[idx] for idx in early],
        values[:, early],
        HORIZON_COLORS[256],
    )
    inset.set_xlim(-2, 52)
    inset.set_title(r"Early recovery ($k\leq50$)", fontsize=8, pad=3)
    inset.tick_params(labelsize=7)
    inset.set_xlabel("$k$", fontsize=7.5)

    _draw_paired_endpoints(ax_pairs, summary)
    _panel_label(ax_pairs, "B")
    _draw_post_effect(ax_effect, summary)
    _panel_label(ax_effect, "C")
    _save(fig, output_dir, "main_figure_1_primary_recovery")


def _plot_main_horizon_effects(summary: Dict[str, Any], output_dir: str) -> None:
    """Main Figure 2: horizon-specific anchor effects before and after recovery."""
    horizon_stats = summary["horizon_endpoint_stats"]
    horizons = list(CONTEXT_LENGTHS)
    fig, (ax_pre, ax_post) = plt.subplots(1, 2, figsize=(7.2, 3.25), sharey=True)

    for ax, endpoint in [(ax_pre, "pre"), (ax_post, "post_500")]:
        for y, horizon in enumerate(horizons):
            stats = horizon_stats.get(str(horizon), horizon_stats.get(horizon))[endpoint]
            raw = np.asarray(stats["raw_diffs"])
            jitter = np.linspace(-0.07, 0.07, len(raw))
            ax.scatter(raw, y + jitter, color=HORIZON_COLORS[horizon], alpha=0.65, s=22, zorder=2)
            low, high = stats["ci_95"]
            ax.errorbar(
                stats["mean"],
                y,
                xerr=[[stats["mean"] - low], [high - stats["mean"]]],
                fmt="D",
                color="black",
                ecolor=HORIZON_COLORS[horizon],
                elinewidth=1.8,
                capsize=3.5,
                markersize=4.5,
                zorder=3,
            )
        ax.axvline(0.0, color="black", lw=0.8)
        ax.set_yticks(range(len(horizons)), [f"$T={h}$" for h in horizons])
        ax.set_xlabel("Descending $-$ ascending anchor BPC")
        _clean_axis(ax, "x")

    ax_post.axvspan(
        -THRESHOLD_MIN_PRE_GAP,
        THRESHOLD_MIN_PRE_GAP,
        color="0.65",
        alpha=0.18,
    )
    ax_pre.set_title("Before recovery ($k=0$)", loc="left", fontsize=9, pad=7)
    ax_post.set_title("After recovery ($k=500$)", loc="left", fontsize=9, pad=7)
    _panel_label(ax_pre, "A")
    _panel_label(ax_post, "B")
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.20, top=0.93, wspace=0.12)
    _save(fig, output_dir, "main_figure_2_horizon_specificity")


def _alignment_values(
    tables: StudyTables,
    seed: int,
    recovery_steps: Sequence[int],
) -> np.ndarray:
    delta_256 = _process_gap(tables, seed, recovery_steps, 256)
    delta_32 = _process_gap(tables, seed, recovery_steps, 32)
    return delta_256 - delta_32


def _plot_main_alignment(
    tables: StudyTables,
    active_seeds: List[int],
    recovery_steps: List[int],
    recovery_k: List[int],
    output_dir: str,
) -> None:
    """Main Figure 3: full and early context-alignment diagnostic."""
    values = np.asarray(
        [_alignment_values(tables, seed, recovery_steps) for seed in active_seeds]
    )
    fig, (ax_full, ax_early) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.1),
        gridspec_kw={"width_ratios": [1.35, 1]},
    )
    for ax in (ax_full, ax_early):
        _draw_trajectory(ax, recovery_k, values, ALIGNMENT_COLOR)
        ax.set_xlabel("Recovery step $k$")
    ax_full.set_ylabel(r"Horizon alignment $A^P(k)$ (BPC)")
    ax_early.set_xlim(-2, 52)
    ax_full.set_title("Full recovery window", loc="left", fontsize=9, pad=7)
    ax_early.set_title("First 50 steps", loc="left", fontsize=9, pad=7)
    _panel_label(ax_full, "A")
    _panel_label(ax_early, "B")
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.20, top=0.93, wspace=0.32)
    _save(fig, output_dir, "main_figure_3_context_alignment")


def _plot_appendix_all_horizons(
    tables: StudyTables,
    active_seeds: List[int],
    recovery_steps: List[int],
    recovery_k: List[int],
    output_dir: str,
) -> None:
    """Appendix Figure A1: complete process trajectories at all horizons."""
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True)
    short_values: List[float] = []

    for idx, horizon in enumerate(CONTEXT_LENGTHS):
        ax = axes[idx // 2, idx % 2]
        values = _trajectories(tables, active_seeds, recovery_steps, horizon)
        if horizon != 256:
            short_values.extend(values.ravel().tolist())
        _draw_trajectory(ax, recovery_k, values, HORIZON_COLORS[horizon], show_legend=idx == 0)
        ax.text(0.04, 0.92, f"$T={horizon}$", transform=ax.transAxes, va="top")
        if idx // 2 == 1:
            ax.set_xlabel("Recovery step $k$")
        if idx % 2 == 0:
            ax.set_ylabel("Descending $-$ ascending BPC")
        _panel_label(ax, chr(ord("A") + idx))
        if idx == 0:
            ax.legend(frameon=False, loc="lower right")

    lower = min(short_values) - 0.02
    upper = max(short_values) + 0.02
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_ylim(lower, upper)
    axes[1, 1].text(
        0.96,
        0.80,
        "Independent y-scale",
        transform=axes[1, 1].transAxes,
        ha="right",
        color="0.35",
        fontsize=7.5,
    )
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.11, top=0.96, hspace=0.27, wspace=0.25)
    _save(fig, output_dir, "appendix_figure_A1_all_horizon_trajectories")


def _plot_appendix_nonmonotonic(
    tables: StudyTables,
    active_seeds: List[int],
    recovery_steps: List[int],
    recovery_k: List[int],
    output_dir: str,
) -> None:
    """Appendix Figure A2: exploratory contrasts for the single nonmonotonic arm."""
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True)
    contrasts = [
        ("Nonmonotonic $-$ ascending", "nonmonotonic", "ascending", NONMONOTONIC_COLORS[0]),
        ("Descending $-$ nonmonotonic", "descending", "nonmonotonic", NONMONOTONIC_COLORS[1]),
    ]

    for idx, horizon in enumerate(CONTEXT_LENGTHS):
        ax = axes[idx // 2, idx % 2]
        for label, first_arm, second_arm, color in contrasts:
            values = np.asarray(
                [
                    [
                        tables.process_bpc(seed, first_arm, step, horizon)
                        - tables.process_bpc(seed, second_arm, step, horizon)
                        for step in recovery_steps
                    ]
                    for seed in active_seeds
                ]
            )
            for row in values:
                ax.plot(
                    recovery_k,
                    row,
                    color=color,
                    alpha=0.20,
                    lw=0.8,
                    marker="o",
                    markersize=2,
                )
            ax.plot(
                recovery_k,
                np.mean(values, axis=0),
                color=color,
                lw=2.0,
                marker="o",
                markersize=3.8,
                label=label,
            )
        ax.axhline(0.0, color="black", lw=0.8)
        ax.text(0.04, 0.92, f"$T={horizon}$", transform=ax.transAxes, va="top")
        if idx // 2 == 1:
            ax.set_xlabel("Recovery step $k$")
        if idx % 2 == 0:
            ax.set_ylabel("Process contrast (BPC)")
        _panel_label(ax, chr(ord("A") + idx))
        _clean_axis(ax)
        if idx == 0:
            ax.legend(frameon=False, loc="best", fontsize=7)

    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.11, top=0.96, hspace=0.27, wspace=0.25)
    _save(fig, output_dir, "appendix_figure_A2_nonmonotonic_exploratory")


def plot_paper_figures(
    tables: StudyTables,
    summary: Dict[str, Any],
    active_seeds: List[int],
    checkpoints: List[int],
    figures_dir: str,
) -> None:
    """Render the post-results manuscript figure set."""
    _set_publication_style()
    output_dir = os.path.join(figures_dir, "paper")
    os.makedirs(output_dir, exist_ok=True)
    recovery_steps, recovery_k = _recovery_axis(checkpoints, summary["base_pre_step"])

    _plot_main_primary(
        tables,
        summary,
        active_seeds,
        recovery_steps,
        recovery_k,
        output_dir,
    )
    _plot_main_horizon_effects(summary, output_dir)
    _plot_main_alignment(
        tables,
        active_seeds,
        recovery_steps,
        recovery_k,
        output_dir,
    )
    _plot_appendix_all_horizons(
        tables,
        active_seeds,
        recovery_steps,
        recovery_k,
        output_dir,
    )
    _plot_appendix_nonmonotonic(
        tables,
        active_seeds,
        recovery_steps,
        recovery_k,
        output_dir,
    )
