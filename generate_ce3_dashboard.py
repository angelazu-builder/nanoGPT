"""
generate_ce3_dashboard.py
Generates the CE-3 corrected 6-panel dashboard from 5-seed paired experiment results.
Panels:
  [0] Training curves (BPC over steps) — mean ± std, 4 arms, seeds 43-46
  [1] Final BPC bar chart — mean ± std, all 5 seeds, with p-value annotations
  [2] Attention entropy over training — mean ± std, seeds 43-46
  [3] Effective rank over training — mean ± std, seeds 43-46
  [4] Context sensitivity (context_bpc_fixed) at step 2500 — per arm
  [5] Gradient noise scaling Tr(Σ) vs T — initialization probe (unchanged)
"""

import json, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ── Data loading ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE, "results", "ce3_paired")

ARMS = ["curriculum", "shuffled", "anti_curriculum", "fixed_long"]
ARM_LABELS = {
    "curriculum":     "Curriculum (32→256)",
    "shuffled":       "Shuffled Control",
    "anti_curriculum":"Anti-Curriculum (256→32)",
    "fixed_long":     "Fixed-Long (256)",
}
COLORS = {
    "curriculum":      "#4C9BE8",   # blue
    "shuffled":        "#57C785",   # green
    "anti_curriculum": "#E8584C",   # red
    "fixed_long":      "#C07EE8",   # purple
}
SEEDS_WITH_LOGS = [43, 44, 45, 46]   # seed 42 first 3 arms missing logs
ALL_SEEDS = [42, 43, 44, 45, 46]

all_data = {}
for seed in ALL_SEEDS:
    with open(os.path.join(RESULTS_DIR, f"seed_{seed}_results.json")) as f:
        all_data[seed] = json.load(f)

with open(os.path.join(RESULTS_DIR, "ce3_summary.json")) as f:
    summary = json.load(f)

# ── Helper: extract log field across seeds ────────────────────────────────────
def extract_curve(field, seeds=SEEDS_WITH_LOGS):
    """Returns dict arm -> (steps, mean_curve, std_curve)"""
    out = {}
    for arm in ARMS:
        curves = []
        for s in seeds:
            d = all_data[s].get(arm, {})
            if "logs" not in d:
                continue
            vals = [row[field] for row in d["logs"]]
            curves.append(vals)
        if not curves:
            continue
        arr = np.array(curves)
        steps_raw = [row["step"] for row in all_data[seeds[0]][arm]["logs"]]
        out[arm] = (steps_raw, arr.mean(0), arr.std(0, ddof=1))
    return out

# ── Gradient probe data (from original study, unaffected by bugs) ─────────────
GRAD_DATA = {
    "T":        [32,   64,    128,   256],
    "Tr_Sigma": [0.838, 1.078, 1.267, 1.633],
    "B_crit":   [0.03,  0.03,  0.03,  0.03],
}

# ── Context sensitivity at final step ─────────────────────────────────────────
def extract_context_sensitivity(seeds=SEEDS_WITH_LOGS):
    ctx_lengths = [32, 64, 128, 256]
    out = {}
    for arm in ARMS:
        per_ctx = {c: [] for c in ctx_lengths}
        for s in seeds:
            d = all_data[s].get(arm, {})
            if "logs" not in d:
                continue
            last = d["logs"][-1]
            for c in ctx_lengths:
                per_ctx[c].append(last["context_bpc_fixed"][str(c)])
        out[arm] = {c: (np.mean(v), np.std(v, ddof=1)) for c, v in per_ctx.items() if v}
    return out

# ── Build figure ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor("#0F1117")

TITLE_COLOR = "#E8EAF0"
GRID_COLOR  = "#2A2D3A"
TICK_COLOR  = "#9BA3B5"
PANEL_BG    = "#161A24"

gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.35,
                      left=0.06, right=0.97, top=0.90, bottom=0.08)
axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color=TITLE_COLOR, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, color=TICK_COLOR, fontsize=8)
    ax.set_ylabel(ylabel, color=TICK_COLOR, fontsize=8)
    ax.tick_params(colors=TICK_COLOR, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.7)

# ── Panel 0: Training curves BPC ─────────────────────────────────────────────
ax = axes[0]
style_ax(ax, "Panel A — Validation BPC During Training\n(mean ± std, seeds 43–46)", "Step", "BPC")
bpc_curves = extract_curve("val_bpc")
for arm, (steps, mean, std) in bpc_curves.items():
    c = COLORS[arm]
    ax.plot(steps, mean, color=c, linewidth=1.8, label=ARM_LABELS[arm])
    ax.fill_between(steps, mean - std, mean + std, color=c, alpha=0.15)
ax.legend(fontsize=6.5, loc="upper right",
          facecolor="#1E2130", edgecolor=GRID_COLOR, labelcolor=TITLE_COLOR)

# ── Panel 1: Final BPC bar chart with stats ───────────────────────────────────
ax = axes[1]
style_ax(ax, "Panel B — Final BPC (Step 2500)\n5-Seed Mean ± Std | Paired t-test vs Curriculum", "Arm", "BPC")
means = [summary["means"][a] for a in ARMS]
stds  = [summary["stds"][a]  for a in ARMS]
x     = np.arange(len(ARMS))
bars  = ax.bar(x, means, yerr=stds, capsize=5,
               color=[COLORS[a] for a in ARMS],
               error_kw={"ecolor": TITLE_COLOR, "linewidth": 1.5},
               width=0.55, zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(["Curriculum", "Shuffled", "Anti-Curric.", "Fixed-Long"],
                   fontsize=7, color=TICK_COLOR)
ax.set_ylim(1.95, 2.45)
ax.axhline(summary["means"]["curriculum"], color=COLORS["curriculum"],
           linestyle="--", linewidth=1.0, alpha=0.5)

# Annotate p-values
curr_vals = summary["bpc_table"]["curriculum"]
annots = {
    "shuffled":       ("p=0.643\nn.s.", "#57C785"),
    "anti_curriculum":("p=0.0001\n[sig.]", "#E8584C"),
    "fixed_long":     ("p=0.886\nn.s.", "#C07EE8"),
}
for i, arm in enumerate(ARMS):
    if arm == "curriculum": continue
    txt, col = annots[arm]
    ax.text(i, means[i] + stds[i] + 0.012, txt,
            ha="center", va="bottom", fontsize=6, color=col, fontweight="bold")
for bar, mean_v in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, mean_v - 0.02,
            f"{mean_v:.4f}", ha="center", va="top", fontsize=6.5, color="white")

# ── Panel 2: Attention entropy over training ──────────────────────────────────
ax = axes[2]
style_ax(ax, "Panel C — Attention Entropy During Training\n(mean ± std, seeds 43–46)", "Step", "Mean Attn Entropy")
ent_curves = extract_curve("mean_entropy")
for arm, (steps, mean, std) in ent_curves.items():
    c = COLORS[arm]
    ax.plot(steps, mean, color=c, linewidth=1.8, label=ARM_LABELS[arm])
    ax.fill_between(steps, mean - std, mean + std, color=c, alpha=0.15)
ax.legend(fontsize=6.5, loc="upper right",
          facecolor="#1E2130", edgecolor=GRID_COLOR, labelcolor=TITLE_COLOR)

# ── Panel 3: Effective rank over training ─────────────────────────────────────
ax = axes[3]
style_ax(ax, "Panel D — Effective Representation Rank\n(mean ± std, seeds 43–46)", "Step", "Mean Rank")
rank_curves = extract_curve("mean_rank")
for arm, (steps, mean, std) in rank_curves.items():
    c = COLORS[arm]
    ax.plot(steps, mean, color=c, linewidth=1.8, label=ARM_LABELS[arm])
    ax.fill_between(steps, mean - std, mean + std, color=c, alpha=0.15)
ax.legend(fontsize=6.5, loc="lower right",
          facecolor="#1E2130", edgecolor=GRID_COLOR, labelcolor=TITLE_COLOR)

# ── Panel 4: Context sensitivity (fixed-target BPC vs context horizon) ─────────
ax = axes[4]
style_ax(ax, "Panel E — Context Sensitivity at Step 2500\n(fixed last-32 scoring, seeds 43–46)", "Context Horizon (tokens)", "BPC")
ctx_data = extract_context_sensitivity()
ctx_lengths = [32, 64, 128, 256]
for arm in ARMS:
    if arm not in ctx_data: continue
    means_c = [ctx_data[arm][c][0] for c in ctx_lengths]
    stds_c  = [ctx_data[arm][c][1] for c in ctx_lengths]
    c = COLORS[arm]
    ax.plot(ctx_lengths, means_c, color=c, linewidth=1.8, marker="o",
            markersize=5, label=ARM_LABELS[arm])
    ax.fill_between(ctx_lengths,
                    np.array(means_c) - np.array(stds_c),
                    np.array(means_c) + np.array(stds_c),
                    color=c, alpha=0.15)
ax.set_xscale("log", base=2)
ax.set_xticks(ctx_lengths)
ax.set_xticklabels(ctx_lengths, fontsize=7)
ax.legend(fontsize=6.5, loc="upper right",
          facecolor="#1E2130", edgecolor=GRID_COLOR, labelcolor=TITLE_COLOR)

# ── Panel 5: Gradient noise scaling ──────────────────────────────────────────
ax = axes[5]
style_ax(ax, "Panel F — Gradient Noise Scaling at Init.\n(Tr(Σ) vs Context Length T, B×T=4096)", "Context Length T", "Grad Variance Tr(Σ)")
T_vals   = GRAD_DATA["T"]
Tr_vals  = GRAD_DATA["Tr_Sigma"]
ax.plot(T_vals, Tr_vals, color="#F5A623", linewidth=2.2, marker="D",
        markersize=7, label="Tr(Σ) measured")
ax.fill_between(T_vals, Tr_vals, alpha=0.12, color="#F5A623")
for t, tr in zip(T_vals, Tr_vals):
    ax.annotate(f"T={t}\n{tr:.3f}", (t, tr),
                textcoords="offset points", xytext=(4, 6),
                fontsize=7, color=TITLE_COLOR)
ax.set_xscale("log", base=2)
ax.set_xticks(T_vals)
ax.set_xticklabels(T_vals, fontsize=7)
ax.text(0.05, 0.90, "B_crit ≈ 0.03\n(stable across T)",
        transform=ax.transAxes, fontsize=7.5,
        color="#F5A623", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1E2130", edgecolor=GRID_COLOR))
ax.legend(fontsize=7, facecolor="#1E2130", edgecolor=GRID_COLOR, labelcolor=TITLE_COLOR)

# ── Super title ───────────────────────────────────────────────────────────────
fig.suptitle(
    "CE-3 Corrected Training Dynamics Dashboard  |  5 Seeds × 4 Arms  |  2,500 Steps  |  Tiny Shakespeare\n"
    "Curriculum ≈ Shuffled ≈ Fixed-Long (p > 0.6)  |  Anti-Curriculum degrades significantly (Δ=+0.178 BPC, p=0.0001)",
    color=TITLE_COLOR, fontsize=11, fontweight="bold", y=0.97
)

out_path = os.path.join(BASE, "results", "ce3_corrected_dashboard.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out_path}")
