"""
Analysis & Statistical Evaluation for Context-Length Recovery Study
Preregistered specification: Section 2, 3, 12, 13, 14, 15, 16
"""

import os
import sys
import json
import math
import numpy as np
from scipy import stats

from .config import (
    SEEDS,
    ARMS,
    CONTEXT_LENGTHS,
    BASE_RESULTS_DIR,
    THRESHOLD_MIN_PRE_GAP,
    THRESHOLD_RECOVERY_RATIO,
    THRESHOLD_RESIDUAL_PERSISTENT,
    THRESHOLD_PLATEAU_250,
)


def load_all_results(results_dir=BASE_RESULTS_DIR, seeds=None):
    """Load results for available or specified seeds."""
    if seeds is None:
        # Dynamically discover which seeds have complete results
        found_seeds = []
        for s in SEEDS:
            all_arms_exist = True
            for arm in ARMS:
                res_file = os.path.join(results_dir, f"seed_{s}", arm, "run_results.json")
                if not os.path.exists(res_file):
                    all_arms_exist = False
                    break
            if all_arms_exist:
                found_seeds.append(s)
        if not found_seeds:
            raise FileNotFoundError(f"No complete seed results found in {results_dir}")
        seeds = found_seeds
        
    data = {}
    for seed in seeds:
        data[seed] = {}
        for arm in ARMS:
            res_file = os.path.join(results_dir, f"seed_{seed}", arm, "run_results.json")
            if not os.path.exists(res_file):
                raise FileNotFoundError(f"Missing results file: {res_file}")
            with open(res_file, "r", encoding="utf-8") as f:
                data[seed][arm] = json.load(f)
    return data, seeds


def compute_paired_stats(differences):
    """
    Given an array of paired differences across seeds (e.g. df=2 for n=3):
    Compute mean, std, 95% paired t-interval, direction consistency.
    """
    diffs = np.array(differences, dtype=float)
    n = len(diffs)
    mean_d = float(np.mean(diffs))
    std_d = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    se_d = std_d / math.sqrt(n) if n > 1 else 0.0
    
    if n > 1:
        t_crit = stats.t.ppf(0.975, df=n - 1)
        ci_lower = mean_d - t_crit * se_d
        ci_upper = mean_d + t_crit * se_d
    else:
        ci_lower, ci_upper = mean_d, mean_d
        
    positive_count = int(np.sum(diffs > 0))
    
    return {
        "raw_diffs": diffs.tolist(),
        "mean": mean_d,
        "std": std_d,
        "se": se_d,
        "ci_95": [float(ci_lower), float(ci_upper)],
        "positive_count": positive_count,
        "total_count": n,
        "direction_consistency": f"{positive_count}/{n}",
    }


def compute_auc(steps, values):
    """Compute trapezoidal Area Under the Curve portably across NumPy versions."""
    if len(steps) < 2:
        return 0.0
    x = np.array(steps, dtype=float)
    y = np.array(values, dtype=float)
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) * 0.5))


def analyze_recovery_study(results_dir=BASE_RESULTS_DIR, generate_plots=True, seeds=None):
    """
    Comprehensive analysis pipeline fulfilling all preregistered requirements:
    1. Primary confirmatory contrast: Delta^A(0), Delta^A(500), absolute recovery, ratio R.
    2. Primary hypothesis test: R < 0.25 (75% recovery).
    3. Multi-horizon trajectories and recovery matrices: T=32, 64, 128, 256.
    4. Transition-local loss shocks (500->501, 1000->1001, 1500->1501, 2000->2001).
    5. Validation BPC AUC (pre-recovery vs recovery).
    6. Nonmonotonic exploratory contrasts: Delta_{N-A}(k) and Delta_{D-N}(k).
    7. Context-sensitive target analysis (C_i).
    8. Case A-E decision rule evaluation.
    9. Matplotlib figures.
    """
    data, active_seeds = load_all_results(results_dir, seeds=seeds)
    print(f"Loaded results for seeds: {active_seeds}")
    
    # Detect pre and post step indices from anchor logs
    sample_run = data[active_seeds[0]]["ascending"]
    anchor_steps = sorted([log["step"] for log in sample_run.get("anchor_logs", [])])
    
    if 2000 in anchor_steps and 2500 in anchor_steps:
        pre_step, post_step = 2000, 2500
        p250_step, p500_step = 2250, 2500
        is_smoke = False
    elif len(anchor_steps) >= 2:
        pre_step, post_step = anchor_steps[-2], anchor_steps[-1]
        p250_step, p500_step = pre_step, post_step
        is_smoke = True
    else:
        raise ValueError(f"Insufficient anchor steps found: {anchor_steps}")
        
    print(f"Analysis endpoints: Pre-recovery step={pre_step}, Post-recovery step={post_step}")
    
    # -------------------------------------------------------------
    # 1. Primary Endpoints (Anchor Panel)
    # -------------------------------------------------------------
    delta_pre_list = []
    delta_post_list = []
    delta_proc_250_list = []
    delta_proc_500_list = []
    
    horizon_anchor_pre = {h: [] for h in CONTEXT_LENGTHS}
    horizon_anchor_post = {h: [] for h in CONTEXT_LENGTHS}
    
    seed_details = {}
    
    def get_anchor_log(run_data, target_step):
        for a_log in run_data.get("anchor_logs", []):
            if a_log["step"] == target_step:
                return a_log
        raise ValueError(f"Step {target_step} missing from anchor logs")
        
    def get_process_log(run_data, target_step):
        for p_log in run_data.get("logs", []):
            if p_log["step"] == target_step:
                return p_log
        raise ValueError(f"Step {target_step} missing from process logs")

    for seed in active_seeds:
        asc_res = data[seed]["ascending"]
        desc_res = data[seed]["descending"]
        nonm_res = data[seed]["nonmonotonic"]
        
        asc_a_pre = get_anchor_log(asc_res, pre_step)
        desc_a_pre = get_anchor_log(desc_res, pre_step)
        nonm_a_pre = get_anchor_log(nonm_res, pre_step)
        
        asc_a_post = get_anchor_log(asc_res, post_step)
        desc_a_post = get_anchor_log(desc_res, post_step)
        nonm_a_post = get_anchor_log(nonm_res, post_step)
        
        d_pre = desc_a_pre["anchor_bpc"] - asc_a_pre["anchor_bpc"]
        d_post = desc_a_post["anchor_bpc"] - asc_a_post["anchor_bpc"]
        
        delta_pre_list.append(d_pre)
        delta_post_list.append(d_post)
        
        for h in CONTEXT_LENGTHS:
            h_str = str(h)
            h_val = h if h in desc_a_pre["horizon_bpc"] else h_str
            d_h_pre = desc_a_pre["horizon_bpc"][h_val] - asc_a_pre["horizon_bpc"][h_val]
            d_h_post = desc_a_post["horizon_bpc"][h_val] - asc_a_post["horizon_bpc"][h_val]
            horizon_anchor_pre[h].append(d_h_pre)
            horizon_anchor_post[h].append(d_h_post)
            
        asc_p250 = get_process_log(asc_res, p250_step)
        desc_p250 = get_process_log(desc_res, p250_step)
        asc_p500 = get_process_log(asc_res, p500_step)
        desc_p500 = get_process_log(desc_res, p500_step)
        
        d_p250 = desc_p250["process_bpc"] - asc_p250["process_bpc"]
        d_p500 = desc_p500["process_bpc"] - asc_p500["process_bpc"]
        
        delta_proc_250_list.append(d_p250)
        delta_proc_500_list.append(d_p500)
        
        seed_details[seed] = {
            "pre": {
                "ascending": asc_a_pre["anchor_bpc"],
                "descending": desc_a_pre["anchor_bpc"],
                "nonmonotonic": nonm_a_pre["anchor_bpc"],
                "delta_D_A": d_pre,
                "delta_N_A": nonm_a_pre["anchor_bpc"] - asc_a_pre["anchor_bpc"],
                "delta_D_N": desc_a_pre["anchor_bpc"] - nonm_a_pre["anchor_bpc"],
            },
            "post": {
                "ascending": asc_a_post["anchor_bpc"],
                "descending": desc_a_post["anchor_bpc"],
                "nonmonotonic": nonm_a_post["anchor_bpc"],
                "delta_D_A": d_post,
                "delta_N_A": nonm_a_post["anchor_bpc"] - asc_a_post["anchor_bpc"],
                "delta_D_N": desc_a_post["anchor_bpc"] - nonm_a_post["anchor_bpc"],
            },
            "plateau": {
                "delta_proc_250": d_p250,
                "delta_proc_500": d_p500,
                "slope_250": d_p250 - d_p500,
            }
        }
        
    stats_pre = compute_paired_stats(delta_pre_list)
    stats_post = compute_paired_stats(delta_post_list)
    
    abs_recovery_list = [pre - post for pre, post in zip(delta_pre_list, delta_post_list)]
    stats_abs_recovery = compute_paired_stats(abs_recovery_list)
    
    mean_pre = stats_pre["mean"]
    mean_post = stats_post["mean"]
    
    if abs(mean_pre) >= THRESHOLD_MIN_PRE_GAP:
        recovery_ratio_R = mean_post / mean_pre
        r_interpretable = True
        pct_recovered = (1.0 - recovery_ratio_R) * 100.0
    else:
        recovery_ratio_R = None
        r_interpretable = False
        pct_recovered = None
        
    mean_proc_250 = float(np.mean(delta_proc_250_list))
    mean_proc_500 = float(np.mean(delta_proc_500_list))
    plateau_diff = mean_proc_500 - mean_proc_250
    recovery_slope_250 = mean_proc_250 - mean_proc_500
    is_plateaued = abs(plateau_diff) < THRESHOLD_PLATEAU_250
    
    # -------------------------------------------------------------
    # 2. Four-Horizon Trajectories & Alignment Matrix
    # -------------------------------------------------------------
    checkpoints = [log["step"] for log in sample_run["logs"]]
    horizon_trajectories = {h: {"steps": checkpoints, "delta_mean": [], "delta_std": []} for h in CONTEXT_LENGTHS}
    
    for step in checkpoints:
        for h in CONTEXT_LENGTHS:
            h_diffs = []
            for seed in active_seeds:
                p_asc = get_process_log(data[seed]["ascending"], step)
                p_desc = get_process_log(data[seed]["descending"], step)
                h_val = h if h in p_desc["horizon_bpc"] else str(h)
                h_diffs.append(p_desc["horizon_bpc"][h_val] - p_asc["horizon_bpc"][h_val])
            horizon_trajectories[h]["delta_mean"].append(float(np.mean(h_diffs)))
            horizon_trajectories[h]["delta_std"].append(float(np.std(h_diffs, ddof=1)) if len(h_diffs) > 1 else 0.0)
            
    horizon_endpoint_stats = {
        h: {
            "pre": compute_paired_stats(horizon_anchor_pre[h]),
            "post": compute_paired_stats(horizon_anchor_post[h]),
        }
        for h in CONTEXT_LENGTHS
    }
    
    # -------------------------------------------------------------
    # 3. Transition-Local Loss Shock Analysis
    # -------------------------------------------------------------
    candidate_pairs = [(500, 501), (1000, 1001), (1500, 1501), (2000, 2001)]
    valid_pairs = [p for p in candidate_pairs if p[0] in checkpoints and p[1] in checkpoints]
    transition_shocks = {}
    
    for s_pre, s_post in valid_pairs:
        pair_key = f"{s_pre}->{s_post}"
        transition_shocks[pair_key] = {}
        for arm in ARMS:
            arm_shocks = []
            for seed in active_seeds:
                p1 = get_process_log(data[seed][arm], s_pre)["process_bpc"]
                p2 = get_process_log(data[seed][arm], s_post)["process_bpc"]
                arm_shocks.append(p2 - p1)
            transition_shocks[pair_key][arm] = {
                "mean_shock_bpc": float(np.mean(arm_shocks)),
                "raw_shocks": arm_shocks,
            }
            
    # -------------------------------------------------------------
    # 4. AUC Analysis
    # -------------------------------------------------------------
    split_step = pre_step
    pre_steps = [s for s in checkpoints if s <= split_step]
    rec_steps = [s for s in checkpoints if s >= split_step]
    
    auc_results = {}
    for arm in ARMS:
        pre_aucs = []
        rec_aucs = []
        for seed in active_seeds:
            bpc_pre = [get_process_log(data[seed][arm], s)["process_bpc"] for s in pre_steps]
            bpc_rec = [get_process_log(data[seed][arm], s)["process_bpc"] for s in rec_steps]
            pre_aucs.append(compute_auc(pre_steps, bpc_pre))
            rec_aucs.append(compute_auc(rec_steps, bpc_rec))
        auc_results[arm] = {
            "pre_recovery_auc": {"mean": float(np.mean(pre_aucs)), "raw": pre_aucs},
            "recovery_auc": {"mean": float(np.mean(rec_aucs)), "raw": rec_aucs},
        }
        
    # -------------------------------------------------------------
    # 5. Nonmonotonic Exploratory Contrasts
    # -------------------------------------------------------------
    nonm_pre_diffs_NA = [seed_details[s]["pre"]["delta_N_A"] for s in active_seeds]
    nonm_post_diffs_NA = [seed_details[s]["post"]["delta_N_A"] for s in active_seeds]
    nonm_pre_diffs_DN = [seed_details[s]["pre"]["delta_D_N"] for s in active_seeds]
    nonm_post_diffs_DN = [seed_details[s]["post"]["delta_D_N"] for s in active_seeds]
    
    nonmonotonic_analysis = {
        "N_minus_A": {
            "pre": compute_paired_stats(nonm_pre_diffs_NA),
            "post": compute_paired_stats(nonm_post_diffs_NA),
        },
        "D_minus_N": {
            "pre": compute_paired_stats(nonm_pre_diffs_DN),
            "post": compute_paired_stats(nonm_post_diffs_DN),
        }
    }
    
    # -------------------------------------------------------------
    # 6. Context-Sensitive Target Analysis (C_i)
    # -------------------------------------------------------------
    sensitivity_analysis = {}
    for s_step in [str(pre_step), str(post_step)]:
        if s_step in sample_run.get("sensitivity_logs", {}):
            sens_stats = {}
            for arm in ARMS:
                c_means = [data[s][arm]["sensitivity_logs"][s_step]["mean_c"] for s in active_seeds]
                c_fracs = [data[s][arm]["sensitivity_logs"][s_step]["frac_gt_0_1"] for s in active_seeds]
                sens_stats[arm] = {
                    "mean_c": float(np.mean(c_means)),
                    "frac_gt_0_1": float(np.mean(c_fracs)),
                }
            sensitivity_analysis[f"step_{s_step}"] = sens_stats
            
    # -------------------------------------------------------------
    # 7. Evaluate Preregistered Decision Rules (Section 16)
    # -------------------------------------------------------------
    decision = {}
    if abs(mean_pre) < THRESHOLD_MIN_PRE_GAP:
        decision["case"] = "Case A: pre-recovery gap does not replicate"
        decision["verdict"] = "STOP"
        decision["rationale"] = (
            f"|mean Delta^A(0)| = {abs(mean_pre):.4f} < {THRESHOLD_MIN_PRE_GAP}. "
            "Pre-recovery gap did not clearly replicate under matched-block design. Stop mechanism work."
        )
    elif abs(mean_post) <= THRESHOLD_MIN_PRE_GAP:
        decision["case"] = "Case B: gap is practically removed"
        decision["verdict"] = "STOP"
        decision["rationale"] = (
            f"|mean Delta^A(500)| = {abs(mean_post):.4f} <= {THRESHOLD_MIN_PRE_GAP}. "
            "Residual gap eliminated. Supports reversible terminal-context/recency explanation."
        )
    elif THRESHOLD_MIN_PRE_GAP < abs(mean_post) <= THRESHOLD_RESIDUAL_PERSISTENT or stats_post["positive_count"] not in [0, len(active_seeds)]:
        decision["case"] = "Case C: small residual is unresolved"
        decision["verdict"] = "UNRESOLVED"
        decision["rationale"] = (
            f"Residual mean Delta^A(500) = {mean_post:.4f} is between 0.02 and 0.03 BPC, "
            f"or seed directions disagree ({stats_post['direction_consistency']}). Do not claim persistent effect."
        )
    elif mean_post > THRESHOLD_RESIDUAL_PERSISTENT and recovery_slope_250 > THRESHOLD_PLATEAU_250:
        decision["case"] = "Case D: gap remains large but is still recovering"
        decision["verdict"] = "EXTEND_RECOVERY"
        decision["rationale"] = (
            f"mean Delta^A(500) = {mean_post:.4f} > {THRESHOLD_RESIDUAL_PERSISTENT} "
            f"and slope Delta^P(250) - Delta^P(500) = {recovery_slope_250:.4f} > {THRESHOLD_PLATEAU_250}. "
            "Recovery is active but incomplete. Trigger one-time 500-step extension to step 3000 via: "
            "`python -m training_dynamics_research.recovery_study.runner --extend`"
        )
    elif (
        mean_post > THRESHOLD_RESIDUAL_PERSISTENT
        and stats_post["positive_count"] == len(active_seeds)
        and is_plateaued
    ):
        decision["case"] = "Case E: persistent-effect candidate"
        decision["verdict"] = "GO_OPTIMIZER_EXPERIMENT"
        decision["rationale"] = (
            f"mean Delta^A(500) = {mean_post:.4f} > {THRESHOLD_RESIDUAL_PERSISTENT}, "
            f"{stats_post['positive_count']}/{len(active_seeds)} seeds positive, and plateau condition satisfied (|{plateau_diff:.4f}| < {THRESHOLD_PLATEAU_250}). "
            "Candidate evidence for persistent path dependence; motivates targeted optimizer-state experiment."
        )
    else:
        decision["case"] = "Case C: unresolved / boundary edge case"
        decision["verdict"] = "UNRESOLVED"
        decision["rationale"] = "Pattern falls into boundary conditions. Report raw data without strong claims."
        
    summary = {
        "active_seeds": active_seeds,
        "is_smoke_test": is_smoke,
        "pre_step": pre_step,
        "post_step": post_step,
        "stats_pre": stats_pre,
        "stats_post": stats_post,
        "stats_abs_recovery": stats_abs_recovery,
        "recovery_ratio_R": recovery_ratio_R,
        "pct_recovered": pct_recovered,
        "r_interpretable": r_interpretable,
        "primary_hypothesis_satisfied": bool(recovery_ratio_R is not None and recovery_ratio_R < THRESHOLD_RECOVERY_RATIO),
        "recovery_slope_250": recovery_slope_250,
        "is_plateaued": is_plateaued,
        "decision": decision,
        "horizon_endpoint_stats": horizon_endpoint_stats,
        "transition_shocks": transition_shocks,
        "auc_results": auc_results,
        "nonmonotonic_analysis": nonmonotonic_analysis,
        "sensitivity_analysis": sensitivity_analysis,
        "seed_details": seed_details,
    }
    
    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved comprehensive summary to {summary_path}")
    
    # -------------------------------------------------------------
    # 8. Matplotlib Plots Generation
    # -------------------------------------------------------------
    if generate_plots:
        try:
            import matplotlib.pyplot as plt
            plots_dir = os.path.join(results_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # Plot 1: Recovery gap trajectories across horizons
            plt.figure(figsize=(10, 6))
            steps_arr = np.array(checkpoints)
            colors = {32: "#9467bd", 64: "#2ca02c", 128: "#ff7f0e", 256: "#1f77b4"}
            for h in CONTEXT_LENGTHS:
                y = np.array(horizon_trajectories[h]["delta_mean"])
                err = np.array(horizon_trajectories[h]["delta_std"])
                plt.plot(steps_arr, y, label=f"T={h}", color=colors[h], lw=2)
                if len(active_seeds) > 1:
                    plt.fill_between(steps_arr, y - err, y + err, color=colors[h], alpha=0.15)
                
            plt.axvline(x=pre_step, color="gray", linestyle="--", label=f"Recovery Onset (Step {pre_step})")
            plt.axhline(y=0.0, color="black", linestyle=":", lw=1)
            plt.axhline(y=0.03, color="red", linestyle=":", label="Persistent Threshold (0.03 BPC)")
            plt.axhline(y=0.02, color="orange", linestyle=":", label="Minimum Pre-gap (0.02 BPC)")
            plt.title("Context Horizon Recovery Trajectories Δ_h(k) [Descending − Ascending]")
            plt.xlabel("Global Step")
            plt.ylabel("Paired Δ BPC")
            plt.legend(loc="upper right")
            plt.grid(True, alpha=0.3)
            p1_path = os.path.join(plots_dir, "recovery_horizon_trajectories.png")
            plt.savefig(p1_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"Generated plot: {p1_path}")
            
            # Plot 2: Per-seed pre vs post endpoint differences
            plt.figure(figsize=(8, 5))
            x_pts = [0, 1]
            for s_idx, s in enumerate(active_seeds):
                pre_v = seed_details[s]["pre"]["delta_D_A"]
                post_v = seed_details[s]["post"]["delta_D_A"]
                plt.plot(x_pts, [pre_v, post_v], marker="o", label=f"Seed {s}", lw=2)
            plt.axhline(y=0.0, color="black", linestyle=":", lw=1)
            plt.axhline(y=0.03, color="red", linestyle=":", alpha=0.5, label="0.03 BPC")
            plt.xticks(x_pts, [f"Pre-Recovery (Step {pre_step})", f"Post-Recovery (Step {post_step})"])
            plt.ylabel("Paired Δ^A BPC (Descending − Ascending)")
            plt.title("Pre-to-Post Recovery Paired Gap by Seed")
            plt.legend()
            plt.grid(True, alpha=0.3)
            p2_path = os.path.join(plots_dir, "paired_seed_recovery.png")
            plt.savefig(p2_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"Generated plot: {p2_path}")
            
        except Exception as e:
            print(f"Note: Plot generation skipped or encountered issue: {e}")
            
    # Terminal Display
    print("\n" + "=" * 68)
    print("CONTEXT-LENGTH RECOVERY STUDY: PREREGISTERED ANALYSIS SUMMARY")
    print("=" * 68)
    print(f"Active Seeds:           {active_seeds} ({'Smoke Test' if is_smoke else 'Formal Runs'})")
    print(f"Pre-recovery Δ^A(0):    {stats_pre['mean']:+.4f} ± {stats_pre['std']:.4f} BPC  (95% CI: [{stats_pre['ci_95'][0]:+.4f}, {stats_pre['ci_95'][1]:+.4f}])")
    print(f"Post-recovery Δ^A(500):  {stats_post['mean']:+.4f} ± {stats_post['std']:.4f} BPC  (95% CI: [{stats_post['ci_95'][0]:+.4f}, {stats_post['ci_95'][1]:+.4f}])")
    print(f"Absolute Recovered:     {stats_abs_recovery['mean']:+.4f} ± {stats_abs_recovery['std']:.4f} BPC")
    if r_interpretable:
        print(f"Recovery Ratio R:       {recovery_ratio_R:.3f} ({pct_recovered:.1f}% recovered) | Criterion R < 0.25: {recovery_ratio_R < THRESHOLD_RECOVERY_RATIO}")
    else:
        print(f"Recovery Ratio R:       N/A (|pre-gap| < {THRESHOLD_MIN_PRE_GAP})")
    print(f"Direction consistency:  {stats_post['direction_consistency']}")
    print(f"Plateau condition:      {'SATISFIED' if is_plateaued else 'ACTIVE RECOVERY'} (slope={recovery_slope_250:+.4f} BPC / 250 steps)")
    print("-" * 68)
    print("FOUR-HORIZON ENDPOINT MATRIX (Anchor Panel):")
    for h in CONTEXT_LENGTHS:
        pre_h = horizon_endpoint_stats[h]["pre"]["mean"]
        post_h = horizon_endpoint_stats[h]["post"]["mean"]
        print(f"  T={h:3d} | Pre: {pre_h:+.4f} BPC | Post: {post_h:+.4f} BPC | Recovered: {pre_h - post_h:+.4f} BPC")
    print("-" * 68)
    print(f"DECISION: {decision['case']}")
    print(f"VERDICT:  {decision['verdict']}")
    print(f"REASON:   {decision['rationale']}")
    print("=" * 68 + "\n")
    
    return summary


if __name__ == "__main__":
    analyze_recovery_study()
