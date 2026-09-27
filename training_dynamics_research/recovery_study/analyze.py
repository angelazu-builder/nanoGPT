"""
Analysis & Statistical Evaluation for Context-Length Recovery Study
Preregistered specification: Section 2, 3, 14, 15, 16
"""

import os
import json
import math
import numpy as np
from scipy import stats
from .config import (
    SEEDS,
    ARMS,
    BASE_RESULTS_DIR,
    THRESHOLD_MIN_PRE_GAP,
    THRESHOLD_RECOVERY_RATIO,
    THRESHOLD_RESIDUAL_PERSISTENT,
    THRESHOLD_PLATEAU_250,
)


def load_all_results(results_dir=BASE_RESULTS_DIR):
    """Load results for all seeds and arms."""
    data = {}
    for seed in SEEDS:
        data[seed] = {}
        for arm in ARMS:
            res_file = os.path.join(results_dir, f"seed_{seed}", arm, "run_results.json")
            if not os.path.exists(res_file):
                raise FileNotFoundError(f"Missing results file: {res_file}")
            with open(res_file, "r", encoding="utf-8") as f:
                data[seed][arm] = json.load(f)
    return data


def compute_paired_stats(differences):
    """
    Given an array of paired differences across seeds (e.g. df=2 for n=3):
    Compute mean, std, 95% paired t-interval.
    """
    diffs = np.array(differences, dtype=float)
    n = len(diffs)
    mean_d = float(np.mean(diffs))
    std_d = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    se_d = std_d / math.sqrt(n) if n > 1 else 0.0
    
    # 95% t-interval (df = n - 1)
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


def analyze_recovery_study(results_dir=BASE_RESULTS_DIR):
    """
    Main analysis pipeline according to Section 16 decision rules.
    """
    data = load_all_results(results_dir)
    
    # 1. Extract Pre (step 2000) and Post (step 2500) Anchor BPCs
    delta_pre_list = []
    delta_post_list = []
    delta_proc_250_list = []
    delta_proc_500_list = []
    
    seed_details = {}
    
    for seed in SEEDS:
        asc_res = data[seed]["ascending"]
        desc_res = data[seed]["descending"]
        nonm_res = data[seed]["nonmonotonic"]
        
        # Helper to get anchor BPC at step
        def get_anchor_bpc(run_data, target_step):
            for a_log in run_data.get("anchor_logs", []):
                if a_log["step"] == target_step:
                    return a_log["anchor_bpc"]
            raise ValueError(f"Step {target_step} missing from anchor logs for seed {seed}")
            
        # Helper to get process BPC at step
        def get_process_bpc(run_data, target_step):
            for p_log in run_data.get("logs", []):
                if p_log["step"] == target_step:
                    return p_log["process_bpc"]
            raise ValueError(f"Step {target_step} missing from process logs for seed {seed}")
            
        asc_pre = get_anchor_bpc(asc_res, 2000)
        desc_pre = get_anchor_bpc(desc_res, 2000)
        nonm_pre = get_anchor_bpc(nonm_res, 2000)
        
        asc_post = get_anchor_bpc(asc_res, 2500)
        desc_post = get_anchor_bpc(desc_res, 2500)
        nonm_post = get_anchor_bpc(nonm_res, 2500)
        
        d_pre = desc_pre - asc_pre
        d_post = desc_post - asc_post
        
        # Process panel values for plateau check
        asc_proc_2250 = get_process_bpc(asc_res, 2250)  # k=250
        desc_proc_2250 = get_process_bpc(desc_res, 2250)
        asc_proc_2500 = get_process_bpc(asc_res, 2500)  # k=500
        desc_proc_2500 = get_process_bpc(desc_res, 2500)
        
        d_proc_250 = desc_proc_2250 - asc_proc_2250
        d_proc_500 = desc_proc_2500 - asc_proc_2500
        
        delta_pre_list.append(d_pre)
        delta_post_list.append(d_post)
        delta_proc_250_list.append(d_proc_250)
        delta_proc_500_list.append(d_proc_500)
        
        seed_details[seed] = {
            "ascending_pre": asc_pre,
            "descending_pre": desc_pre,
            "nonmonotonic_pre": nonm_pre,
            "delta_pre": d_pre,
            "ascending_post": asc_post,
            "descending_post": desc_post,
            "nonmonotonic_post": nonm_post,
            "delta_post": d_post,
            "delta_proc_250": d_proc_250,
            "delta_proc_500": d_proc_500,
        }
        
    stats_pre = compute_paired_stats(delta_pre_list)
    stats_post = compute_paired_stats(delta_post_list)
    
    # Absolute recovery: Delta^A(0) - Delta^A(500)
    abs_recovery_list = [pre - post for pre, post in zip(delta_pre_list, delta_post_list)]
    stats_abs_recovery = compute_paired_stats(abs_recovery_list)
    
    # Recovery ratio R = Delta^A(500) / Delta^A(0)
    mean_pre = stats_pre["mean"]
    mean_post = stats_post["mean"]
    
    if abs(mean_pre) >= THRESHOLD_MIN_PRE_GAP:
        recovery_ratio_R = mean_post / mean_pre
        r_interpretable = True
    else:
        recovery_ratio_R = None
        r_interpretable = False
        
    # Process panel plateau slope: Delta^P(250) - Delta^P(500)
    mean_proc_250 = float(np.mean(delta_proc_250_list))
    mean_proc_500 = float(np.mean(delta_proc_500_list))
    plateau_diff = mean_proc_500 - mean_proc_250  # negative if recovering
    recovery_slope_250 = mean_proc_250 - mean_proc_500
    is_plateaued = abs(plateau_diff) < THRESHOLD_PLATEAU_250
    
    # 2. Evaluate Decision Rules (Section 16)
    decision = {}
    if abs(mean_pre) < THRESHOLD_MIN_PRE_GAP:
        decision["case"] = "Case A: pre-recovery gap does not replicate"
        decision["verdict"] = "STOP"
        decision["rationale"] = (
            f"|mean Delta^A(0)| = {abs(mean_pre):.4f} < {THRESHOLD_MIN_PRE_GAP}. "
            "Pre-recovery gap too small to interpret. Stop mechanism work."
        )
    elif abs(mean_post) <= THRESHOLD_MIN_PRE_GAP:
        decision["case"] = "Case B: gap is practically removed"
        decision["verdict"] = "STOP"
        decision["rationale"] = (
            f"|mean Delta^A(500)| = {abs(mean_post):.4f} <= {THRESHOLD_MIN_PRE_GAP}. "
            "Residual gap eliminated. Supports reversible terminal-context/recency explanation."
        )
    elif THRESHOLD_MIN_PRE_GAP < abs(mean_post) <= THRESHOLD_RESIDUAL_PERSISTENT or stats_post["positive_count"] not in [0, 3]:
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
            "Recovery is active but incomplete. Trigger one-time 500-step extension to step 3000."
        )
    elif (
        mean_post > THRESHOLD_RESIDUAL_PERSISTENT
        and stats_post["positive_count"] == 3
        and is_plateaued
    ):
        decision["case"] = "Case E: persistent-effect candidate"
        decision["verdict"] = "GO_OPTIMIZER_EXPERIMENT"
        decision["rationale"] = (
            f"mean Delta^A(500) = {mean_post:.4f} > {THRESHOLD_RESIDUAL_PERSISTENT}, "
            f"3/3 seeds positive, and plateau condition satisfied (|{plateau_diff:.4f}| < {THRESHOLD_PLATEAU_250}). "
            "Candidate evidence for persistent path dependence; motivates targeted optimizer-state experiment."
        )
    else:
        decision["case"] = "Case C: unresolved / boundary edge case"
        decision["verdict"] = "UNRESOLVED"
        decision["rationale"] = "Pattern falls into boundary conditions. Report raw data without strong claims."
        
    summary = {
        "stats_pre": stats_pre,
        "stats_post": stats_post,
        "stats_abs_recovery": stats_abs_recovery,
        "recovery_ratio_R": recovery_ratio_R,
        "r_interpretable": r_interpretable,
        "recovery_slope_250": recovery_slope_250,
        "is_plateaued": is_plateaued,
        "decision": decision,
        "seed_details": seed_details,
    }
    
    # Save summary.json
    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print("\n" + "=" * 60)
    print("RECOVERY STUDY ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Pre-recovery Δ^A(0):   {stats_pre['mean']:+.4f} ± {stats_pre['std']:.4f} BPC  (95% CI: [{stats_pre['ci_95'][0]:+.4f}, {stats_pre['ci_95'][1]:+.4f}])")
    print(f"Post-recovery Δ^A(500): {stats_post['mean']:+.4f} ± {stats_post['std']:.4f} BPC  (95% CI: [{stats_post['ci_95'][0]:+.4f}, {stats_post['ci_95'][1]:+.4f}])")
    print(f"Absolute Recovered:    {stats_abs_recovery['mean']:+.4f} ± {stats_abs_recovery['std']:.4f} BPC")
    if r_interpretable:
        print(f"Recovery Ratio R:      {recovery_ratio_R:.3f} (criterion: R < 0.25)")
    else:
        print(f"Recovery Ratio R:      N/A (|pre-gap| < 0.02)")
    print(f"Direction consistency: {stats_post['direction_consistency']}")
    print(f"Plateau condition:     {'SATISFIED' if is_plateaued else 'STILL CHANGING'} (slope={recovery_slope_250:+.4f})")
    print("-" * 60)
    print(f"DECISION: {decision['case']}")
    print(f"VERDICT:  {decision['verdict']}")
    print(f"REASON:   {decision['rationale']}")
    print("=" * 60 + "\n")
    
    return summary


if __name__ == "__main__":
    analyze_recovery_study()
