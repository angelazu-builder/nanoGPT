"""
Analysis & Statistical Evaluation for Context-Length Recovery Study
Preregistered specification: Section 2, 3, 12, 13, 14, 15, 16
"""

import os
import sys
import json
import math
import argparse
import numpy as np
from scipy import stats

from .config import (
    SEEDS,
    ARMS,
    CONTEXT_LENGTHS,
    BASE_RESULTS_DIR,
    SMOKE_RESULTS_DIR,
    PROCESS_EVAL_STEPS,
    ANCHOR_EVAL_STEPS,
    THRESHOLD_MIN_PRE_GAP,
    THRESHOLD_RECOVERY_RATIO,
    THRESHOLD_RESIDUAL_PERSISTENT,
    THRESHOLD_PLATEAU_250,
    MAX_STEPS,
    HARD_CAP_STEPS,
)
from .figures import export_tidy_csvs, generate_contract_figures


def load_all_results(results_dir=BASE_RESULTS_DIR, seeds=None):
    """Load results for available or specified seeds."""
    if seeds is None:
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


def validate_formal_gate(data, active_seeds):
    """
    Enforce strict Preregistration Gate for Formal Analysis Mode:
    - 3 seeds (42, 43, 44), 3 arms
    - All runs completed=True, smoke_test=False, current_step >= 2500
    - All 17 process evaluation checkpoints present
    - Anchor steps 2000 and 2500 present
    - All 4 horizons present with 16 process / 32 anchor sequence BPCs
    - All values finite
    - Anchor first-16 exactly reconstruct process mean
    """
    expected_seeds = set(SEEDS)
    expected_arms = set(ARMS)
    expected_process_steps = set(PROCESS_EVAL_STEPS)
    expected_anchor_steps = set(ANCHOR_EVAL_STEPS)
    expected_horizons = set(CONTEXT_LENGTHS)
    
    if set(active_seeds) != expected_seeds:
        raise ValueError(f"Formal Gate Failed: Expected seeds {expected_seeds}, got {set(active_seeds)}")
        
    for seed in expected_seeds:
        if set(data[seed].keys()) != expected_arms:
            raise ValueError(f"Formal Gate Failed: Seed {seed} missing arms. Found {set(data[seed].keys())}")
            
        for arm in expected_arms:
            run_data = data[seed][arm]
            if not run_data.get("completed", False):
                raise ValueError(f"Formal Gate Failed: Run {arm} seed {seed} is not marked completed=True")
            if run_data.get("smoke_test", False):
                raise ValueError(f"Formal Gate Failed: Run {arm} seed {seed} is marked as a smoke test")
            if run_data.get("current_step", 0) < MAX_STEPS:
                raise ValueError(f"Formal Gate Failed: Run {arm} seed {seed} step={run_data.get('current_step')} < {MAX_STEPS}")
                
            log_steps = {l["step"] for l in run_data.get("logs", [])}
            if not expected_process_steps.issubset(log_steps):
                missing = expected_process_steps - log_steps
                raise ValueError(f"Formal Gate Failed: Run {arm} seed {seed} missing process checkpoints: {missing}")
                
            anchor_steps = {l["step"] for l in run_data.get("anchor_logs", [])}
            if not expected_anchor_steps.issubset(anchor_steps):
                missing = expected_anchor_steps - anchor_steps
                raise ValueError(f"Formal Gate Failed: Run {arm} seed {seed} missing anchor checkpoints: {missing}")
                
            # Verify horizon matrices and finiteness
            for l in run_data.get("logs", []):
                h_res = l.get("horizon_results", {})
                for h in expected_horizons:
                    h_val = h if h in h_res else str(h)
                    if h_val not in h_res:
                        raise ValueError(f"Formal Gate Failed: Step {l['step']} missing horizon T={h}")
                    seqs = h_res[h_val].get("seq_bpc", [])
                    if len(seqs) != 16:
                        raise ValueError(f"Formal Gate Failed: Step {l['step']} T={h} expected 16 sequence BPCs, got {len(seqs)}")
                    if not np.all(np.isfinite(seqs)):
                        raise ValueError(f"Formal Gate Failed: Step {l['step']} T={h} contains non-finite values")
                        
            # Verify anchor panel sequence count (32) and nesting reconstruction
            for a_log in run_data.get("anchor_logs", []):
                step = a_log["step"]
                h_res = a_log.get("horizon_results", {})
                p_log = next(l for l in run_data["logs"] if l["step"] == step)
                p_res = p_log.get("horizon_results", {})
                
                for h in expected_horizons:
                    h_val = h if h in h_res else str(h)
                    a_seqs = h_res[h_val].get("seq_bpc", [])
                    if len(a_seqs) != 32:
                        raise ValueError(f"Formal Gate Failed: Anchor step {step} T={h} expected 32 sequence BPCs, got {len(a_seqs)}")
                    if not np.all(np.isfinite(a_seqs)):
                        raise ValueError(f"Formal Gate Failed: Anchor step {step} T={h} contains non-finite values")
                    
                    # Reconstruction check: first 16 anchor sequences must match process mean
                    reconstructed_mean = float(np.mean(a_seqs[:16]))
                    proc_mean = p_res[h_val]["mean_bpc"]
                    if abs(reconstructed_mean - proc_mean) > 1e-4:
                        raise ValueError(
                            f"Formal Gate Failed: Nesting invariant violated at step {step} T={h}: "
                            f"reconstructed={reconstructed_mean:.6f} vs process={proc_mean:.6f}"
                        )
                        
    print("✔ Formal Preregistration Gate Passed: All 9 runs, 17 checkpoints, nested panels, and finiteness verified.")
    return True


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


def analyze_recovery_study(
    results_dir=BASE_RESULTS_DIR,
    generate_plots=True,
    seeds=None,
    allow_partial=False,
):
    """
    Comprehensive analysis pipeline fulfilling all preregistered requirements:
    1. Formal Gate Validation (or explicit partial debug banner).
    2. Primary confirmatory contrast: Delta^A(0), Delta^A(500), absolute recovery, ratio R.
    3. Primary hypothesis test: R < 0.25 (75% recovery).
    4. Multi-horizon trajectories and recovery matrices: T=32, 64, 128, 256.
    5. Adaptive extension analysis preserving base endpoints.
    6. Transition-local loss shocks.
    7. Validation BPC AUC.
    8. Nonmonotonic exploratory contrasts.
    9. Context-sensitive target analysis.
    10. Case A-E decision rule evaluation.
    11. Amendment 001 contract figures & CSV tables.
    """
    data, active_seeds = load_all_results(results_dir, seeds=seeds)
    
    if not allow_partial:
        validate_formal_gate(data, active_seeds)
    else:
        print("\n" + "*" * 68)
        print("***   DEBUG / PARTIAL ANALYSIS — NOT A FORMAL RESULT             ***")
        print("*" * 68 + "\n")
        
    sample_run = data[active_seeds[0]]["ascending"]
    anchor_steps = sorted([log["step"] for log in sample_run.get("anchor_logs", [])])
    
    base_pre_step = 2000 if 2000 in anchor_steps else anchor_steps[0]
    base_post_step = 2500 if 2500 in anchor_steps else anchor_steps[-1]
    
    # Check if all runs completed the one-time extension (step 3000)
    has_complete_extension = all(
        data[s][a].get("current_step", 0) >= HARD_CAP_STEPS
        for s in active_seeds for a in ARMS
    )
    
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

    # -------------------------------------------------------------
    # 1. Base Pre/Post Analysis (Steps 2000 & 2500)
    # -------------------------------------------------------------
    delta_pre_list = []
    delta_post_list = []
    delta_p250_list = []
    delta_p500_list = []
    
    seed_details = {}
    
    for seed in active_seeds:
        asc_res = data[seed]["ascending"]
        desc_res = data[seed]["descending"]
        nonm_res = data[seed]["nonmonotonic"]
        
        asc_a_pre = get_anchor_log(asc_res, base_pre_step)
        desc_a_pre = get_anchor_log(desc_res, base_pre_step)
        nonm_a_pre = get_anchor_log(nonm_res, base_pre_step)
        
        asc_a_post = get_anchor_log(asc_res, base_post_step)
        desc_a_post = get_anchor_log(desc_res, base_post_step)
        nonm_a_post = get_anchor_log(nonm_res, base_post_step)
        
        d_pre = desc_a_pre["anchor_bpc"] - asc_a_pre["anchor_bpc"]
        d_post = desc_a_post["anchor_bpc"] - asc_a_post["anchor_bpc"]
        
        delta_pre_list.append(d_pre)
        delta_post_list.append(d_post)
        
        # Plateau steps for standard recovery: 2250 (k=250) and 2500 (k=500)
        p250_step = 2250 if any(l["step"] == 2250 for l in asc_res["logs"]) else base_pre_step
        p500_step = 2500 if any(l["step"] == 2500 for l in asc_res["logs"]) else base_post_step
        
        asc_p250 = get_process_log(asc_res, p250_step)
        desc_p250 = get_process_log(desc_res, p250_step)
        asc_p500 = get_process_log(asc_res, p500_step)
        desc_p500 = get_process_log(desc_res, p500_step)
        
        d_p250 = desc_p250["process_bpc"] - asc_p250["process_bpc"]
        d_p500 = desc_p500["process_bpc"] - asc_p500["process_bpc"]
        
        delta_p250_list.append(d_p250)
        delta_p500_list.append(d_p500)
        
        seed_details[seed] = {
            "pre": {
                "ascending": asc_a_pre["anchor_bpc"],
                "descending": desc_a_pre["anchor_bpc"],
                "nonmonotonic": nonm_a_pre["anchor_bpc"],
                "delta_D_A": d_pre,
                "delta_N_A": nonm_a_pre["anchor_bpc"] - asc_a_pre["anchor_bpc"],
                "delta_D_N": desc_a_pre["anchor_bpc"] - nonm_a_pre["anchor_bpc"],
            },
            "post_500": {
                "ascending": asc_a_post["anchor_bpc"],
                "descending": desc_a_post["anchor_bpc"],
                "nonmonotonic": nonm_a_post["anchor_bpc"],
                "delta_D_A": d_post,
                "delta_N_A": nonm_a_post["anchor_bpc"] - asc_a_post["anchor_bpc"],
                "delta_D_N": desc_a_post["anchor_bpc"] - nonm_a_post["anchor_bpc"],
            },
            "plateau_500": {
                "delta_proc_250": d_p250,
                "delta_proc_500": d_p500,
                "slope_250": d_p250 - d_p500,
            }
        }
        
    stats_pre = compute_paired_stats(delta_pre_list)
    stats_post_500 = compute_paired_stats(delta_post_list)
    
    abs_recovery_500 = [pre - post for pre, post in zip(delta_pre_list, delta_post_list)]
    stats_abs_recovery_500 = compute_paired_stats(abs_recovery_500)
    
    mean_pre = stats_pre["mean"]
    mean_post_500 = stats_post_500["mean"]
    
    if abs(mean_pre) >= THRESHOLD_MIN_PRE_GAP:
        recovery_ratio_R = mean_post_500 / mean_pre
        r_interpretable = True
        pct_recovered = (1.0 - recovery_ratio_R) * 100.0
    else:
        recovery_ratio_R = None
        r_interpretable = False
        pct_recovered = None
        
    mean_p250 = float(np.mean(delta_p250_list))
    mean_p500 = float(np.mean(delta_p500_list))
    slope_250 = mean_p250 - mean_p500
    is_plateaued_500 = abs(mean_p500 - mean_p250) < THRESHOLD_PLATEAU_250
    
    base_results = {
        "pre_step": base_pre_step,
        "post_step": base_post_step,
        "stats_pre": stats_pre,
        "stats_post_500": stats_post_500,
        "stats_abs_recovery_500": stats_abs_recovery_500,
        "recovery_ratio_R": recovery_ratio_R,
        "pct_recovered": pct_recovered,
        "r_interpretable": r_interpretable,
        "primary_hypothesis_satisfied": bool(recovery_ratio_R is not None and recovery_ratio_R < THRESHOLD_RECOVERY_RATIO),
        "recovery_slope_250": slope_250,
        "is_plateaued": is_plateaued_500,
    }
    
    # -------------------------------------------------------------
    # 2. Extension Analysis (if completed to step 3000)
    # -------------------------------------------------------------
    extension_results = None
    if has_complete_extension:
        delta_ext_list = []
        delta_p750_list = []
        delta_p1000_list = []
        
        for seed in active_seeds:
            asc_a3000 = get_anchor_log(data[seed]["ascending"], 3000)
            desc_a3000 = get_anchor_log(data[seed]["descending"], 3000)
            d_ext = desc_a3000["anchor_bpc"] - asc_a3000["anchor_bpc"]
            delta_ext_list.append(d_ext)
            
            asc_run = data[seed]["ascending"]
            desc_run = data[seed]["descending"]
            p750 = get_process_log(desc_run, 2750)["process_bpc"] - get_process_log(asc_run, 2750)["process_bpc"]
            p1000 = get_process_log(desc_run, 3000)["process_bpc"] - get_process_log(asc_run, 3000)["process_bpc"]
            delta_p750_list.append(p750)
            delta_p1000_list.append(p1000)
            
            seed_details[seed]["post_1000"] = {
                "ascending": asc_a3000["anchor_bpc"],
                "descending": desc_a3000["anchor_bpc"],
                "delta_D_A": d_ext,
            }
            
        stats_ext = compute_paired_stats(delta_ext_list)
        late_slope = float(np.mean(delta_p750_list)) - float(np.mean(delta_p1000_list))
        is_plateaued_ext = abs(float(np.mean(delta_p1000_list)) - float(np.mean(delta_p750_list))) < THRESHOLD_PLATEAU_250
        
        extension_results = {
            "stats_post_1000": stats_ext,
            "late_slope_250": late_slope,
            "is_plateaued_1000": is_plateaued_ext,
        }
        
    # -------------------------------------------------------------
    # 3. Decision Rules (Section 16 & Amendment Extension Logic)
    # -------------------------------------------------------------
    decision = {}
    if has_complete_extension:
        # Decision at Hard Cap (Step 3000): NEVER return EXTEND_RECOVERY again!
        mean_ext = extension_results["stats_post_1000"]["mean"]
        late_slope = extension_results["late_slope_250"]
        all_three_pos = extension_results["stats_post_1000"]["positive_count"] == len(active_seeds)
        
        if late_slope > THRESHOLD_PLATEAU_250:
            decision["case"] = "Case D (Terminal): Unresolved at Hard Cap"
            decision["verdict"] = "UNRESOLVED_AT_HARD_CAP"
            decision["rationale"] = (
                f"At step 3000 hard cap, mean Delta^A(1000) = {mean_ext:.4f} BPC is still decreasing "
                f"by {late_slope:.4f} > {THRESHOLD_PLATEAU_250} BPC over the final interval. "
                "Slow recovery cannot be distinguished from persistent path dependence; no further extension permitted."
            )
        elif mean_ext > THRESHOLD_RESIDUAL_PERSISTENT and all_three_pos and extension_results["is_plateaued_1000"]:
            decision["case"] = "Case E (Extended): Persistent-effect candidate"
            decision["verdict"] = "GO_OPTIMIZER_EXPERIMENT"
            decision["rationale"] = (
                f"At step 3000 hard cap, mean Delta^A(1000) = {mean_ext:.4f} > {THRESHOLD_RESIDUAL_PERSISTENT}, "
                f"3/3 seeds positive, and plateau condition satisfied. Motivates targeted optimizer-state experiment."
            )
        else:
            decision["case"] = "Case B/C (Extended): Residual eliminated or unresolved"
            decision["verdict"] = "UNRESOLVED_OR_RECOVERED"
            decision["rationale"] = f"At step 3000 hard cap, residual mean Delta^A(1000) = {mean_ext:.4f} BPC."
    else:
        # Standard decision on Base 2500 endpoints
        if abs(mean_pre) < THRESHOLD_MIN_PRE_GAP:
            decision["case"] = "Case A: pre-recovery gap does not replicate"
            decision["verdict"] = "STOP"
            decision["rationale"] = f"|mean Delta^A(0)| = {abs(mean_pre):.4f} < {THRESHOLD_MIN_PRE_GAP}. Stop mechanism work."
        elif abs(mean_post_500) <= THRESHOLD_MIN_PRE_GAP:
            decision["case"] = "Case B: gap is practically removed"
            decision["verdict"] = "STOP"
            decision["rationale"] = f"|mean Delta^A(500)| = {abs(mean_post_500):.4f} <= {THRESHOLD_MIN_PRE_GAP}. Supports reversible recency."
        elif THRESHOLD_MIN_PRE_GAP < abs(mean_post_500) <= THRESHOLD_RESIDUAL_PERSISTENT or stats_post_500["positive_count"] not in [0, len(active_seeds)]:
            decision["case"] = "Case C: small residual is unresolved"
            decision["verdict"] = "UNRESOLVED"
            decision["rationale"] = f"Residual mean Delta^A(500) = {mean_post_500:.4f} is between 0.02 and 0.03 BPC, or seed directions disagree."
        elif mean_post_500 > THRESHOLD_RESIDUAL_PERSISTENT and slope_250 > THRESHOLD_PLATEAU_250:
            decision["case"] = "Case D: gap remains large but is still recovering"
            decision["verdict"] = "EXTEND_RECOVERY"
            decision["rationale"] = (
                f"mean Delta^A(500) = {mean_post_500:.4f} > {THRESHOLD_RESIDUAL_PERSISTENT} "
                f"and slope Delta^P(250) - Delta^P(500) = {slope_250:.4f} > {THRESHOLD_PLATEAU_250}. "
                "Trigger one-time 500-step extension to step 3000 via: "
                "`python -m training_dynamics_research.recovery_study.runner --extend`"
            )
        elif mean_post_500 > THRESHOLD_RESIDUAL_PERSISTENT and stats_post_500["positive_count"] == len(active_seeds) and is_plateaued_500:
            decision["case"] = "Case E: persistent-effect candidate"
            decision["verdict"] = "GO_OPTIMIZER_EXPERIMENT"
            decision["rationale"] = (
                f"mean Delta^A(500) = {mean_post_500:.4f} > {THRESHOLD_RESIDUAL_PERSISTENT}, "
                f"3/3 seeds positive, and plateau condition satisfied. Motivates targeted optimizer-state experiment."
            )
        else:
            decision["case"] = "Case C: unresolved / boundary edge case"
            decision["verdict"] = "UNRESOLVED"
            decision["rationale"] = "Pattern falls into boundary conditions."

    # -------------------------------------------------------------
    # 4. Multi-Horizon, Shock, AUC, and Contrasts
    # -------------------------------------------------------------
    checkpoints = [log["step"] for log in sample_run["logs"]]
    
    # Four-horizon endpoint matrices
    horizon_endpoint_stats = {
        h: {
            "pre": compute_paired_stats([
                get_anchor_log(data[s]["descending"], base_pre_step)["horizon_bpc"].get(h, get_anchor_log(data[s]["descending"], base_pre_step)["horizon_bpc"].get(str(h))) -
                get_anchor_log(data[s]["ascending"], base_pre_step)["horizon_bpc"].get(h, get_anchor_log(data[s]["ascending"], base_pre_step)["horizon_bpc"].get(str(h)))
                for s in active_seeds
            ]),
            "post_500": compute_paired_stats([
                get_anchor_log(data[s]["descending"], base_post_step)["horizon_bpc"].get(h, get_anchor_log(data[s]["descending"], base_post_step)["horizon_bpc"].get(str(h))) -
                get_anchor_log(data[s]["ascending"], base_post_step)["horizon_bpc"].get(h, get_anchor_log(data[s]["ascending"], base_post_step)["horizon_bpc"].get(str(h)))
                for s in active_seeds
            ]),
        }
        for h in CONTEXT_LENGTHS
    }
    
    # Transition shocks
    candidate_pairs = [
        (500, 501), (1000, 1001), (1500, 1501), (2000, 2001),
        (10, 11), (20, 21), (30, 31), (40, 41)
    ]
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
            
    # AUC Analysis
    pre_steps = [s for s in checkpoints if s <= base_pre_step]
    rec_steps = [s for s in checkpoints if s >= base_pre_step]
    auc_results = {}
    for arm in ARMS:
        pre_aucs = [compute_auc(pre_steps, [get_process_log(data[s][arm], st)["process_bpc"] for st in pre_steps]) for s in active_seeds]
        rec_aucs = [compute_auc(rec_steps, [get_process_log(data[s][arm], st)["process_bpc"] for st in rec_steps]) for s in active_seeds]
        auc_results[arm] = {
            "pre_recovery_auc": {"mean": float(np.mean(pre_aucs)), "raw": pre_aucs},
            "recovery_auc": {"mean": float(np.mean(rec_aucs)), "raw": rec_aucs},
        }

    # Context sensitivity pooled top-10% analysis
    sensitivity_analysis = {}
    for s_step in [str(base_pre_step), str(base_post_step)]:
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

    summary = {
        "active_seeds": active_seeds,
        "has_complete_extension": has_complete_extension,
        "base_results": base_results,
        "extension_results": extension_results,
        "decision": decision,
        "horizon_endpoint_stats": horizon_endpoint_stats,
        "transition_shocks": transition_shocks,
        "auc_results": auc_results,
        "sensitivity_analysis": sensitivity_analysis,
        "seed_details": seed_details,
    }
    
    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved comprehensive summary to {summary_path}")

    # -------------------------------------------------------------
    # 5. Figures and Tidy Data CSV Exports (Amendment 001)
    # -------------------------------------------------------------
    if generate_plots:
        try:
            figures_dir = os.path.join(results_dir, "figures")
            export_tidy_csvs(data, active_seeds, checkpoints, base_pre_step, base_post_step, figures_dir)
            generate_contract_figures(data, active_seeds, checkpoints, base_pre_step, base_post_step, figures_dir)
        except Exception as e:
            if allow_partial:
                print(f"DEBUG: Figure/CSV contract export failed: {e}")
            else:
                raise

    # Terminal Summary
    print("\n" + "=" * 68)
    print("CONTEXT-LENGTH RECOVERY STUDY: PREREGISTERED ANALYSIS SUMMARY")
    print("=" * 68)
    print(f"Active Seeds:           {active_seeds}")
    print(f"Pre-recovery Δ^A(0):    {stats_pre['mean']:+.4f} ± {stats_pre['std']:.4f} BPC  (95% CI: [{stats_pre['ci_95'][0]:+.4f}, {stats_pre['ci_95'][1]:+.4f}])")
    print(f"Post-recovery Δ^A(500):  {stats_post_500['mean']:+.4f} ± {stats_post_500['std']:.4f} BPC  (95% CI: [{stats_post_500['ci_95'][0]:+.4f}, {stats_post_500['ci_95'][1]:+.4f}])")
    print(f"Absolute Recovered:     {stats_abs_recovery_500['mean']:+.4f} ± {stats_abs_recovery_500['std']:.4f} BPC")
    if r_interpretable:
        print(f"Recovery Ratio R:       {recovery_ratio_R:.3f} ({pct_recovered:.1f}% recovered) | Criterion R < 0.25: {recovery_ratio_R < THRESHOLD_RECOVERY_RATIO}")
    else:
        print(f"Recovery Ratio R:       N/A (|pre-gap| < {THRESHOLD_MIN_PRE_GAP})")
    print(f"Direction consistency:  {stats_post_500['direction_consistency']}")
    print(f"Plateau condition:      {'SATISFIED' if is_plateaued_500 else 'ACTIVE RECOVERY'} (slope={slope_250:+.4f} BPC / 250 steps)")
    if has_complete_extension:
        print("-" * 68)
        print(f"Extension Δ^A(1000):    {extension_results['stats_post_1000']['mean']:+.4f} ± {extension_results['stats_post_1000']['std']:.4f} BPC")
        print(f"Late Plateau Slope:     {extension_results['late_slope_250']:+.4f} BPC / 250 steps")
    print("-" * 68)
    print("FOUR-HORIZON ENDPOINT MATRIX (Anchor Panel):")
    for h in CONTEXT_LENGTHS:
        pre_h = horizon_endpoint_stats[h]["pre"]["mean"]
        post_h = horizon_endpoint_stats[h]["post_500"]["mean"]
        print(f"  T={h:3d} | Pre: {pre_h:+.4f} BPC | Post: {post_h:+.4f} BPC | Recovered: {pre_h - post_h:+.4f} BPC")
    print("-" * 68)
    print(f"DECISION: {decision['case']}")
    print(f"VERDICT:  {decision['verdict']}")
    print(f"REASON:   {decision['rationale']}")
    print("=" * 68 + "\n")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Context-Length Recovery Study")
    parser.add_argument("--results-dir", type=str, default=BASE_RESULTS_DIR, help="Results directory to analyze")
    parser.add_argument("--allow-partial", action="store_true", help="Allow partial/debug analysis without formal gate")
    args = parser.parse_args()
    
    analyze_recovery_study(results_dir=args.results_dir, allow_partial=args.allow_partial)
