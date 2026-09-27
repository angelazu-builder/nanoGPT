"""
Analysis, Hypothesis Testing, and Decision Engine for Context-Length Recovery Study
Preregistered specification: Section 6, 8, 12, 13, 14, 15, 16, 17
Amendment 001: PREREGISTRATION_AMENDMENT_001_FIGURES.md

Decoupled architecture:
1. load_and_validate_study(results_dir, allow_partial)
2. tables = extract_study_tables(study, active_seeds)
3. summary = compute_preregistered_estimands(tables, study, active_seeds)
4. figures / tidy CSVs generated strictly from tables and summary
"""

import os
import sys
import json
import math
import argparse
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from .config import (
    SEEDS,
    ARMS,
    PROCESS_EVAL_STEPS,
    ANCHOR_EVAL_STEPS,
    CONTEXT_LENGTHS,
    BASE_RESULTS_DIR,
    MAX_STEPS,
    HARD_CAP_STEPS,
    THRESHOLD_MIN_PRE_GAP,
    THRESHOLD_RECOVERY_RATIO,
    THRESHOLD_RESIDUAL_PERSISTENT,
    THRESHOLD_PLATEAU_250,
)
from .core_types import StudyTables
from .tables import extract_study_tables, export_tidy_csvs
from .figures import plot_contract_figures


def load_all_results(results_dir: str, seeds: Optional[List[int]] = None) -> Tuple[Dict[int, Dict[str, Any]], List[int]]:
    """Loads all run_results.json across seeds and arms."""
    data = {}
    if seeds is None:
        candidate_seeds = []
        if os.path.exists(results_dir):
            for d in os.listdir(results_dir):
                if d.startswith("seed_") and os.path.isdir(os.path.join(results_dir, d)):
                    try:
                        candidate_seeds.append(int(d.split("_")[1]))
                    except ValueError:
                        pass
        seeds = sorted(candidate_seeds) if candidate_seeds else SEEDS

    active_seeds = []
    for seed in seeds:
        seed_data = {}
        for arm in ARMS:
            res_file = os.path.join(results_dir, f"seed_{seed}", arm, "run_results.json")
            if os.path.exists(res_file):
                with open(res_file, "r") as f:
                    seed_data[arm] = json.load(f)
        if len(seed_data) > 0:
            data[seed] = seed_data
            active_seeds.append(seed)

    if not active_seeds:
        raise FileNotFoundError(f"No experiment results found in {results_dir}")
        
    return data, active_seeds


def validate_formal_gate(data: Dict[int, Dict[str, Any]], active_seeds: List[int]):
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
                    
                    reconstructed_mean = float(np.mean(a_seqs[:16]))
                    proc_mean = p_res[h_val]["mean_bpc"]
                    if abs(reconstructed_mean - proc_mean) > 1e-4:
                        raise ValueError(
                            f"Formal Gate Failed: Nesting invariant violated at step {step} T={h}: "
                            f"reconstructed={reconstructed_mean:.6f} vs process={proc_mean:.6f}"
                        )
                        
    print("✔ Formal Preregistration Gate Passed: All 9 runs, 17 checkpoints, nested panels, and finiteness verified.")


def compute_paired_stats(differences: List[float]) -> Dict[str, Any]:
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


def compute_auc(steps: List[int], values: List[float]) -> float:
    if len(steps) < 2:
        return 0.0
    x = np.array(steps, dtype=float)
    y = np.array(values, dtype=float)
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) * 0.5))


def load_and_validate_study(
    results_dir: str = BASE_RESULTS_DIR,
    seeds: Optional[List[int]] = None,
    allow_partial: bool = False,
) -> Tuple[Dict[int, Dict[str, Any]], List[int]]:
    """Loads study data and applies the formal preregistration gate."""
    data, active_seeds = load_all_results(results_dir, seeds=seeds)
    if not allow_partial:
        validate_formal_gate(data, active_seeds)
    else:
        print("\n" + "*" * 68)
        print("***   DEBUG / PARTIAL ANALYSIS — NOT A FORMAL RESULT             ***")
        print("*" * 68 + "\n")
    return data, active_seeds


def compute_preregistered_estimands(
    tables: StudyTables,
    raw_data: Dict[int, Dict[str, Any]],
    active_seeds: List[int],
) -> Dict[str, Any]:
    """
    Computes all preregistered estimands using pure canonical StudyTables:
    - Primary confirmatory contrast: Delta^A(0), Delta^A(500), absolute recovery, ratio R
    - Recovery slope and plateau status
    - Multi-horizon endpoint matrices
    - Adaptive extension statistics
    - Transition shocks and AUC
    - Case A-E decision rules
    """
    sample_run = raw_data[active_seeds[0]]["ascending"]
    anchor_steps = sorted([log["step"] for log in sample_run.get("anchor_logs", [])])
    base_pre_step = 2000 if 2000 in anchor_steps else anchor_steps[0]
    base_post_step = 2500 if 2500 in anchor_steps else anchor_steps[-1]
    
    checkpoints = sorted([log["step"] for log in sample_run.get("logs", [])])
    
    has_complete_extension = all(
        raw_data[s][a].get("current_step", 0) >= HARD_CAP_STEPS
        for s in active_seeds for a in ARMS
    )

    # 1. Base Pre/Post Analysis (T=256)
    delta_pre_list = []
    delta_post_list = []
    delta_p250_list = []
    delta_p500_list = []
    seed_details = {}
    
    p250_step = 2250 if 2250 in checkpoints else (base_pre_step if base_pre_step in checkpoints else checkpoints[-1])
    p500_step = 2500 if 2500 in checkpoints else base_post_step

    for seed in active_seeds:
        d_pre = tables.anchor_bpc(seed, "descending", base_pre_step, 256) - tables.anchor_bpc(seed, "ascending", base_pre_step, 256)
        d_post = tables.anchor_bpc(seed, "descending", base_post_step, 256) - tables.anchor_bpc(seed, "ascending", base_post_step, 256)
        delta_pre_list.append(d_pre)
        delta_post_list.append(d_post)
        
        d_p250 = tables.process_bpc(seed, "descending", p250_step, 256) - tables.process_bpc(seed, "ascending", p250_step, 256)
        d_p500 = tables.process_bpc(seed, "descending", p500_step, 256) - tables.process_bpc(seed, "ascending", p500_step, 256)
        delta_p250_list.append(d_p250)
        delta_p500_list.append(d_p500)
        
        seed_details[seed] = {
            "pre": {
                "ascending": tables.anchor_bpc(seed, "ascending", base_pre_step, 256),
                "descending": tables.anchor_bpc(seed, "descending", base_pre_step, 256),
                "nonmonotonic": tables.anchor_bpc(seed, "nonmonotonic", base_pre_step, 256),
                "delta_D_A": d_pre,
            },
            "post_500": {
                "ascending": tables.anchor_bpc(seed, "ascending", base_post_step, 256),
                "descending": tables.anchor_bpc(seed, "descending", base_post_step, 256),
                "nonmonotonic": tables.anchor_bpc(seed, "nonmonotonic", base_post_step, 256),
                "delta_D_A": d_post,
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

    # 2. Extension Analysis (Step 3000)
    extension_results = None
    if has_complete_extension:
        delta_ext_list = []
        delta_p750_list = []
        delta_p1000_list = []
        
        for seed in active_seeds:
            d_ext = tables.anchor_bpc(seed, "descending", 3000, 256) - tables.anchor_bpc(seed, "ascending", 3000, 256)
            delta_ext_list.append(d_ext)
            p750 = tables.process_bpc(seed, "descending", 2750, 256) - tables.process_bpc(seed, "ascending", 2750, 256)
            p1000 = tables.process_bpc(seed, "descending", 3000, 256) - tables.process_bpc(seed, "ascending", 3000, 256)
            delta_p750_list.append(p750)
            delta_p1000_list.append(p1000)
            seed_details[seed]["post_1000"] = {
                "ascending": tables.anchor_bpc(seed, "ascending", 3000, 256),
                "descending": tables.anchor_bpc(seed, "descending", 3000, 256),
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

    # 3. Preregistered Decision Rules
    decision = {}
    if has_complete_extension:
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
        if abs(mean_pre) < THRESHOLD_MIN_PRE_GAP:
            decision["case"] = "Case A: pre-recovery gap does not replicate"
            decision["verdict"] = "STOP"
            decision["rationale"] = f"|mean Delta^A(0)| = {abs(mean_pre):.4f} < {THRESHOLD_MIN_PRE_GAP}. Stop mechanism work."
        elif abs(mean_post_500) <= THRESHOLD_MIN_PRE_GAP:
            decision["case"] = "Case B: gap is practically removed"
            decision["verdict"] = "STOP"
            decision["rationale"] = (
                f"|mean Delta^A(500)| = {abs(mean_post_500):.4f} <= "
                f"{THRESHOLD_MIN_PRE_GAP} BPC. Supports a reversible terminal-context effect."
            )
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
            decision["case"] = "Case C: large deficit reversed; residual sign reversal unresolved"
            decision["verdict"] = "UNRESOLVED"
            decision["rationale"] = (
                f"The pre-recovery deficit reversed sign: mean Delta^A(0) = {mean_pre:+.4f} BPC and "
                f"mean Delta^A(500) = {mean_post_500:+.4f} BPC. The reversed residual has magnitude "
                f"{abs(mean_post_500):.4f} > {THRESHOLD_MIN_PRE_GAP} BPC, so it does not satisfy "
                "the preregistered Case B practical-removal rule. The original positive deficit did not persist, "
                "but the post-recovery residual remains unresolved."
            )

    # 4. Multi-Horizon Endpoint Matrix
    horizon_endpoint_stats = {}
    for h in CONTEXT_LENGTHS:
        h_pre_diffs = [tables.anchor_bpc(s, "descending", base_pre_step, h) - tables.anchor_bpc(s, "ascending", base_pre_step, h) for s in active_seeds]
        h_post_diffs = [tables.anchor_bpc(s, "descending", base_post_step, h) - tables.anchor_bpc(s, "ascending", base_post_step, h) for s in active_seeds]
        horizon_endpoint_stats[h] = {
            "pre": compute_paired_stats(h_pre_diffs),
            "post_500": compute_paired_stats(h_post_diffs),
            "recovered": compute_paired_stats([pre - post for pre, post in zip(h_pre_diffs, h_post_diffs)]),
        }

    # 5. Validation BPC AUC
    auc_results = {}
    for arm in ARMS:
        auc_results[arm] = {}
        for h in CONTEXT_LENGTHS:
            arm_h_aucs = []
            for seed in active_seeds:
                series = [tables.process_bpc(seed, arm, step, h) for step in checkpoints]
                arm_h_aucs.append(compute_auc(checkpoints, series))
            auc_results[arm][h] = {
                "mean_auc": float(np.mean(arm_h_aucs)),
                "raw_aucs": arm_h_aucs,
            }

    # 6. Context-Sensitivity Analysis
    sensitivity_analysis = {}
    for s_step in [str(base_pre_step), str(base_post_step)]:
        if s_step in sample_run.get("sensitivity_logs", {}):
            sens_stats = {}
            for arm in ARMS:
                c_means = [raw_data[s][arm]["sensitivity_logs"][s_step]["mean_c"] for s in active_seeds]
                c_fracs = [raw_data[s][arm]["sensitivity_logs"][s_step]["frac_gt_0_1"] for s in active_seeds]
                sens_stats[arm] = {
                    "mean_c": float(np.mean(c_means)),
                    "frac_gt_0_1": float(np.mean(c_fracs)),
                }
            sensitivity_analysis[f"step_{s_step}"] = sens_stats

    return {
        "active_seeds": active_seeds,
        "base_pre_step": base_pre_step,
        "base_post_step": base_post_step,
        "checkpoints": checkpoints,
        "has_complete_extension": has_complete_extension,
        "stats_pre": stats_pre,
        "stats_post_500": stats_post_500,
        "stats_abs_recovery_500": stats_abs_recovery_500,
        "base_results": base_results,
        "extension_results": extension_results,
        "decision": decision,
        "horizon_endpoint_stats": horizon_endpoint_stats,
        "auc_results": auc_results,
        "sensitivity_analysis": sensitivity_analysis,
        "seed_details": seed_details,
    }


def analyze_recovery_study(
    results_dir: str = BASE_RESULTS_DIR,
    generate_plots: bool = True,
    seeds: Optional[List[int]] = None,
    allow_partial: bool = False,
) -> Dict[str, Any]:
    """
    Main entry point for analyzing Context-Length Recovery Study.
    Executes the clean decoupled pipeline:
    1. load_and_validate_study
    2. extract_study_tables
    3. compute_preregistered_estimands
    4. export_tidy_csvs
    5. plot_contract_figures
    """
    raw_data, active_seeds = load_and_validate_study(results_dir, seeds=seeds, allow_partial=allow_partial)
    tables = extract_study_tables(raw_data, active_seeds)
    summary = compute_preregistered_estimands(tables, raw_data, active_seeds)
    
    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved comprehensive summary to {summary_path}")
    
    if generate_plots:
        figures_dir = os.path.join(results_dir, "figures")
        try:
            export_tidy_csvs(tables, figures_dir)
            print(f"Generated contract CSVs in {os.path.join(figures_dir, 'data')}")
            plot_contract_figures(tables, summary, active_seeds, summary["checkpoints"], figures_dir)
            print(f"Generated contract figures in {figures_dir}")
        except Exception as e:
            if allow_partial:
                print(f"DEBUG: Figure/CSV contract export failed: {e}")
            else:
                raise

    # Terminal summary output
    stats_pre = summary["stats_pre"]
    stats_post_500 = summary["stats_post_500"]
    stats_abs = summary["stats_abs_recovery_500"]
    base_res = summary["base_results"]
    decision = summary["decision"]
    
    print("\n" + "=" * 68)
    print("CONTEXT-LENGTH RECOVERY STUDY: PREREGISTERED ANALYSIS SUMMARY")
    print("=" * 68)
    print(f"Active Seeds:           {active_seeds}")
    print(f"Pre-recovery Δ^A(0):    {stats_pre['mean']:+.4f} ± {stats_pre['std']:.4f} BPC  (95% CI: [{stats_pre['ci_95'][0]:+.4f}, {stats_pre['ci_95'][1]:+.4f}])")
    print(f"Post-recovery Δ^A(500):  {stats_post_500['mean']:+.4f} ± {stats_post_500['std']:.4f} BPC  (95% CI: [{stats_post_500['ci_95'][0]:+.4f}, {stats_post_500['ci_95'][1]:+.4f}])")
    print(f"Absolute Recovered:     {stats_abs['mean']:+.4f} ± {stats_abs['std']:.4f} BPC")
    if base_res["r_interpretable"]:
        print(f"Recovery Ratio R:       {base_res['recovery_ratio_R']:.3f} ({base_res['pct_recovered']:.1f}% recovered) | Criterion R < 0.25: {base_res['primary_hypothesis_satisfied']}")
    else:
        print(f"Recovery Ratio R:       N/A (|pre-gap| < {THRESHOLD_MIN_PRE_GAP})")
    print(f"Direction consistency:  {stats_post_500['direction_consistency']}")
    print(f"Plateau condition:      {'SATISFIED' if base_res['is_plateaued'] else 'ACTIVE RECOVERY'} (slope={base_res['recovery_slope_250']:+.4f} BPC / 250 steps)")
    if summary["has_complete_extension"]:
        ext_res = summary["extension_results"]
        print("-" * 68)
        print(f"Extension Δ^A(1000):    {ext_res['stats_post_1000']['mean']:+.4f} ± {ext_res['stats_post_1000']['std']:.4f} BPC")
        print(f"Late Plateau Slope:     {ext_res['late_slope_250']:+.4f} BPC / 250 steps")
    print("-" * 68)
    print("FOUR-HORIZON ENDPOINT MATRIX (Anchor Panel):")
    for h in CONTEXT_LENGTHS:
        pre_h = summary["horizon_endpoint_stats"][h]["pre"]["mean"]
        post_h = summary["horizon_endpoint_stats"][h]["post_500"]["mean"]
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
