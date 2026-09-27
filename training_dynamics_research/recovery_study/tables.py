"""
Canonical Tidy Table extraction and transformation.
Converts run outputs into typed, flat tabular data frames (lists of records).
Provides the single source of truth for both analyze.py and figures.py.
"""

import os
import csv
import math
from typing import Any, Dict, List, Optional
import numpy as np
from scipy import stats

from .core_types import StudyTables
from .config import CONTEXT_LENGTHS, ARMS


def extract_study_tables(study_data: Dict[int, Dict[str, Any]], active_seeds: List[int]) -> StudyTables:
    """
    Extract canonical tidy tables from nested study dictionary.
    Guarantees consistent column names and types across all consumers.
    """
    tables = StudyTables()
    
    # 1. Process Horizon BPC
    for seed in active_seeds:
        for arm in ARMS:
            if arm not in study_data[seed]:
                continue
            run = study_data[seed][arm]
            for log in run.get("logs", []):
                step = log["step"]
                h_res = log.get("horizon_results", {})
                for h in CONTEXT_LENGTHS:
                    h_val = h if h in h_res else str(h)
                    if h_val in h_res:
                        mean_bpc = float(h_res[h_val]["mean_bpc"])
                        tables._proc_map[(seed, arm, step, h)] = mean_bpc
                        tables.process_horizon_records.append({
                            "seed": seed,
                            "arm": arm,
                            "step": step,
                            "evaluation_horizon": h,
                            "mean_bpc": mean_bpc,
                        })
                        
    # 2. Anchor Horizon BPC & 3. Anchor Sequence BPC
    for seed in active_seeds:
        for arm in ARMS:
            if arm not in study_data[seed]:
                continue
            run = study_data[seed][arm]
            for a_log in run.get("anchor_logs", []):
                step = a_log["step"]
                h_res = a_log.get("horizon_results", {})
                for h in CONTEXT_LENGTHS:
                    h_val = h if h in h_res else str(h)
                    if h_val in h_res:
                        seqs = [float(v) for v in h_res[h_val].get("seq_bpc", [])]
                        n = len(seqs)
                        mean_b = float(np.mean(seqs))
                        std_b = float(np.std(seqs, ddof=1)) if n > 1 else 0.0
                        se_b = std_b / math.sqrt(n) if n > 1 else 0.0
                        if n > 1:
                            t_crit = stats.t.ppf(0.975, df=n - 1)
                            ci_low = mean_b - t_crit * se_b
                            ci_high = mean_b + t_crit * se_b
                        else:
                            ci_low, ci_high = mean_b, mean_b
                            
                        tables._anch_map[(seed, arm, step, h)] = mean_b
                        tables.anchor_horizon_records.append({
                            "seed": seed,
                            "arm": arm,
                            "step": step,
                            "evaluation_horizon": h,
                            "mean_bpc": mean_b,
                            "std_bpc": std_b,
                            "se_bpc": se_b,
                            "ci_95_lower": float(ci_low),
                            "ci_95_upper": float(ci_high),
                            "n_sequences": n,
                        })
                        
                        for seq_idx, bpc_val in enumerate(seqs):
                            tables.anchor_sequence_records.append({
                                "seed": seed,
                                "arm": arm,
                                "step": step,
                                "evaluation_horizon": h,
                                "sequence_index": seq_idx,
                                "bpc": bpc_val,
                            })

    # 4. Recovery Contrasts at Anchor Steps
    seed_0_arms = study_data[active_seeds[0]]
    sample_arm = seed_0_arms.get("ascending", next(iter(seed_0_arms.values()))) if seed_0_arms else {}
    anchor_steps = sorted([l["step"] for l in sample_arm.get("anchor_logs", [])])
    
    for seed in active_seeds:
        if not all(arm in study_data[seed] for arm in ["ascending", "descending", "nonmonotonic"]):
            continue
        for step in anchor_steps:
            asc_log = next((l for l in study_data[seed]["ascending"].get("anchor_logs", []) if l["step"] == step), None)
            desc_log = next((l for l in study_data[seed]["descending"].get("anchor_logs", []) if l["step"] == step), None)
            nonm_log = next((l for l in study_data[seed]["nonmonotonic"].get("anchor_logs", []) if l["step"] == step), None)
            
            if not (asc_log and desc_log and nonm_log):
                continue
                
            for h in CONTEXT_LENGTHS:
                h_val = h if h in asc_log["horizon_results"] else str(h)
                bpc_asc = asc_log["horizon_results"][h_val]["mean_bpc"]
                bpc_desc = desc_log["horizon_results"][h_val]["mean_bpc"]
                bpc_nonm = nonm_log["horizon_results"][h_val]["mean_bpc"]
                
                tables.recovery_contrast_records.append({
                    "seed": seed,
                    "step": step,
                    "evaluation_horizon": h,
                    "contrast": "descending_minus_ascending",
                    "bpc_descending": bpc_desc,
                    "bpc_ascending": bpc_asc,
                    "bpc_nonmonotonic": bpc_nonm,
                    "delta_descending_minus_ascending": bpc_desc - bpc_asc,
                    "delta_nonmonotonic_minus_ascending": bpc_nonm - bpc_asc,
                    "delta_descending_minus_nonmonotonic": bpc_desc - bpc_nonm,
                })

    # 5. Transition Shocks
    sample_logs = sample_arm.get("logs", [])
    checkpoints = {l["step"] for l in sample_logs}
    candidate_pairs = [
        (500, 501), (1000, 1001), (1500, 1501), (2000, 2001),
        (10, 11), (20, 21), (30, 31), (40, 41)
    ]
    valid_pairs = [p for p in candidate_pairs if p[0] in checkpoints and p[1] in checkpoints]
    
    for s_pre, s_post in valid_pairs:
        pair_label = f"{s_pre}->{s_post}"
        for seed in active_seeds:
            for arm in ARMS:
                if arm not in study_data[seed]:
                    continue
                logs = study_data[seed][arm].get("logs", [])
                l_pre = next((l for l in logs if l["step"] == s_pre), None)
                l_post = next((l for l in logs if l["step"] == s_post), None)
                if not (l_pre and l_post):
                    continue
                for h in CONTEXT_LENGTHS:
                    h_val = h if h in l_pre["horizon_bpc"] else str(h)
                    v_pre = l_pre["horizon_bpc"].get(h_val)
                    v_post = l_post["horizon_bpc"].get(h_val)
                    if v_pre is not None and v_post is not None:
                        tables.transition_shock_records.append({
                            "seed": seed,
                            "arm": arm,
                            "transition": pair_label,
                            "step_pre": s_pre,
                            "step_post": s_post,
                            "evaluation_horizon": h,
                            "shock_bpc": float(v_post - v_pre),
                        })

    return tables


def export_tidy_csvs(tables: StudyTables, output_dir: str):
    """Save all five canonical tidy tables to CSV files."""
    data_dir = os.path.join(output_dir, "data") if not output_dir.endswith("data") else output_dir
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. process_horizon_bpc.csv
    if tables.process_horizon_records:
        path = os.path.join(data_dir, "process_horizon_bpc.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["seed", "arm", "step", "evaluation_horizon", "mean_bpc"])
            writer.writeheader()
            writer.writerows(tables.process_horizon_records)
            
    # 2. anchor_horizon_bpc.csv
    if tables.anchor_horizon_records:
        path = os.path.join(data_dir, "anchor_horizon_bpc.csv")
        fieldnames = ["seed", "arm", "step", "evaluation_horizon", "mean_bpc", "std_bpc", "se_bpc", "ci_95_lower", "ci_95_upper", "n_sequences"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(tables.anchor_horizon_records)
            
    # 3. anchor_sequence_bpc.csv
    if tables.anchor_sequence_records:
        path = os.path.join(data_dir, "anchor_sequence_bpc.csv")
        fieldnames = ["seed", "arm", "step", "evaluation_horizon", "sequence_index", "bpc"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(tables.anchor_sequence_records)
            
    # 4. recovery_contrasts.csv
    if tables.recovery_contrast_records:
        path = os.path.join(data_dir, "recovery_contrasts.csv")
        fieldnames = [
            "seed", "step", "evaluation_horizon", "contrast",
            "bpc_descending", "bpc_ascending", "bpc_nonmonotonic",
            "delta_descending_minus_ascending", "delta_nonmonotonic_minus_ascending",
            "delta_descending_minus_nonmonotonic"
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(tables.recovery_contrast_records)
            
    # 5. transition_shocks.csv
    if tables.transition_shock_records:
        path = os.path.join(data_dir, "transition_shocks.csv")
        fieldnames = ["seed", "arm", "transition", "step_pre", "step_post", "evaluation_horizon", "shock_bpc"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(tables.transition_shock_records)
