"""
Execution Runner for Context-Length Recovery Study
Preregistered specification: Section 4, 11, 16, 17
"""

import os
import sys
import json
import time
import argparse
import platform
import subprocess
import numpy as np
import torch

# Ensure repository root is on path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import dataset as ds
from model import MiniTransformerLM
from .config import (
    SEEDS,
    ARMS,
    SCHEDULES,
    ARM_ROTATION,
    BASE_RESULTS_DIR,
    DEVICE,
    LR,
    MIN_LR,
    WEIGHT_DECAY,
    GRAD_CLIP,
    N_EMBD,
    N_HEAD,
    N_LAYER,
    BLOCK_SIZE,
    POS_EMB_TYPE,
    DROPOUT,
    PROCESS_EVAL_STEPS,
    ANCHOR_EVAL_STEPS,
    EXTENSION_PROCESS_EVAL_STEPS,
    EXTENSION_ANCHOR_EVAL_STEPS,
    N_PROCESS_SEQUENCES,
    N_ANCHOR_SEQUENCES,
    VAL_CONTEXT,
    MAX_STEPS,
    HARD_CAP_STEPS,
)
from .schedules import build_run_schedule, get_lr
from .manifests import (
    generate_scheduled_manifest,
    generate_recovery_manifest,
    generate_extended_recovery_manifest,
    generate_validation_manifest,
)
from .evaluation import (
    build_validation_tensors,
    evaluate_panel,
    slice_process_metrics_from_anchor,
)


def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR).decode().strip()
    except Exception:
        return "unknown"


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def sync_device():
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def get_memory_mb():
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / (1024 * 1024)
    elif torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def atomic_save_json(obj, file_path):
    tmp_path = file_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp_path, file_path)


def save_checkpoint(model, optimizer, step, arm, seed, logs, anchor_logs, sensitivity_logs, elapsed_sec, ckpt_path):
    """
    P1 requirement: Full state checkpointing for seamless resume and adaptive extension.
    Preserves model weights, Adam moments, global step, and RNG states.
    """
    payload = {
        "global_step": step,
        "arm": arm,
        "seed": seed,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "mps_rng_state": torch.mps.get_rng_state() if hasattr(torch, "mps") and torch.backends.mps.is_available() else None,
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "logs": logs,
        "anchor_logs": anchor_logs,
        "sensitivity_logs": sensitivity_logs,
        "elapsed_sec": elapsed_sec,
        "git_commit": get_git_commit(),
    }
    torch.save(payload, ckpt_path)
    print(f"  [SAVED FULL CKPT] {ckpt_path}")


def run_single_arm(
    arm,
    seed,
    train_data,
    sched_manifest,
    rec_manifest,
    ext_manifest,
    process_val,
    anchor_val,
    device,
    output_dir,
    total_steps=MAX_STEPS,
    eval_steps=None,
    anchor_eval_steps=None,
    smoke_test=False,
    resume_from=None,
):
    """
    Execute training and evaluation for a single arm and seed.
    Supports complete state checkpointing, resumption, and adaptive extension.
    """
    os.makedirs(output_dir, exist_ok=True)
    res_path = os.path.join(output_dir, "run_results.json")
    
    if eval_steps is None:
        eval_steps = set(PROCESS_EVAL_STEPS)
    else:
        eval_steps = set(eval_steps)
        
    if anchor_eval_steps is None:
        anchor_eval_steps = set(ANCHOR_EVAL_STEPS)
    else:
        anchor_eval_steps = set(anchor_eval_steps)
        
    x_proc, y_proc = process_val
    x_anch, y_anch = anchor_val
    
    vocab_size = len(ds.chars)
    model = MiniTransformerLM(
        vocab_size=vocab_size,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        block_size=BLOCK_SIZE,
        pos_emb_type=POS_EMB_TYPE,
        dropout=DROPOUT,
    ).to(device)
    
    optimizer = model.configure_optimizers(weight_decay=WEIGHT_DECAY, learning_rate=LR)
    
    logs = []
    anchor_logs = []
    sensitivity_logs = {}
    start_step = 1
    accumulated_time = 0.0
    
    # Check if resuming from checkpoint
    if resume_from and os.path.exists(resume_from):
        print(f"  [RESUMING] Loading checkpoint from {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step = ckpt["global_step"] + 1
        logs = ckpt.get("logs", [])
        anchor_logs = ckpt.get("anchor_logs", [])
        sensitivity_logs = ckpt.get("sensitivity_logs", {})
        accumulated_time = ckpt.get("elapsed_sec", 0.0)
        if ckpt.get("torch_rng_state") is not None:
            torch.set_rng_state(ckpt["torch_rng_state"])
        if ckpt.get("numpy_rng_state") is not None:
            np.random.set_state(ckpt["numpy_rng_state"])
        if ckpt.get("mps_rng_state") is not None and hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.set_rng_state(ckpt["mps_rng_state"])
        print(f"  [RESUMED] Resumed at step {start_step}")
    else:
        set_seed(seed)
        
    if smoke_test:
        schedule_info = []
        for blk_idx, T in enumerate(SCHEDULES[arm]):
            B = 4096 // T
            for s in range(10):
                schedule_info.append({
                    "step": len(schedule_info) + 1,
                    "T": T,
                    "B": B,
                    "is_recovery": False,
                    "is_extension": False,
                    "occurrence_idx": s,
                })
        for s in range(10):
            schedule_info.append({
                "step": len(schedule_info) + 1,
                "T": 256,
                "B": 16,
                "is_recovery": True,
                "is_extension": False,
                "occurrence_idx": s,
            })
        total_steps = len(schedule_info)
        eval_steps = {1, 10, 20, 30, 40, 41, 50}
        anchor_eval_steps = {40, 50}
    else:
        schedule_info = build_run_schedule(arm, total_steps=total_steps)
        
    model.train()
    sync_device()
    t_start = time.time()
    
    print(f"\n▶ Starting Run: Arm='{arm}', Seed={seed}, Steps={start_step}..{total_steps}, Device={device}")
    
    for step_cfg in schedule_info:
        step = step_cfg["step"]
        if step < start_step:
            continue
            
        T = step_cfg["T"]
        B = step_cfg["B"]
        is_rec = step_cfg["is_recovery"]
        is_ext = step_cfg["is_extension"]
        occ_idx = step_cfg["occurrence_idx"]
        
        # P1 fix: hook up extended_manifest when is_extension is True
        if is_ext:
            start_indices = ext_manifest[occ_idx]
        elif is_rec:
            start_indices = rec_manifest[occ_idx]
        else:
            start_indices = sched_manifest[T][occ_idx]
            
        x_batch = torch.stack([train_data[i : i + T] for i in start_indices]).to(device)
        y_batch = torch.stack([train_data[i + 1 : i + T + 1] for i in start_indices]).to(device)
        
        # Learning rate
        lr = get_lr(step, max_steps=total_steps if smoke_test else MAX_STEPS)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
            
        # Forward & Backward
        sync_device()
        t0 = time.time()
        logits, loss = model(x_batch, y_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        sync_device()
        step_duration = time.time() - t0
        
        # Checkpoint evaluation (P2 fix: zero duplicate forwards)
        if step in eval_steps:
            elapsed = accumulated_time + (time.time() - t_start)
            grad_norm = torch.norm(torch.stack([
                torch.norm(p.grad.detach()) for p in model.parameters() if p.grad is not None
            ])).item()
            
            if step in anchor_eval_steps:
                # 1. Forward 32 anchor sequences ONCE
                a_eval = evaluate_panel(model, x_anch, y_anch, device)
                
                # 2. Slice the first 16 sequences for process panel metrics (zero duplicate forwards!)
                p_eval = slice_process_metrics_from_anchor(a_eval, n_process=N_PROCESS_SEQUENCES)
                
                # Context sensitivity comes directly from a_eval
                if a_eval.get("sensitivity") is not None:
                    sensitivity_logs[str(step)] = a_eval["sensitivity"]
                    
                anchor_log = {
                    "step": step,
                    "anchor_bpc": a_eval["bpc"],
                    "anchor_seq_bpc": a_eval["seq_bpc"],
                    "horizon_results": a_eval["horizon_results"],
                    "horizon_bpc": {h: a_eval["horizon_results"][h]["mean_bpc"] for h in a_eval["horizon_results"]},
                    "mean_rank": a_eval["mean_rank"],
                    "mean_entropy": a_eval["mean_entropy"],
                }
                anchor_logs.append(anchor_log)
                
                # Save full state checkpoint (weights + Adam moments + RNG states)
                ckpt_path = os.path.join(output_dir, f"checkpoint_step_{step}.pt")
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    arm=arm,
                    seed=seed,
                    logs=logs,
                    anchor_logs=anchor_logs,
                    sensitivity_logs=sensitivity_logs,
                    elapsed_sec=elapsed,
                    ckpt_path=ckpt_path,
                )
            else:
                # Routine checkpoint: evaluate only the 16 process sequences
                p_eval = evaluate_panel(model, x_proc, y_proc, device)
                
            log_entry = {
                "step": step,
                "T": T,
                "B": B,
                "lr": lr,
                "train_loss": loss.item(),
                "process_bpc": p_eval["bpc"],
                "process_seq_bpc": p_eval["seq_bpc"],
                "horizon_results": p_eval["horizon_results"],
                "horizon_bpc": {h: p_eval["horizon_results"][h]["mean_bpc"] for h in p_eval["horizon_results"]},
                "mean_rank": p_eval["mean_rank"],
                "mean_entropy": p_eval["mean_entropy"],
                "grad_norm": grad_norm,
                "step_sec": step_duration,
                "elapsed_sec": elapsed,
                "mem_mb": get_memory_mb(),
            }
            logs.append(log_entry)
            
            # Incremental persistence: flush results after each evaluation checkpoint
            partial_results = {
                "arm": arm,
                "seed": seed,
                "total_steps": total_steps,
                "current_step": step,
                "total_time_sec": elapsed,
                "logs": logs,
                "anchor_logs": anchor_logs,
                "sensitivity_logs": sensitivity_logs,
            }
            atomic_save_json(partial_results, res_path)
            
            print(
                f"  [{arm:12s}|s{seed}] Step {step:4d}/{total_steps} | T={T:3d} | "
                f"Proc BPC={p_eval['bpc']:.4f} | "
                f"Train Loss={loss.item():.4f} | "
                f"Time={step_duration:.3f}s",
                flush=True
            )
            
    total_time = accumulated_time + (time.time() - t_start)
    print(f"✔ Completed Arm='{arm}', Seed={seed} in {total_time:.1f}s")
    
    results = {
        "arm": arm,
        "seed": seed,
        "total_steps": total_steps,
        "total_time_sec": total_time,
        "logs": logs,
        "anchor_logs": anchor_logs,
        "sensitivity_logs": sensitivity_logs,
    }
    atomic_save_json(results, res_path)
    print(f"  [SAVED FINAL RESULTS] {res_path}")
    
    return results


def run_experiment(
    smoke_test=False,
    seeds=SEEDS,
    target_arm=None,
    results_dir=BASE_RESULTS_DIR,
    extend=False,
    overwrite=False,
):
    """
    Sequential execution manager with rotation order, full provenance, and adaptive extension support.
    """
    os.makedirs(results_dir, exist_ok=True)
    train_data = ds.train_data
    val_data = ds.val_data
    train_len = len(train_data)
    val_len = len(val_data)
    
    device = DEVICE if torch.backends.mps.is_available() else "cpu"
    print(f"=== Context-Length Recovery Study Execution ===")
    print(f"Hardware backend: {device}")
    print(f"Git commit: {get_git_commit()}")
    print(f"Platform: {platform.platform()} | Python: {platform.python_version()} | PyTorch: {torch.__version__}")
    print(f"Seeds: {seeds}")
    print(f"Target arm: {target_arm if target_arm else 'All (rotated)'}")
    print(f"Smoke test: {smoke_test}")
    print(f"Extension mode: {extend}")
    
    # 1. Validation Manifest
    val_manifest_path = os.path.join(results_dir, "validation_manifest.npz")
    if not os.path.exists(val_manifest_path):
        val_start_indices = generate_validation_manifest(val_length=val_len)
        np.savez_compressed(
            val_manifest_path,
            anchor_indices=val_start_indices,
            process_indices=val_start_indices[:N_PROCESS_SEQUENCES],
        )
        print(f"Saved validation manifest to {val_manifest_path}")
    else:
        npz = np.load(val_manifest_path)
        val_start_indices = npz["anchor_indices"]
        print(f"Loaded existing validation manifest from {val_manifest_path}")
        
    process_val = build_validation_tensors(val_data, val_start_indices[:N_PROCESS_SEQUENCES], device)
    anchor_val = build_validation_tensors(val_data, val_start_indices, device)
    
    # Full Provenance in run_config.json
    run_config = {
        "seeds": seeds,
        "arms": ARMS,
        "schedules": SCHEDULES,
        "arm_rotation": ARM_ROTATION,
        "max_steps": HARD_CAP_STEPS if extend else MAX_STEPS,
        "smoke_test": smoke_test,
        "extend_mode": extend,
        "device": device,
        "git_commit": get_git_commit(),
        "preregistration_commit": "f1bf7b7",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "dataset_train_len": train_len,
        "dataset_val_len": val_len,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    config_path = os.path.join(results_dir, "run_config.json")
    atomic_save_json(run_config, config_path)
    
    # 2. Sequential execution across seeds following rotation order
    all_results = {}
    target_seeds = [42] if smoke_test else seeds
    
    for seed in target_seeds:
        print(f"\n==========================================")
        print(f"Preparing Manifests for Seed {seed}")
        print(f"==========================================")
        sched_manifest = generate_scheduled_manifest(seed, train_len)
        rec_manifest = generate_recovery_manifest(seed, train_len)
        ext_manifest = generate_extended_recovery_manifest(seed, train_len)
        
        # Determine arm order
        full_rotation = ARM_ROTATION.get(seed, ARMS)
        if target_arm:
            if target_arm not in ARMS:
                raise ValueError(f"Unknown arm '{target_arm}'. Expected one of {ARMS}")
            arm_order = [target_arm]
        else:
            arm_order = full_rotation
            
        print(f"Execution order for Seed {seed}: {arm_order}")
        
        for arm in arm_order:
            arm_dir = os.path.join(results_dir, f"seed_{seed}", arm)
            res_file = os.path.join(arm_dir, "run_results.json")
            
            # Extension mode setup
            if extend:
                ckpt_2500 = os.path.join(arm_dir, "checkpoint_step_2500.pt")
                if not os.path.exists(ckpt_2500):
                    raise FileNotFoundError(f"Cannot extend arm {arm} seed {seed}: {ckpt_2500} not found. Run main 2500 steps first.")
                
                # Check if already completed extension
                if os.path.exists(res_file) and not overwrite:
                    with open(res_file, "r") as f:
                        cur_data = json.load(f)
                    if cur_data.get("current_step", 0) >= HARD_CAP_STEPS:
                        print(f"  [SKIP] Arm {arm} seed {seed} already extended to {HARD_CAP_STEPS}. Use --overwrite to re-run.")
                        continue
                        
                res = run_single_arm(
                    arm=arm,
                    seed=seed,
                    train_data=train_data,
                    sched_manifest=sched_manifest,
                    rec_manifest=rec_manifest,
                    ext_manifest=ext_manifest,
                    process_val=process_val,
                    anchor_val=anchor_val,
                    device=device,
                    output_dir=arm_dir,
                    total_steps=HARD_CAP_STEPS,
                    eval_steps=set(PROCESS_EVAL_STEPS + EXTENSION_PROCESS_EVAL_STEPS),
                    anchor_eval_steps=set(ANCHOR_EVAL_STEPS + EXTENSION_ANCHOR_EVAL_STEPS),
                    smoke_test=False,
                    resume_from=ckpt_2500,
                )
            else:
                # Normal 2500-step run
                if os.path.exists(res_file) and not overwrite and not smoke_test:
                    with open(res_file, "r") as f:
                        cur_data = json.load(f)
                    if cur_data.get("current_step", 0) >= MAX_STEPS:
                        print(f"  [SKIP] Arm {arm} seed {seed} already completed {MAX_STEPS} steps. Use --overwrite to re-run.")
                        continue
                        
                res = run_single_arm(
                    arm=arm,
                    seed=seed,
                    train_data=train_data,
                    sched_manifest=sched_manifest,
                    rec_manifest=rec_manifest,
                    ext_manifest=ext_manifest,
                    process_val=process_val,
                    anchor_val=anchor_val,
                    device=device,
                    output_dir=arm_dir,
                    total_steps=MAX_STEPS,
                    smoke_test=smoke_test,
                )
                
            all_results[f"seed_{seed}_{arm}"] = res
            
    print(f"\nAll runs completed successfully!")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Context-Length Recovery Study")
    parser.add_argument("--smoke-test", action="store_true", help="Run 10-step smoke test for verification")
    parser.add_argument("--seed", type=int, default=None, help="Run only a specific seed")
    parser.add_argument("--arm", type=str, default=None, help="Run only a specific arm")
    parser.add_argument("--extend", action="store_true", help="Trigger preregistered one-time extension (2501..3000 steps)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing completed run results")
    args = parser.parse_args()
    
    if args.seed is not None:
        target_seeds = [args.seed]
    else:
        target_seeds = SEEDS
        
    run_experiment(
        smoke_test=args.smoke_test,
        seeds=target_seeds,
        target_arm=args.arm,
        extend=args.extend,
        overwrite=args.overwrite,
    )
