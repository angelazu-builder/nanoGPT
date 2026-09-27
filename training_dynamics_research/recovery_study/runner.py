"""
Execution Runner for Context-Length Recovery Study
Preregistered specification: Section 4, 11, 17
"""

import os
import sys
import json
import time
import argparse
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
    N_PROCESS_SEQUENCES,
    N_ANCHOR_SEQUENCES,
    VAL_CONTEXT,
    MAX_STEPS,
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
    evaluate_context_sensitivity,
)


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


def run_single_arm(
    arm,
    seed,
    train_data,
    sched_manifest,
    rec_manifest,
    process_val,
    anchor_val,
    device,
    output_dir,
    total_steps=MAX_STEPS,
    eval_steps=None,
    anchor_eval_steps=None,
    smoke_test=False,
):
    """
    Execute training and evaluation for a single arm and seed.
    """
    os.makedirs(output_dir, exist_ok=True)
    set_seed(seed)
    
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
    
    if smoke_test:
        # 10 steps per block, 10 recovery steps
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
        
    logs = []
    anchor_logs = []
    sensitivity_logs = {}
    
    model.train()
    sync_device()
    t_start = time.time()
    last_step_time = t_start
    
    print(f"\n▶ Starting Run: Arm='{arm}', Seed={seed}, Steps={total_steps}, Device={device}")
    
    for step_cfg in schedule_info:
        step = step_cfg["step"]
        T = step_cfg["T"]
        B = step_cfg["B"]
        is_rec = step_cfg["is_recovery"]
        occ_idx = step_cfg["occurrence_idx"]
        
        # Manifest draw
        if not is_rec:
            start_indices = sched_manifest[T][occ_idx]
        else:
            start_indices = rec_manifest[occ_idx]
            
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
        
        # Checkpoint evaluation
        if step in eval_steps:
            # 1. Process evaluation (16 sequences)
            p_eval = evaluate_panel(model, x_proc, y_proc, device)
            
            grad_norm = torch.norm(torch.stack([
                torch.norm(p.grad.detach()) for p in model.parameters() if p.grad is not None
            ])).item()
            
            log_entry = {
                "step": step,
                "T": T,
                "B": B,
                "lr": lr,
                "train_loss": loss.item(),
                "process_bpc": p_eval["bpc"],
                "process_seq_bpc": p_eval["seq_bpc"],
                "horizon_bpc": {h: p_eval["horizon_results"][h]["mean_bpc"] for h in p_eval["horizon_results"]},
                "mean_rank": p_eval["mean_rank"],
                "mean_entropy": p_eval["mean_entropy"],
                "grad_norm": grad_norm,
                "step_sec": step_duration,
                "elapsed_sec": time.time() - t_start,
                "mem_mb": get_memory_mb(),
            }
            logs.append(log_entry)
            
            # 2. Anchor evaluation (32 sequences) at anchor steps
            if step in anchor_eval_steps:
                a_eval = evaluate_panel(model, x_anch, y_anch, device)
                sens_eval = evaluate_context_sensitivity(model, x_anch, y_anch, device)
                
                anchor_log = {
                    "step": step,
                    "anchor_bpc": a_eval["bpc"],
                    "anchor_seq_bpc": a_eval["seq_bpc"],
                    "horizon_bpc": {h: a_eval["horizon_results"][h]["mean_bpc"] for h in a_eval["horizon_results"]},
                    "mean_rank": a_eval["mean_rank"],
                    "mean_entropy": a_eval["mean_entropy"],
                }
                anchor_logs.append(anchor_log)
                sensitivity_logs[str(step)] = sens_eval
                
                # Save model weights checkpoint at anchor step
                ckpt_path = os.path.join(output_dir, f"checkpoint_step_{step}.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"  [SAVED CKPT] {ckpt_path}")
                
            print(
                f"  [{arm:12s}|s{seed}] Step {step:4d}/{total_steps} | T={T:3d} | "
                f"Proc BPC={p_eval['bpc']:.4f} | "
                f"Train Loss={loss.item():.4f} | "
                f"Time={step_duration:.3f}s",
                flush=True
            )
            
    total_time = time.time() - t_start
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
    
    res_path = os.path.join(output_dir, "run_results.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  [SAVED RESULTS] {res_path}")
    
    return results


def run_experiment(smoke_test=False, seeds=SEEDS, results_dir=BASE_RESULTS_DIR):
    """
    Sequential execution manager with rotation order (Section 17).
    """
    os.makedirs(results_dir, exist_ok=True)
    train_data = ds.train_data
    val_data = ds.val_data
    train_len = len(train_data)
    val_len = len(val_data)
    
    device = DEVICE if torch.backends.mps.is_available() else "cpu"
    print(f"=== Context-Length Recovery Study Execution ===")
    print(f"Hardware backend: {device}")
    print(f"Seeds: {seeds}")
    print(f"Smoke test: {smoke_test}")
    
    # 1. Generate & Save fixed validation manifest
    val_start_indices = generate_validation_manifest(val_length=val_len)
    val_manifest_path = os.path.join(results_dir, "validation_manifest.npz")
    np.savez_compressed(
        val_manifest_path,
        anchor_indices=val_start_indices,
        process_indices=val_start_indices[:N_PROCESS_SEQUENCES],
    )
    print(f"Saved validation manifest to {val_manifest_path}")
    
    # Build validation tensors
    process_val = build_validation_tensors(val_data, val_start_indices[:N_PROCESS_SEQUENCES], device)
    anchor_val = build_validation_tensors(val_data, val_start_indices, device)
    
    # Save run configuration
    run_config = {
        "seeds": seeds,
        "arms": ARMS,
        "schedules": SCHEDULES,
        "arm_rotation": ARM_ROTATION,
        "max_steps": MAX_STEPS,
        "smoke_test": smoke_test,
        "device": device,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    config_path = os.path.join(results_dir, "run_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
        
    # 2. Sequential execution across seeds following rotation order
    all_results = {}
    
    target_seeds = [42] if smoke_test else seeds
    
    for seed in target_seeds:
        print(f"\n==========================================")
        print(f"Generating Manifests for Seed {seed}")
        print(f"==========================================")
        sched_manifest = generate_scheduled_manifest(seed, train_len)
        rec_manifest = generate_recovery_manifest(seed, train_len)
        
        arm_order = ARM_ROTATION.get(seed, ARMS)
        print(f"Execution order for Seed {seed}: {arm_order}")
        
        for arm in arm_order:
            arm_dir = os.path.join(results_dir, f"seed_{seed}", arm)
            res = run_single_arm(
                arm=arm,
                seed=seed,
                train_data=train_data,
                sched_manifest=sched_manifest,
                rec_manifest=rec_manifest,
                process_val=process_val,
                anchor_val=anchor_val,
                device=device,
                output_dir=arm_dir,
                smoke_test=smoke_test,
            )
            all_results[f"seed_{seed}_{arm}"] = res
            
    print(f"\nAll {len(all_results)} runs completed successfully!")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Context-Length Recovery Study")
    parser.add_argument("--smoke-test", action="store_true", help="Run 10-step smoke test for verification")
    parser.add_argument("--seed", type=int, default=None, help="Run only a specific seed")
    parser.add_argument("--arm", type=str, default=None, help="Run only a specific arm")
    args = parser.parse_args()
    
    if args.seed is not None:
        target_seeds = [args.seed]
    else:
        target_seeds = SEEDS
        
    run_experiment(smoke_test=args.smoke_test, seeds=target_seeds)
