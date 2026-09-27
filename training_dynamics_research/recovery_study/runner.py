"""
Execution Runner for Context-Length Recovery Study
Preregistered specification: Section 4, 11, 16, 17

Refactored Architecture:
1. Immutable RunSpec
2. Unified initialize_run(spec) with strict RNG binding before model construction
3. Unidirectional execute_training(state) pipeline (train -> eval -> persist -> checkpoint)
4. Explicit finalize_run(state)
"""

import os
import sys
import json
import time
import argparse
import platform
import subprocess
import glob
from typing import Any, Dict, Optional, Set
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
    SMOKE_RESULTS_DIR,
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
from .core_types import RunSpec, RunState
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


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR).decode().strip()
    except Exception:
        return "unknown"


def set_seed(seed: int):
    """Strictly sets all random number generators across Python, NumPy, CPU PyTorch, and GPU/MPS."""
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


def get_memory_mb() -> float:
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / (1024 * 1024)
    elif torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def atomic_save_json(obj: Any, file_path: str):
    tmp_path = file_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp_path, file_path)


def save_state_checkpoint(state: RunState, step: int):
    """
    Saves a complete execution checkpoint:
    - Model parameters and Adam optimizer state
    - Global step, elapsed time, git commit
    - Evaluation logs including current step
    - Complete RNG states for exact replication
    """
    ckpt_path = os.path.join(state.spec.run_dir, f"checkpoint_step_{step}.pt")
    payload = {
        "global_step": step,
        "arm": state.spec.arm,
        "seed": state.spec.seed,
        "model_state": state.model.state_dict(),
        "optimizer_state": state.optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "mps_rng_state": torch.mps.get_rng_state() if hasattr(torch, "mps") and torch.backends.mps.is_available() else None,
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "logs": state.logs,
        "anchor_logs": state.anchor_logs,
        "sensitivity_logs": state.sensitivity_logs,
        "elapsed_sec": state.elapsed_sec,
        "git_commit": get_git_commit(),
    }
    torch.save(payload, ckpt_path)
    print(f"  [SAVED FULL CKPT] {ckpt_path}")


def make_run_spec(
    arm: str,
    seed: int,
    total_steps: int = MAX_STEPS,
    is_smoke: bool = False,
    is_extension: bool = False,
    results_dir: str = BASE_RESULTS_DIR,
) -> RunSpec:
    return RunSpec(
        arm=arm,
        seed=seed,
        total_steps=total_steps,
        is_smoke=is_smoke,
        is_extension=is_extension,
        results_dir=results_dir,
    )


def initialize_run(spec: RunSpec, device: str = DEVICE, overwrite: bool = False) -> RunState:
    """
    Step 1 of execution pipeline:
    - Guarantees set_seed(spec.seed) is called BEFORE parameter initialization.
    - Instantiates model and optimizer.
    - Handles checkpoint detection and resumption with state restoration.
    """
    os.makedirs(spec.run_dir, exist_ok=True)
    
    # Check if already completed
    if os.path.exists(spec.results_file) and not overwrite:
        try:
            with open(spec.results_file, "r") as f:
                cur_data = json.load(f)
            if cur_data.get("completed", False) and cur_data.get("current_step", 0) >= spec.total_steps:
                # Return dummy state marked complete
                return RunState(
                    spec=spec,
                    model=None,
                    optimizer=None,
                    step=spec.total_steps + 1,
                    is_complete=True,
                )
        except Exception:
            pass

    # 1. CRITICAL: Bind RNG seed before model parameter initialization!
    set_seed(spec.seed)
    
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
    
    state = RunState(
        spec=spec,
        model=model,
        optimizer=optimizer,
        step=1,
        elapsed_sec=0.0,
        logs=[],
        anchor_logs=[],
        sensitivity_logs={},
        is_complete=False,
    )
    
    # 2. Check for checkpoint to resume from
    resume_ckpt = None
    if not overwrite:
        if spec.is_extension:
            ckpt_2500 = os.path.join(spec.run_dir, "checkpoint_step_2500.pt")
            if os.path.exists(ckpt_2500):
                resume_ckpt = ckpt_2500
            else:
                raise FileNotFoundError(
                    f"Cannot extend {spec.arm} seed {spec.seed}: {ckpt_2500} not found. Complete main 2500 steps first."
                )
        else:
            # Look for latest step checkpoint
            existing_ckpts = glob.glob(os.path.join(spec.run_dir, "checkpoint_step_*.pt"))
            if existing_ckpts:
                # Extract highest step
                def parse_step(path):
                    try:
                        return int(os.path.basename(path).split("_step_")[1].split(".pt")[0])
                    except Exception:
                        return -1
                sorted_ckpts = sorted(existing_ckpts, key=parse_step)
                latest = sorted_ckpts[-1]
                if parse_step(latest) > 0:
                    resume_ckpt = latest

    if resume_ckpt and os.path.exists(resume_ckpt):
        print(f"  [RESUMING] Loading checkpoint from {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        state.step = ckpt["global_step"] + 1
        state.logs = ckpt.get("logs", [])
        state.anchor_logs = ckpt.get("anchor_logs", [])
        state.sensitivity_logs = ckpt.get("sensitivity_logs", {})
        state.elapsed_sec = ckpt.get("elapsed_sec", 0.0)
        
        if ckpt.get("torch_rng_state") is not None:
            torch.set_rng_state(ckpt["torch_rng_state"])
        if ckpt.get("numpy_rng_state") is not None:
            np.random.set_state(ckpt["numpy_rng_state"])
        if ckpt.get("mps_rng_state") is not None and hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.set_rng_state(ckpt["mps_rng_state"])
        print(f"  [RESUMED] Resumed at step {state.step}")
        
    return state


def execute_training(state: RunState, data_cache: Dict[str, Any], device: str = DEVICE) -> RunState:
    """
    Step 2 of execution pipeline: Unidirectional training & evaluation loop.
    Enforces monotonic progression:
    train_step -> evaluate_panel -> append_logs -> flush_json -> save_checkpoint.
    """
    if state.is_complete:
        return state

    spec = state.spec
    train_data = data_cache["train_data"]
    sched_manifest = data_cache["sched_manifest"]
    rec_manifest = data_cache["rec_manifest"]
    ext_manifest = data_cache["ext_manifest"]
    x_proc, y_proc = data_cache["process_val"]
    x_anch, y_anch = data_cache["anchor_val"]
    val_start_indices = data_cache.get("val_start_indices")
    
    # Build schedule and evaluation steps
    if spec.is_smoke:
        schedule_info = []
        for blk_idx, T in enumerate(SCHEDULES[spec.arm]):
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
        eval_steps = {1, 10, 20, 30, 40, 41, 50}
        anchor_eval_steps = {40, 50}
    else:
        schedule_info = build_run_schedule(spec.arm, total_steps=spec.total_steps)
        eval_steps = set(PROCESS_EVAL_STEPS)
        anchor_eval_steps = set(ANCHOR_EVAL_STEPS)
        if spec.is_extension:
            eval_steps.update(EXTENSION_PROCESS_EVAL_STEPS)
            anchor_eval_steps.update(EXTENSION_ANCHOR_EVAL_STEPS)

    state.model.train()
    sync_device()
    t_start = time.time()
    accumulated_time = state.elapsed_sec
    
    print(f"\n▶ Starting Run: Arm='{spec.arm}', Seed={spec.seed}, Steps={state.step}..{spec.total_steps}, Device={device}")
    
    for step_cfg in schedule_info:
        step = step_cfg["step"]
        if step < state.step:
            continue
            
        T = step_cfg["T"]
        B = step_cfg["B"]
        is_rec = step_cfg["is_recovery"]
        is_ext = step_cfg["is_extension"]
        occ_idx = step_cfg["occurrence_idx"]
        
        # Manifest selection
        if is_ext:
            start_indices = ext_manifest[occ_idx]
        elif is_rec:
            start_indices = rec_manifest[occ_idx]
        else:
            start_indices = sched_manifest[T][occ_idx]
            
        x_batch = torch.stack([train_data[i : i + T] for i in start_indices]).to(device)
        y_batch = torch.stack([train_data[i + 1 : i + T + 1] for i in start_indices]).to(device)
        
        # Learning rate
        lr = get_lr(step, max_steps=spec.total_steps if spec.is_smoke else MAX_STEPS)
        for pg in state.optimizer.param_groups:
            pg["lr"] = lr
            
        # Optimization step
        sync_device()
        t0 = time.time()
        logits, loss = state.model(x_batch, y_batch)
        state.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(state.model.parameters(), GRAD_CLIP)
        state.optimizer.step()
        sync_device()
        step_duration = time.time() - t0
        
        state.elapsed_sec = accumulated_time + (time.time() - t_start)
        
        # Monotonic Evaluation Pipeline
        if step in eval_steps:
            grad_norm = torch.norm(torch.stack([
                torch.norm(p.grad.detach()) for p in state.model.parameters() if p.grad is not None
            ])).item()
            
            if step in anchor_eval_steps:
                # 1. Forward 32 anchor sequences ONCE
                a_eval = evaluate_panel(
                    state.model, x_anch, y_anch, device, val_start_indices=val_start_indices
                )
                # 2. Slice the first 16 sequences for process panel metrics (zero duplicate forward passes)
                p_eval = slice_process_metrics_from_anchor(a_eval, n_process=N_PROCESS_SEQUENCES)
                
                if a_eval.get("sensitivity") is not None:
                    state.sensitivity_logs[str(step)] = a_eval["sensitivity"]
                    
                anchor_log = {
                    "step": step,
                    "anchor_bpc": a_eval["bpc"],
                    "anchor_seq_bpc": a_eval["seq_bpc"],
                    "horizon_results": a_eval["horizon_results"],
                    "horizon_bpc": {h: a_eval["horizon_results"][h]["mean_bpc"] for h in a_eval["horizon_results"]},
                    "mean_rank": a_eval["mean_rank"],
                    "mean_entropy": a_eval["mean_entropy"],
                }
                state.anchor_logs.append(anchor_log)
            else:
                p_eval = evaluate_panel(state.model, x_proc, y_proc, device)
                
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
                "elapsed_sec": state.elapsed_sec,
                "mem_mb": get_memory_mb(),
            }
            state.logs.append(log_entry)
            
            # Incremental persistence: flush current run results JSON
            partial_results = {
                "schema_version": "1.0",
                "arm": spec.arm,
                "seed": spec.seed,
                "total_steps": spec.total_steps,
                "current_step": step,
                "completed": False,
                "smoke_test": spec.is_smoke,
                "is_extension": spec.is_extension,
                "total_time_sec": state.elapsed_sec,
                "logs": state.logs,
                "anchor_logs": state.anchor_logs,
                "sensitivity_logs": state.sensitivity_logs,
            }
            atomic_save_json(partial_results, spec.results_file)
            
            # Save full checkpoint if this is an anchor evaluation step
            if step in anchor_eval_steps:
                save_state_checkpoint(state, step)
                
            print(
                f"  [{spec.arm:<11}|s{spec.seed}] Step {step:4d}/{spec.total_steps} | "
                f"T={T:3d} | Proc BPC={p_eval['bpc']:.4f} | Train Loss={loss.item():.4f} | Time={step_duration:.3f}s"
            )
            
        state.step = step + 1

    return state


def finalize_run(state: RunState) -> Dict[str, Any]:
    """
    Step 3 of execution pipeline: Mark run as completed and commit final results JSON.
    """
    if state.is_complete and state.model is None:
        # Loaded directly from previous completed run
        with open(state.spec.results_file, "r") as f:
            return json.load(f)
            
    state.is_complete = True
    final_results = {
        "schema_version": "1.0",
        "arm": state.spec.arm,
        "seed": state.spec.seed,
        "total_steps": state.spec.total_steps,
        "current_step": state.spec.total_steps,
        "completed": True,
        "smoke_test": state.spec.is_smoke,
        "is_extension": state.spec.is_extension,
        "total_time_sec": state.elapsed_sec,
        "logs": state.logs,
        "anchor_logs": state.anchor_logs,
        "sensitivity_logs": state.sensitivity_logs,
    }
    atomic_save_json(final_results, state.spec.results_file)
    print(f"✔ Completed Arm='{state.spec.arm}', Seed={state.spec.seed} in {state.elapsed_sec:.1f}s")
    print(f"  [SAVED FINAL RESULTS] {state.spec.results_file}")
    return final_results


def run_single_arm(
    arm: str,
    seed: int,
    train_data: torch.Tensor,
    sched_manifest: Dict[int, np.ndarray],
    rec_manifest: np.ndarray,
    ext_manifest: np.ndarray,
    process_val: Any,
    anchor_val: Any,
    device: str,
    output_dir: str,
    total_steps: int = MAX_STEPS,
    smoke_test: bool = False,
    is_extension: bool = False,
    overwrite: bool = False,
    val_start_indices: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Single arm execution wrapper invoking the 3-step pipeline."""
    spec = RunSpec(
        arm=arm,
        seed=seed,
        total_steps=total_steps,
        is_smoke=smoke_test,
        is_extension=is_extension,
        results_dir=output_dir,
    )
    data_cache = {
        "train_data": train_data,
        "sched_manifest": sched_manifest,
        "rec_manifest": rec_manifest,
        "ext_manifest": ext_manifest,
        "process_val": process_val,
        "anchor_val": anchor_val,
        "val_start_indices": val_start_indices,
    }
    state = initialize_run(spec, device=device, overwrite=overwrite)
    execute_training(state, data_cache, device=device)
    return finalize_run(state)


def run_recovery_study(
    results_dir: str = BASE_RESULTS_DIR,
    seeds: Optional[List[int]] = None,
    target_arm: Optional[str] = None,
    smoke_test: bool = False,
    extend: bool = False,
    device: str = DEVICE,
    overwrite: bool = False,
):
    """
    Top-level orchestrator executing the full 3x3=9 experiment matrix.
    Follows prespecified Latin-square rotation order across seeds.
    """
    if seeds is None:
        seeds = SEEDS

    if smoke_test and results_dir == BASE_RESULTS_DIR:
        results_dir = SMOKE_RESULTS_DIR
        
    os.makedirs(results_dir, exist_ok=True)
    
    print("=== Context-Length Recovery Study Execution ===")
    print(f"Hardware backend: {device}")
    print(f"Results Directory: {results_dir}")
    print(f"Git commit: {get_git_commit()}")
    print(f"Platform: {platform.platform()} | Python: {platform.python_version()} | PyTorch: {torch.__version__}")
    print(f"Seeds: {seeds}")
    print(f"Target arm: {target_arm if target_arm else 'All (rotated)'}")
    print(f"Smoke test: {smoke_test}")
    print(f"Extension mode: {extend}")
    
    # 1. Dataset & Manifest Initialization
    train_data = torch.as_tensor(ds.train_data, dtype=torch.long)
    val_data = torch.as_tensor(ds.val_data, dtype=torch.long)
    train_len = len(train_data)
    val_len = len(val_data)
    
    val_manifest_path = os.path.join(results_dir, "validation_manifest.npz")
    if not os.path.exists(val_manifest_path):
        val_start_indices = generate_validation_manifest(
            n_sequences=N_ANCHOR_SEQUENCES,
            context_length=VAL_CONTEXT,
            val_length=val_len,
        )
        np.savez(
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
    
    # Provenance Configuration
    run_config = {
        "seeds": seeds,
        "arms": ARMS,
        "schedules": SCHEDULES,
        "arm_rotation": ARM_ROTATION,
        "max_steps": HARD_CAP_STEPS if extend else (50 if smoke_test else MAX_STEPS),
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
    required_steps = HARD_CAP_STEPS if extend else (50 if smoke_test else MAX_STEPS)
    
    for seed in target_seeds:
        print(f"\n==========================================")
        print(f"Preparing Manifests for Seed {seed}")
        print(f"==========================================")
        sched_manifest = generate_scheduled_manifest(seed, train_len)
        rec_manifest = generate_recovery_manifest(seed, train_len)
        ext_manifest = generate_extended_recovery_manifest(seed, train_len)
        
        full_rotation = ARM_ROTATION.get(seed, ARMS)
        arm_order = [target_arm] if target_arm else full_rotation
        print(f"Execution order for Seed {seed}: {arm_order}")
        
        for arm in arm_order:
            spec = make_run_spec(
                arm=arm,
                seed=seed,
                total_steps=required_steps,
                is_smoke=smoke_test,
                is_extension=extend,
                results_dir=results_dir,
            )
            state = initialize_run(spec, device=device, overwrite=overwrite)
            if state.is_complete:
                print(f"  [SKIP] Arm {arm} seed {seed} already completed. Use --overwrite to re-run.")
                continue
                
            data_cache = {
                "train_data": train_data,
                "sched_manifest": sched_manifest,
                "rec_manifest": rec_manifest,
                "ext_manifest": ext_manifest,
                "process_val": process_val,
                "anchor_val": anchor_val,
                "val_start_indices": val_start_indices,
            }
            execute_training(state, data_cache, device=device)
            res = finalize_run(state)
            
            if seed not in all_results:
                all_results[seed] = {}
            all_results[seed][arm] = res
            
    print("\nAll runs completed successfully!")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run Context-Length Recovery Study")
    parser.add_argument("--results-dir", type=str, default=BASE_RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=None, help="Run single seed")
    parser.add_argument("--arm", type=str, default=None, choices=ARMS, help="Run single arm")
    parser.add_argument("--smoke-test", action="store_true", help="Run 50-step smoke test")
    parser.add_argument("--extend", action="store_true", help="Run 500-step adaptive extension to step 3000")
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing runs")
    args = parser.parse_args()
    
    seeds = [args.seed] if args.seed is not None else SEEDS
    run_recovery_study(
        results_dir=args.results_dir,
        seeds=seeds,
        target_arm=args.arm,
        smoke_test=args.smoke_test,
        extend=args.extend,
        device=args.device,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
