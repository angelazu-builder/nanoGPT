"""
Schedule implementations for Context-Length Recovery Study
Preregistered specification: Section 8 & Section 5
"""

import math
from .config import (
    SCHEDULES,
    BLOCK_STEPS,
    PRE_RECOVERY_STEPS,
    RECOVERY_STEPS,
    MAX_STEPS,
    BATCH_SIZES,
    WARMUP_STEPS,
    LR,
    MIN_LR,
)


def build_scheduled_contexts(arm):
    """
    Build 2000-step pre-recovery context schedule for the given arm.
    """
    if arm not in SCHEDULES:
        raise ValueError(f"Unknown arm: {arm}. Expected one of {list(SCHEDULES.keys())}")
    
    contexts = []
    for context_length in SCHEDULES[arm]:
        contexts.extend([context_length] * BLOCK_STEPS)
    assert len(contexts) == PRE_RECOVERY_STEPS, f"Expected {PRE_RECOVERY_STEPS} contexts, got {len(contexts)}"
    return contexts


def get_lr(step, max_steps=MAX_STEPS, warmup_steps=WARMUP_STEPS, lr=LR, min_lr=MIN_LR):
    """
    Learning rate schedule:
    - Linear warmup for steps 1..warmup_steps
    - Cosine decay down to min_lr at max_steps (step 2500)
    - For steps > max_steps (e.g. extended recovery 2501..3000), clamped at min_lr.
    """
    if step < warmup_steps:
        return lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


def build_run_schedule(arm, total_steps=MAX_STEPS):
    """
    Build the full list of step specifications for an arm run.
    Each element contains:
      - step: 1-indexed global step (1 .. total_steps)
      - T: context length (32, 64, 128, 256)
      - B: batch size (4096 // T)
      - is_recovery: boolean indicating whether step is in recovery stage (>2000)
      - is_extension: boolean indicating whether step is in extended recovery (>2500)
      - occurrence_idx: index into the corresponding manifest pool (0..499)
    """
    pre_contexts = build_scheduled_contexts(arm)
    steps_info = []
    
    # Counter for scheduled draws per context length (0..499)
    scheduled_counters = {32: 0, 64: 0, 128: 0, 256: 0}
    
    for step in range(1, total_steps + 1):
        if step <= PRE_RECOVERY_STEPS:
            T = pre_contexts[step - 1]
            B = BATCH_SIZES[T]
            occ_idx = scheduled_counters[T]
            scheduled_counters[T] += 1
            steps_info.append({
                "step": step,
                "T": T,
                "B": B,
                "is_recovery": False,
                "is_extension": False,
                "occurrence_idx": occ_idx,
            })
        elif step <= MAX_STEPS:
            # Standard recovery stage (steps 2001..2500)
            T = 256
            B = BATCH_SIZES[T]
            recovery_idx = step - PRE_RECOVERY_STEPS - 1  # 0..499
            steps_info.append({
                "step": step,
                "T": T,
                "B": B,
                "is_recovery": True,
                "is_extension": False,
                "occurrence_idx": recovery_idx,
            })
        else:
            # Extended recovery stage (steps 2501..3000)
            T = 256
            B = BATCH_SIZES[T]
            ext_idx = step - MAX_STEPS - 1  # 0..499
            steps_info.append({
                "step": step,
                "T": T,
                "B": B,
                "is_recovery": True,
                "is_extension": True,
                "occurrence_idx": ext_idx,
            })
            
    return steps_info
