"""
CE-3: Paired Batch Manifest Experiment — 5 Seeds × 4 Arms
==========================================================
Fixes:
  - Bug #3: Pre-generated batch manifest ensures all four arms see
    IDENTICAL text spans at each context length (only order differs).
  - Bug #2: Fixed-target context sensitivity probe — always scores
    only the last 32 target positions regardless of context horizon.

Design:
  - For each seed s in {42,43,44,45,46}:
      * Pre-sample 750 start-indices per context length {32,64,128,256}
        using np.random.seed(s * 1000 + T). This is the shared manifest.
      * All 4 arms draw from this pool in their respective order.
  - Report: mean ± std BPC across 5 seeds; paired t-test Curriculum vs each arm.

CE-2 fix is embedded in evaluate_model_v2() below.
"""

import os
import sys
import json
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

# Force unbuffered output so progress is visible even when piped
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import dataset as ds
from model import MiniTransformerLM


# ──────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def compute_effective_rank(tensor_2d):
    if tensor_2d.ndim > 2:
        tensor_2d = tensor_2d.reshape(-1, tensor_2d.size(-1))
    tensor_cpu = tensor_2d.detach().cpu().float()
    _, S, _ = torch.linalg.svd(tensor_cpu, full_matrices=False)
    S_sum = torch.sum(S)
    if S_sum < 1e-12:
        return 1.0
    p = S / S_sum
    p = p[p > 1e-12]
    entropy = -torch.sum(p * torch.log(p)).item()
    return math.exp(entropy)


def compute_normalized_entropy(att_matrix):
    B, nh, T, _ = att_matrix.shape
    seq_lens = torch.arange(1, T + 1, device=att_matrix.device).float()
    h_max = torch.mean(torch.log(seq_lens)).item()
    if h_max < 1e-6:
        return 1.0
    p = att_matrix + 1e-12
    ent_per_token = -torch.sum(att_matrix * torch.log(p), dim=-1)
    avg_ent = torch.mean(ent_per_token).item()
    return min(1.0, max(0.0, avg_ent / h_max))


def get_lr(it, max_iters=2500, warmup_iters=400, lr=1e-3, min_lr=1e-4):
    if it < warmup_iters:
        return lr * it / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


# ──────────────────────────────────────────────────────────────────────
# CE-2 FIX: evaluate_model with fixed last-32 target positions
# ──────────────────────────────────────────────────────────────────────

FIXED_TARGET_LEN = 32   # always score only the last 32 target positions

@torch.no_grad()
def evaluate_model_v2(model, val_batch, device):
    """
    CE-2 fix: context sensitivity probe uses fixed last-32 target positions.

    For each context horizon h in [32, 64, 128, 256]:
      - Feed x_val[:, -h:]  (h chars of history)
      - Compute loss ONLY on the last FIXED_TARGET_LEN=32 positions
        i.e. logits[:, -32:, :] vs y_val[:, -32:]
    This isolates the effect of context length from the effect of
    which token positions are being scored.
    """
    model.eval()
    x_val, y_val = val_batch
    x_val, y_val = x_val.to(device), y_val.to(device)

    logits, loss, layer_hiddens, att_matrices = model(x_val, y_val, return_diagnostics=True)
    val_loss = loss.item()
    bpc = val_loss / math.log(2.0)

    # Diagnostics
    avg_ranks = [compute_effective_rank(h) for h in layer_hiddens]
    mean_rank = float(np.mean(avg_ranks))
    avg_entropies = [compute_normalized_entropy(att) for att in att_matrices]
    mean_entropy = float(np.mean(avg_entropies))

    # CE-2: Fixed-target context sensitivity
    B_val, T_val = x_val.shape
    context_losses_fixed = {}
    for ctx_len in [32, 64, 128, 256]:
        if ctx_len <= T_val:
            x_sub = x_val[:, -ctx_len:]                  # variable context
            # CE-2 FIX: always score only the last FIXED_TARGET_LEN targets
            tgt_len = min(FIXED_TARGET_LEN, ctx_len)
            y_fixed = y_val[:, -tgt_len:]                 # fixed target window

            sub_logits, _ = model(x_sub, None)            # (B, ctx_len, V)
            # score only the last tgt_len positions of the sub_logits
            scored_logits = sub_logits[:, -tgt_len:, :]   # (B, tgt_len, V)
            B, Ts, C = scored_logits.shape
            loss_fixed = F.cross_entropy(
                scored_logits.reshape(B * Ts, C),
                y_fixed.reshape(B * Ts)
            )
            context_losses_fixed[ctx_len] = loss_fixed.item() / math.log(2.0)

    model.train()
    return {
        "val_loss": val_loss,
        "bpc": bpc,
        "mean_rank": mean_rank,
        "mean_entropy": mean_entropy,
        "context_bpc_fixed": context_losses_fixed,    # CE-2 corrected
    }


# ──────────────────────────────────────────────────────────────────────
# CE-3: Batch Manifest Generation
# ──────────────────────────────────────────────────────────────────────

CONTEXT_LENGTHS = [32, 64, 128, 256]
BATCH_SIZES     = {32: 128, 64: 64, 128: 32, 256: 16}
MAX_ITERS       = 2500
STEPS_PER_PHASE = 625                          # each T gets exactly 625 steps

# Pool size calculation:
# For curriculum, shuffled, anti-curriculum, each T has 625 steps.
# But fixed_long runs ALL 2500 steps at T=256!
# Therefore, pool for T=256 must have 2500 * 16 = 40,000 draws.
POOL_SIZE = {
    32:  STEPS_PER_PHASE * BATCH_SIZES[32],   # 625 * 128 = 80,000
    64:  STEPS_PER_PHASE * BATCH_SIZES[64],   # 625 * 64  = 40,000
    128: STEPS_PER_PHASE * BATCH_SIZES[128],  # 625 * 32  = 20,000
    256: MAX_ITERS       * BATCH_SIZES[256],  # 2500 * 16 = 40,000 (supports Fixed-Long!)
}

def generate_batch_manifest(seed, train_data_len):
    """
    Pre-sample enough start-indices per context length so every step
    draws a unique, non-overlapping slice.
    Uses seed = global_seed * 1000 + T so pools are independent across T.
    """
    manifest = {}
    for T in CONTEXT_LENGTHS:
        pool_size = POOL_SIZE[T]
        rng = np.random.RandomState(seed * 1000 + T)
        manifest[T] = rng.randint(0, train_data_len - T, size=pool_size)
    return manifest


def build_step_configs(schedule_type, seed):
    """Return list of (T, B, step_within_phase) for each of MAX_ITERS steps.
    step_within_phase is used to compute the non-overlapping pool slice:
      pool[step_idx*B : (step_idx+1)*B]
    """
    configs = []
    if schedule_type == "curriculum":
        phases = [32, 64, 128, 256]
    elif schedule_type == "anti_curriculum":
        phases = [256, 128, 64, 32]
    elif schedule_type == "fixed_long":
        phases = [256, 256, 256, 256]
    elif schedule_type == "shuffled":
        ordered = [32]*625 + [64]*625 + [128]*625 + [256]*625
        rng = np.random.RandomState(seed + 100)
        rng.shuffle(ordered)
        counters = {T: 0 for T in CONTEXT_LENGTHS}
        for T in ordered:
            step_idx = counters[T]
            configs.append((T, BATCH_SIZES[T], step_idx))
            counters[T] += 1
        return configs

    counters = {T: 0 for T in CONTEXT_LENGTHS}
    for phase_T in phases:
        for _ in range(STEPS_PER_PHASE):
            step_idx = counters[phase_T]
            configs.append((phase_T, BATCH_SIZES[phase_T], step_idx))
            counters[phase_T] += 1
    return configs


# ──────────────────────────────────────────────────────────────────────
# Single arm runner
# ──────────────────────────────────────────────────────────────────────

def run_arm(arm_name, schedule_type, seed, manifest, val_batch, device):
    set_seed(seed)
    vocab_size = len(ds.chars)
    model = MiniTransformerLM(
        vocab_size=vocab_size, n_embd=256, n_head=8, n_layer=6,
        block_size=256, pos_emb_type="rope"
    ).to(device)
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=1e-3)
    train_data = ds.train_data

    step_configs = build_step_configs(schedule_type, seed)
    logs = []
    model.train()
    t0 = time.time()

    for step in range(1, MAX_ITERS + 1):
        T, B, step_idx = step_configs[step - 1]

        # CE-3 FIX: non-overlapping slice from pre-generated manifest
        # Each step gets unique contiguous slice: pool[step_idx*B : (step_idx+1)*B]
        pool = manifest[T]
        ix = torch.tensor(pool[step_idx * B : (step_idx + 1) * B], dtype=torch.long)
        x = torch.stack([train_data[i:i+T] for i in ix]).to(device)
        y = torch.stack([train_data[i+1:i+T+1] for i in ix]).to(device)


        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 250 == 0 or step == MAX_ITERS:
            diag = evaluate_model_v2(model, val_batch, device)
            grad_norm = torch.norm(torch.stack([
                torch.norm(p.grad.detach()) for p in model.parameters()
                if p.grad is not None
            ])).item()
            logs.append({
                "step": step, "T": T, "B": B, "lr": lr,
                "train_loss": loss.item(),
                "val_loss": diag["val_loss"],
                "val_bpc": diag["bpc"],
                "mean_rank": diag["mean_rank"],
                "mean_entropy": diag["mean_entropy"],
                "grad_norm": grad_norm,
                "context_bpc_fixed": diag["context_bpc_fixed"],
                "elapsed_sec": time.time() - t0,
            })
            print(
                f"  [{arm_name:30s}] step {step:4d} | T={T:3d} | "
                f"val_bpc={diag['bpc']:.4f} | rank={diag['mean_rank']:.1f} | "
                f"attn_ent={diag['mean_entropy']:.4f}",
                flush=True
            )

    return {
        "arm_name": arm_name,
        "schedule_type": schedule_type,
        "seed": seed,
        "final_bpc": logs[-1]["val_bpc"],
        "logs": logs,
    }


# ──────────────────────────────────────────────────────────────────────
# 5-seed paired experiment
# ──────────────────────────────────────────────────────────────────────

ARMS = [
    ("Curriculum (32→256)",       "curriculum"),
    ("Shuffled Control",           "shuffled"),
    ("Anti-Curriculum (256→32)",   "anti_curriculum"),
    ("Fixed-Long Baseline (256)",  "fixed_long"),
]
SEEDS = [42, 43, 44, 45, 46]


def run_all(output_dir="results/ce3_paired", log_path="results/ce3_run_log.txt"):
    os.makedirs(output_dir, exist_ok=True)
    device = 'mps' if torch.backends.mps.is_available() else \
             ('cuda' if torch.cuda.is_available() else 'cpu')

    # Open log file for direct writing (avoids tee pipe buffering)
    log_fh = open(log_path, 'w', buffering=1)  # line-buffered

    def log(msg=""):
        print(msg, flush=True)
        log_fh.write(msg + "\n")
        log_fh.flush()

    log(f"\n{'='*65}")
    log(f"CE-3: Paired Batch Manifest — 5 Seeds × 4 Arms")
    log(f"Device: {device}")
    log(f"{'='*65}\n")

    train_data_len = len(ds.train_data)
    val_data       = ds.val_data
    log(f"train_data_len={train_data_len}, val_data_len={len(val_data)}")

    # Collect final BPC per arm per seed
    bpc_table = {sched: [] for _, sched in ARMS}
    all_results = {}

    for seed in SEEDS:
        log(f"\n{'━'*65}")
        log(f"SEED {seed}")
        log(f"{'━'*65}")

        # Fixed validation batch — same across all arms for this seed
        set_seed(seed * 999)
        val_ix = torch.randint(len(val_data) - 256, (16,))
        val_x  = torch.stack([val_data[i:i+256] for i in val_ix])
        val_y  = torch.stack([val_data[i+1:i+256+1] for i in val_ix])
        val_batch = (val_x, val_y)

        # Generate batch manifest — shared across all arms for this seed
        manifest = generate_batch_manifest(seed, train_data_len)
        log(f"Manifest generated (seed={seed}): "
            f"{[f'T={T}:{len(manifest[T])} indices' for T in CONTEXT_LENGTHS]}")

        seed_path = os.path.join(output_dir, f"seed_{seed}_results.json")
        seed_results = {}
        if os.path.exists(seed_path):
            try:
                with open(seed_path, "r") as f:
                    seed_results = json.load(f)
            except Exception:
                seed_results = {}

        for arm_name, sched_type in ARMS:
            if sched_type in seed_results:
                log(f"\n  ── Arm: {arm_name} (seed={seed}) [CACHED, skipping] ──")
                res = seed_results[sched_type]
                bpc_table[sched_type].append(res["final_bpc"])
                log(f"  ⚡ {arm_name} loaded from cache: final_bpc={res['final_bpc']:.5f}")
                continue

            log(f"\n  ── Arm: {arm_name} (seed={seed}) ──")
            res = run_arm(arm_name, sched_type, seed, manifest, val_batch, device)
            seed_results[sched_type] = res
            bpc_table[sched_type].append(res["final_bpc"])
            log(f"  ✅ {arm_name} done: final_bpc={res['final_bpc']:.5f}")

            # Save immediately after each arm so progress is never lost
            with open(seed_path, "w") as f:
                json.dump(seed_results, f, indent=2)

        log(f"\n  Saved seed-{seed} results → {seed_path}")
        all_results[seed] = seed_results

    # ── Statistical summary ───────────────────────────────────────────
    log_fh.close()
    # Reopen for appending summary
    log_fh = open(log_path, 'a', buffering=1)
    def log(msg=""):
        print(msg, flush=True)
        log_fh.write(msg + "\n")
        log_fh.flush()

    log(f"\n{'='*65}")
    log(f"CE-3 FINAL SUMMARY (across {len(SEEDS)} paired seeds)")
    log(f"{'='*65}")
    log(f"\n{'Arm':<35} {'Mean BPC':>10} {'Std BPC':>10} {'Min':>8} {'Max':>8}")
    log("-" * 75)

    arm_means = {}
    for arm_name, sched in ARMS:
        vals = bpc_table[sched]
        mean_bpc = float(np.mean(vals))
        std_bpc  = float(np.std(vals, ddof=1))
        arm_means[sched] = mean_bpc
        log(f"{arm_name:<35} {mean_bpc:>10.5f} {std_bpc:>10.5f} "
            f"{min(vals):>8.5f} {max(vals):>8.5f}")

    # Paired t-tests vs Curriculum
    log(f"\n── Paired t-tests vs Curriculum ──")
    curr_vals = bpc_table["curriculum"]
    for arm_name, sched in ARMS:
        if sched == "curriculum":
            continue
        other_vals = bpc_table[sched]
        t_stat, p_val = stats.ttest_rel(curr_vals, other_vals)
        diff = arm_means["curriculum"] - arm_means[sched]
        sig = "✅ p<0.05" if p_val < 0.05 else "— n.s."
        log(f"  Curriculum vs {arm_name:<30}: Δ={diff:+.5f} BPC | "
            f"t={t_stat:.3f} | p={p_val:.3f} {sig}")

    # Hypothesis verdict
    log(f"\n── CE-3 HYPOTHESIS VERDICT ──")
    anti_vals = bpc_table["anti_curriculum"]
    anti_t, anti_p = stats.ttest_rel(curr_vals, anti_vals)
    anti_diff = float(np.mean(anti_vals)) - float(np.mean(curr_vals))
    if anti_p < 0.05 and anti_diff > 0.05:
        log(f"  Anti-Curriculum: degradation CONFIRMED (Δ={anti_diff:+.4f} BPC, p={anti_p:.4f})")
    else:
        log(f"  Anti-Curriculum: degradation NOT confirmed at p<0.05 (Δ={anti_diff:+.4f}, p={anti_p:.4f})")

    shuf_vals = bpc_table["shuffled"]
    shuf_t, shuf_p = stats.ttest_rel(curr_vals, shuf_vals)
    shuf_diff = float(np.mean(curr_vals)) - float(np.mean(shuf_vals))
    if abs(shuf_diff) < 0.01 or shuf_p > 0.05:
        log(f"  Curriculum vs Shuffled: EQUIVALENT within ±0.01 BPC (Δ={shuf_diff:+.5f}, p={shuf_p:.4f})")
    else:
        log(f"  Curriculum vs Shuffled: DISTINGUISHABLE (Δ={shuf_diff:+.5f}, p={shuf_p:.4f})")

    # Save summary
    summary = {
        "seeds": SEEDS,
        "bpc_table": {k: list(map(float, v)) for k, v in bpc_table.items()},
        "means": {k: float(np.mean(v)) for k, v in bpc_table.items()},
        "stds":  {k: float(np.std(v, ddof=1)) for k, v in bpc_table.items()},
    }
    summary_path = os.path.join(output_dir, "ce3_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nSummary saved → {summary_path}")
    log_fh.close()
    return summary


if __name__ == "__main__":
    run_all()
