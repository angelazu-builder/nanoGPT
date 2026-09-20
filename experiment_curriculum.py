import os
import json
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import dataset as ds
from model import MiniTransformerLM

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def compute_effective_rank(tensor_2d):
    """ Computes Effective Rank = exp(-sum p_k log p_k) where p_k = sigma_k / sum(sigma) """
    if tensor_2d.ndim > 2:
        tensor_2d = tensor_2d.reshape(-1, tensor_2d.size(-1))
    # Double precision SVD on CPU for MPS compatibility and stability
    tensor_cpu = tensor_2d.detach().cpu().float()
    U, S, V = torch.linalg.svd(tensor_cpu, full_matrices=False)
    S_sum = torch.sum(S)
    if S_sum < 1e-12:
        return 1.0
    p = S / S_sum
    p = p[p > 1e-12]
    entropy = -torch.sum(p * torch.log(p)).item()
    return math.exp(entropy)

def compute_normalized_entropy(att_matrix):
    """
    Computes normalized causal attention entropy H_bar = H(A) / H_max(T).
    att_matrix shape: (B, nh, T, T)
    """
    B, nh, T, _ = att_matrix.shape
    mask = torch.tril(torch.ones((T, T), device=att_matrix.device)).bool()
    
    # Calculate H_max(T)
    seq_lens = torch.arange(1, T + 1, device=att_matrix.device).float()
    h_max = torch.mean(torch.log(seq_lens)).item()
    if h_max < 1e-6:
        return 1.0

    p = att_matrix + 1e-12
    log_p = torch.log(p)
    ent_per_token = -torch.sum(att_matrix * log_p, dim=-1) # (B, nh, T)
    avg_ent = torch.mean(ent_per_token).item()
    return min(1.0, max(0.0, avg_ent / h_max))

@torch.no_grad()
def evaluate_model(model, val_batch, device):
    """ Evaluates loss, BPC, attention entropy, and effective rank on fixed validation batch """
    model.eval()
    x_val, y_val = val_batch
    x_val, y_val = x_val.to(device), y_val.to(device)
    
    logits, loss, layer_hiddens, att_matrices = model(x_val, y_val, return_diagnostics=True)
    val_loss = loss.item()
    bpc = val_loss / math.log(2.0)

    # Compute diagnostic metrics
    avg_ranks = []
    for h in layer_hiddens:
        avg_ranks.append(compute_effective_rank(h))
    mean_rank = float(np.mean(avg_ranks))

    avg_entropies = []
    for att in att_matrices:
        avg_entropies.append(compute_normalized_entropy(att))
    mean_entropy = float(np.mean(avg_entropies))

    # Context Sensitivity Probe (Yao et al. 2024): Evaluate loss under truncated contexts
    B_val, T_val = x_val.shape
    context_losses = {}
    for ctx_len in [32, 64, 128, 256]:
        if ctx_len <= T_val:
            x_sub = x_val[:, -ctx_len:]
            y_sub = y_val[:, -ctx_len:]
            sub_logits, sub_loss = model(x_sub, y_sub)
            context_losses[ctx_len] = sub_loss.item() / math.log(2.0)

    model.train()
    return {
        "val_loss": val_loss,
        "bpc": bpc,
        "mean_rank": mean_rank,
        "mean_entropy": mean_entropy,
        "context_bpc": context_losses
    }

def get_lr(it, max_iters=2500, warmup_iters=400, learning_rate=1e-3, min_lr=1e-4):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def run_experimental_arm(arm_name, schedule_type, seed=42, max_iters=2500):
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=======================================================")
    print(f"🚀 Launching Experimental Arm: '{arm_name}' (Seed: {seed})")
    print(f"=======================================================")

    set_seed(seed)
    vocab_size = len(ds.chars)
    model = MiniTransformerLM(
        vocab_size=vocab_size,
        n_embd=256,
        n_head=8,
        n_layer=6,
        block_size=256,
        pos_emb_type="rope"
    ).to(device)

    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=1e-3)
    train_data, val_data = ds.train_data, ds.val_data

    # Generate fixed validation batch (B=16, T=256)
    set_seed(999)
    val_ix = torch.randint(len(val_data) - 256, (16,))
    val_x = torch.stack([val_data[i:i+256] for i in val_ix])
    val_y = torch.stack([val_data[i+1:i+256+1] for i in val_ix])
    val_batch = (val_x, val_y)

    set_seed(seed)

    # Build sequence schedule for 2500 steps
    step_configs = []
    if schedule_type == "curriculum":
        # 32 -> 64 -> 128 -> 256
        for step in range(max_iters):
            if step < 625:
                step_configs.append((32, 128))
            elif step < 1250:
                step_configs.append((64, 64))
            elif step < 1875:
                step_configs.append((128, 32))
            else:
                step_configs.append((256, 16))
    elif schedule_type == "anti_curriculum":
        # 256 -> 128 -> 64 -> 32
        for step in range(max_iters):
            if step < 625:
                step_configs.append((256, 16))
            elif step < 1250:
                step_configs.append((128, 32))
            elif step < 1875:
                step_configs.append((64, 64))
            else:
                step_configs.append((32, 128))
    elif schedule_type == "shuffled":
        # Exact same 625 count of each, randomly shuffled
        blocks = [(32, 128)] * 625 + [(64, 64)] * 625 + [(128, 32)] * 625 + [(256, 16)] * 625
        rng = np.random.RandomState(seed + 100)
        rng.shuffle(blocks)
        step_configs = blocks
    elif schedule_type == "fixed_long":
        # Static 256 throughout
        step_configs = [(256, 16)] * max_iters

    logs = []
    start_time = time.time()
    model.train()

    for step in range(1, max_iters + 1):
        T, B = step_configs[step - 1]
        
        # Sample batch
        ix = torch.randint(len(train_data) - T, (B,))
        x = torch.stack([train_data[i:i+T] for i in ix]).to(device)
        y = torch.stack([train_data[i+1:i+T+1] for i in ix]).to(device)

        # Update learning rate
        lr = get_lr(step, max_iters=max_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Evaluate diagnostics every 250 steps or step 1
        if step == 1 or step % 250 == 0 or step == max_iters:
            diag = evaluate_model(model, val_batch, device)
            grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in model.parameters() if p.grad is not None])).item()
            
            log_entry = {
                "step": step,
                "context_length": T,
                "batch_size": B,
                "lr": lr,
                "train_loss": loss.item(),
                "val_loss": diag["val_loss"],
                "val_bpc": diag["bpc"],
                "mean_rank": diag["mean_rank"],
                "mean_entropy": diag["mean_entropy"],
                "grad_norm": grad_norm,
                "context_bpc": diag["context_bpc"],
                "elapsed_sec": time.time() - start_time
            }
            logs.append(log_entry)
            print(f"Step {step:4d}/{max_iters} | T={T:3d} | Train Loss: {loss.item():.4f} | Val BPC: {diag['bpc']:.4f} | Rank: {diag['mean_rank']:.2f} | Attn Ent: {diag['mean_entropy']:.4f}")

    return {
        "arm_name": arm_name,
        "schedule_type": schedule_type,
        "seed": seed,
        "final_bpc": logs[-1]["val_bpc"],
        "logs": logs
    }

def run_all_experiments(seed=42):
    os.makedirs("results", exist_ok=True)
    arms = [
        ("Curriculum (32->256)", "curriculum"),
        ("Shuffled Control", "shuffled"),
        ("Anti-Curriculum (256->32)", "anti_curriculum"),
        ("Fixed-Long Baseline (256)", "fixed_long")
    ]
    
    all_results = {}
    for arm_name, sched_type in arms:
        res = run_experimental_arm(arm_name, sched_type, seed=seed)
        all_results[sched_type] = res

    output_file = "results/curriculum_experiment_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ All 4 experimental arms completed! Results saved to '{output_file}'.")
    return all_results

if __name__ == "__main__":
    run_all_experiments(seed=42)
