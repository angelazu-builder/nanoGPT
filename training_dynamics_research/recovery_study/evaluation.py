"""
Evaluation suite for Context-Length Recovery Study
Preregistered specification: Section 10, 11, 12, 13
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from .config import (
    FIXED_TARGET_LEN,
    CONTEXT_LENGTHS,
    VAL_CONTEXT,
)


def compute_effective_rank(tensor_2d):
    """Compute effective rank from singular value distribution."""
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
    """Compute normalized attention entropy across heads and tokens."""
    B, nh, T, _ = att_matrix.shape
    seq_lens = torch.arange(1, T + 1, device=att_matrix.device).float()
    h_max = torch.mean(torch.log(seq_lens)).item()
    if h_max < 1e-6:
        return 1.0
    p = att_matrix + 1e-12
    ent_per_token = -torch.sum(att_matrix * torch.log(p), dim=-1)
    avg_ent = torch.mean(ent_per_token).item()
    return min(1.0, max(0.0, avg_ent / h_max))


def build_validation_tensors(val_data, start_indices, device):
    """
    Given validation data and start indices, construct (x_val, y_val) tensors.
    Shape: (N, VAL_CONTEXT) where VAL_CONTEXT = 256.
    """
    x_list = [val_data[i : i + VAL_CONTEXT] for i in start_indices]
    y_list = [val_data[i + 1 : i + VAL_CONTEXT + 1] for i in start_indices]
    x_val = torch.stack(x_list).to(device)
    y_val = torch.stack(y_list).to(device)
    return x_val, y_val


@torch.no_grad()
def evaluate_panel(model, x_val, y_val, device, horizons=CONTEXT_LENGTHS):
    """
    Evaluate validation panel (either 16 process sequences or 32 anchor sequences).
    Scores ONLY the final FIXED_TARGET_LEN = 32 target positions for all horizons.
    Returns per-sequence losses and summary metrics.
    """
    model.eval()
    N_seq, T_seq = x_val.shape
    assert T_seq == VAL_CONTEXT, f"Expected validation context {VAL_CONTEXT}, got {T_seq}"
    
    # 1. Forward full context (256) for general diagnostics
    logits, loss, layer_hiddens, att_matrices = model(x_val, y_val, return_diagnostics=True)
    
    avg_ranks = [compute_effective_rank(h) for h in layer_hiddens]
    mean_rank = float(np.mean(avg_ranks))
    avg_entropies = [compute_normalized_entropy(att) for att in att_matrices]
    mean_entropy = float(np.mean(avg_entropies))
    
    # 2. Score each horizon h in [32, 64, 128, 256] on the final FIXED_TARGET_LEN targets
    horizon_results = {}
    target_len = FIXED_TARGET_LEN
    y_targets = y_val[:, -target_len:]  # Shape: (N_seq, 32)
    
    for h in horizons:
        x_sub = x_val[:, -h:]  # Sub-sequence of length h
        sub_logits, _ = model(x_sub, None)  # Shape: (N_seq, h, V)
        scored_logits = sub_logits[:, -target_len:, :]  # Shape: (N_seq, 32, V)
        
        # Compute loss per sequence (mean over the 32 targets for each sequence)
        N_s, Ts, V = scored_logits.shape
        loss_matrix = F.cross_entropy(
            scored_logits.reshape(N_s * Ts, V),
            y_targets.reshape(N_s * Ts),
            reduction='none'
        ).reshape(N_s, Ts)  # (N_seq, 32) in nats
        
        # Per-sequence BPC: mean over the 32 positions, divided by ln(2)
        seq_bpc = (loss_matrix.mean(dim=1) / math.log(2.0)).cpu().numpy().tolist()
        mean_bpc = float(np.mean(seq_bpc))
        
        horizon_results[h] = {
            "mean_bpc": mean_bpc,
            "seq_bpc": seq_bpc,
        }
        
    model.train()
    
    return {
        "bpc": horizon_results[256]["mean_bpc"],  # Primary target horizon T=256
        "seq_bpc": horizon_results[256]["seq_bpc"],
        "horizon_results": horizon_results,
        "mean_rank": mean_rank,
        "mean_entropy": mean_entropy,
    }


@torch.no_grad()
def evaluate_context_sensitivity(model, x_val, y_val, device):
    """
    Section 13: Same-target context intervention at global steps 2000 & 2500.
    For each target position i in the final 32 tokens across all sequences:
      ell_i(T) = -log2 P(x_i | c_T)
      C_i = ell_i(32) - ell_i(256)
    Returns:
      token_losses: dict mapping T -> flat array of target token BPCs
      C_i: array of bit differences
      mean_C, median_C, frac_gt_0_1
    """
    model.eval()
    target_len = FIXED_TARGET_LEN
    y_targets = y_val[:, -target_len:]  # (N_seq, 32)
    
    token_losses = {}
    for h in [32, 256]:
        x_sub = x_val[:, -h:]
        sub_logits, _ = model(x_sub, None)
        scored_logits = sub_logits[:, -target_len:, :]
        N_s, Ts, V = scored_logits.shape
        loss_matrix = F.cross_entropy(
            scored_logits.reshape(N_s * Ts, V),
            y_targets.reshape(N_s * Ts),
            reduction='none'
        ).reshape(N_s, Ts)
        token_bpc = (loss_matrix / math.log(2.0)).cpu().numpy().flatten()
        token_losses[h] = token_bpc
        
    c_i = token_losses[32] - token_losses[256]
    mean_c = float(np.mean(c_i))
    median_c = float(np.median(c_i))
    frac_gt_0_1 = float(np.mean(c_i > 0.1))
    
    model.train()
    
    return {
        "c_i": c_i.tolist(),
        "mean_c": mean_c,
        "median_c": median_c,
        "frac_gt_0_1": frac_gt_0_1,
        "token_bpc_32": token_losses[32].tolist(),
        "token_bpc_256": token_losses[256].tolist(),
    }
