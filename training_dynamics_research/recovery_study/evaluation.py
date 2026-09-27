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
    N_PROCESS_SEQUENCES,
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
def evaluate_panel(model, x_val, y_val, device, horizons=CONTEXT_LENGTHS, val_start_indices=None):
    """
    Evaluate validation panel with ZERO duplicate forwards:
    1. Forward full context (256) once with diagnostics.
    2. Reuse 256-step logits for horizon h=256 (no second forward!).
    3. Forward sub-contexts only for h in [32, 64, 128].
    4. Store BOTH mean_bpc and per-sequence seq_bpc for ALL horizons.
    5. Extract token-level losses for h=32 and h=256 with structured target records.
    """
    model.eval()
    N_seq, T_seq = x_val.shape
    assert T_seq == VAL_CONTEXT, f"Expected validation context {VAL_CONTEXT}, got {T_seq}"
    
    target_len = FIXED_TARGET_LEN
    y_targets = y_val[:, -target_len:]  # Shape: (N_seq, 32)
    
    # 1. Forward full context (256) once for both diagnostics and h=256 scoring
    full_logits, _, layer_hiddens, att_matrices = model(x_val, y_val, return_diagnostics=True)
    
    avg_ranks = [compute_effective_rank(h) for h in layer_hiddens]
    mean_rank = float(np.mean(avg_ranks))
    avg_entropies = [compute_normalized_entropy(att) for att in att_matrices]
    mean_entropy = float(np.mean(avg_entropies))
    
    horizon_results = {}
    token_losses = {}
    
    for h in horizons:
        if h == 256:
            # Reuse full forward logits directly!
            scored_logits = full_logits[:, -target_len:, :]
        else:
            x_sub = x_val[:, -h:]
            sub_logits, _ = model(x_sub, None)
            scored_logits = sub_logits[:, -target_len:, :]
            
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
        
        # Cache token-level losses for sensitivity probe (T=32 and T=256)
        if h in [32, 256]:
            token_losses[h] = (loss_matrix / math.log(2.0)).cpu().numpy().flatten()
            
    # Compute context sensitivity statistics if both 32 and 256 were scored
    sensitivity = None
    if 32 in token_losses and 256 in token_losses:
        c_i = token_losses[32] - token_losses[256]
        
        target_records = []
        if val_start_indices is not None:
            for sequence_index, absolute_start in enumerate(val_start_indices):
                for target_offset in range(target_len):
                    flat_idx = sequence_index * target_len + target_offset
                    abs_pos = int(absolute_start + VAL_CONTEXT - target_len + 1 + target_offset)
                    bpc_32 = float(token_losses[32][flat_idx])
                    bpc_256 = float(token_losses[256][flat_idx])
                    target_records.append({
                        "validation_sequence_index": sequence_index,
                        "target_offset_within_scored_window": target_offset,
                        "absolute_validation_position": abs_pos,
                        "bpc_32": bpc_32,
                        "bpc_256": bpc_256,
                        "context_gain": bpc_32 - bpc_256,
                    })
                    
        sensitivity = {
            "c_i": c_i.tolist(),
            "mean_c": float(np.mean(c_i)),
            "median_c": float(np.median(c_i)),
            "frac_gt_0_1": float(np.mean(c_i > 0.1)),
            "token_bpc_32": token_losses[32].tolist(),
            "token_bpc_256": token_losses[256].tolist(),
            "target_records": target_records,
        }
        
    model.train()
    
    return {
        "bpc": horizon_results[256]["mean_bpc"],
        "seq_bpc": horizon_results[256]["seq_bpc"],
        "horizon_results": horizon_results,
        "mean_rank": mean_rank,
        "mean_entropy": mean_entropy,
        "sensitivity": sensitivity,
    }


def slice_process_metrics_from_anchor(anchor_eval_result, n_process=N_PROCESS_SEQUENCES):
    """
    Given evaluation results on the full 32-sequence anchor panel,
    derive the exact 16-sequence process panel metrics without re-running forward passes!
    Because the process panel is defined as the first 16 sequences of the anchor panel.
    """
    proc_horizon_results = {}
    for h, res in anchor_eval_result["horizon_results"].items():
        proc_seq = res["seq_bpc"][:n_process]
        proc_horizon_results[h] = {
            "mean_bpc": float(np.mean(proc_seq)),
            "seq_bpc": proc_seq,
        }
        
    return {
        "bpc": proc_horizon_results[256]["mean_bpc"],
        "seq_bpc": proc_horizon_results[256]["seq_bpc"],
        "horizon_results": proc_horizon_results,
        "mean_rank": anchor_eval_result["mean_rank"],
        "mean_entropy": anchor_eval_result["mean_entropy"],
    }
