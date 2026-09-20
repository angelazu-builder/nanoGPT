import time
import torch
import torch.nn as nn
import numpy as np
import dataset as ds
from model import MiniTransformerLM

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def run_gradient_probe(num_trials=30):
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚡ Running Initialization Gradient Probe on device: {device}")
    
    set_seed(42)
    vocab_size = len(ds.chars)
    model = MiniTransformerLM(
        vocab_size=vocab_size,
        n_embd=256,
        n_head=8,
        n_layer=6,
        block_size=256,
        pos_emb_type="rope"
    ).to(device)
    model.eval() # Disable dropout during gradient probing for deterministic measurement

    context_configs = [
        {"T": 32, "B": 128},
        {"T": 64, "B": 64},
        {"T": 128, "B": 32},
        {"T": 256, "B": 16},
    ]

    # Pre-sample data batches to ensure fair comparison
    train_data = ds.train_data
    results = {}

    for cfg in context_configs:
        T, B = cfg["T"], cfg["B"]
        tokens_per_step = B * T
        print(f"\n🔬 Probing Context T={T:3d} | Batch B={B:3d} (Total Tokens/Step = {tokens_per_step})")

        grad_vectors = []
        losses = []
        step_times = []

        set_seed(42 + T) # Independent deterministic sampling for each context size

        for t in range(num_trials):
            # Sample batch from dataset
            ix = torch.randint(len(train_data) - T, (B,))
            x = torch.stack([train_data[i:i+T] for i in ix]).to(device)
            y = torch.stack([train_data[i+1:i+T+1] for i in ix]).to(device)

            model.zero_grad(set_to_none=True)
            t0 = time.time()
            logits, loss = model(x, y)
            loss.backward()
            t1 = time.time()
            step_times.append((t1 - t0) * 1000.0)
            losses.append(loss.item())

            # Flatten all gradients into a single 1D vector
            grads = []
            for p in model.parameters():
                if p.grad is not None:
                    grads.append(p.grad.detach().view(-1))
            grad_vec = torch.cat(grads)
            grad_vectors.append(grad_vec.cpu())

        grad_matrix = torch.stack(grad_vectors) # (num_trials, P)
        mean_grad = torch.mean(grad_matrix, dim=0) # (P,)
        
        norm_mean_grad = torch.norm(mean_grad).item()
        
        # Calculate per-parameter variance
        var_grad = torch.var(grad_matrix, dim=0, unbiased=True) # (P,)
        tr_sigma = torch.sum(var_grad).item() # Total gradient variance Tr(Sigma)

        # Critical Noise Scale B_crit = Tr(Sigma) / ||g_bar||^2
        b_crit = tr_sigma / (norm_mean_grad ** 2 + 1e-12)

        # Average Pairwise Cosine Similarity
        normalized_grads = grad_matrix / (torch.norm(grad_matrix, dim=1, keepdim=True) + 1e-12)
        sim_matrix = normalized_grads @ normalized_grads.T
        # Mask out diagonal
        mask = ~torch.eye(num_trials, dtype=torch.bool)
        avg_cos_sim = sim_matrix[mask].mean().item()

        loss_mean = float(np.mean(losses))
        loss_var = float(np.var(losses))
        avg_step_time = float(np.mean(step_times))

        print(f"  - Mean Loss: {loss_mean:.4f} (Var: {loss_var:.6f})")
        print(f"  - Mean Grad Norm ||g_bar||: {norm_mean_grad:.4f}")
        print(f"  - Total Grad Variance Tr(Σ): {tr_sigma:.4f}")
        print(f"  - Gradient Noise Scale B_crit: {b_crit:.2f}")
        print(f"  - Avg Pairwise Cosine Sim: {avg_cos_sim:.4f}")
        print(f"  - Step Time: {avg_step_time:.2f} ms")

        results[T] = {
            "T": T, "B": B,
            "loss_mean": loss_mean, "loss_var": loss_var,
            "norm_mean_grad": norm_mean_grad, "tr_sigma": tr_sigma,
            "b_crit": b_crit, "avg_cos_sim": avg_cos_sim,
            "avg_step_time": avg_step_time
        }

    return results

if __name__ == "__main__":
    run_gradient_probe()
