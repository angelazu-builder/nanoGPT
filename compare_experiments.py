#!/opt/miniconda3/bin/python3
import os
import json
import math
import matplotlib.pyplot as plt

os.makedirs("Day 2", exist_ok=True)
os.makedirs("Day 3", exist_ok=True)

def load_exp(exp_path):
    config_path = os.path.join(exp_path, "config.json")
    loss_path = os.path.join(exp_path, "loss_history.json")
    if not os.path.exists(loss_path):
        return None
    with open(loss_path, "r") as f:
        data = json.load(f)
    cfg = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as cf:
            cfg = json.load(cf)
    
    # Filter out incomplete draft runs (less than 1000 steps)
    steps = data.get("steps", [])
    if not steps or max(steps) < 1000:
        return None
        
    return {"path": exp_path, "data": data, "config": cfg}

# -------------------------------------------------------------
# Chart 1: Day 2 Clean Comparison - Context Expansion (exp05 vs exp06)
# -------------------------------------------------------------
exp05 = load_exp("Day 2/runs/exp05") # Char, block=64
exp06 = load_exp("Day 2/runs/exp06") # Char, block=256

if exp05 and exp06:
    plt.figure(figsize=(9, 5))
    plt.plot(exp05["data"]["steps"], exp05["data"]["val_loss"], label="exp05: Char Baseline (block_size=64)", color="#1f77b4", linewidth=2.5)
    plt.plot(exp06["data"]["steps"], exp06["data"]["val_loss"], label="exp06: Char Context Expansion (block_size=256)", color="#2ca02c", linewidth=2.5)

    plt.axhline(y=1.47, color="green", linestyle="--", linewidth=1.5, label="Karpathy Target Baseline (1.47)")
    plt.xlabel("Training Steps", fontsize=11)
    plt.ylabel("Validation Loss (Nats)", fontsize=11)
    plt.title("Day 2: Character-Level Context Window Expansion (block=64 vs block=256)", fontsize=13, fontweight="bold")
    plt.legend(fontsize=10, loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("Day 2/day2_char_scaling_benchmark.png", dpi=300)
    plt.close()

# -------------------------------------------------------------
# Chart 2: Day 3 Clean Comparison - Subword BPE vs Char Champions (exp05, exp06 vs exp07)
# -------------------------------------------------------------
exp07 = load_exp("Day 3/runs/exp07") # BPE, block=128

if exp05 and exp06 and exp07:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Raw CrossEntropy Loss (Only Val Loss solid lines)
    ax1.plot(exp05["data"]["steps"], exp05["data"]["val_loss"], label="exp05: Char (block=64, V=65)", color="#1f77b4", linewidth=2.2)
    ax1.plot(exp06["data"]["steps"], exp06["data"]["val_loss"], label="exp06: Char (block=256, V=65)", color="#2ca02c", linewidth=2.2)
    ax1.plot(exp07["data"]["steps"], exp07["data"]["val_loss"], label="exp07: Subword BPE (block=128, V=50k)", color="#ff7f0e", linewidth=2.5)

    ax1.axhline(y=1.47, color="green", linestyle="--", linewidth=1.5, label="Karpathy Reference (~1.47)")
    ax1.set_xlabel("Training Steps", fontsize=11)
    ax1.set_ylabel("Validation Loss (Nats)", fontsize=11)
    ax1.set_title("(A) Raw CrossEntropy Loss", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Panel B: Fair BPC Normalized Loss
    # Char BPC = Loss / ln(2)
    # BPE BPC = Loss / (ln(2) * 3.30)
    bpc_05 = [v / math.log(2) for v in exp05["data"]["val_loss"]]
    bpc_06 = [v / math.log(2) for v in exp06["data"]["val_loss"]]
    bpc_07 = [v / (math.log(2) * 3.30) for v in exp07["data"]["val_loss"]]

    ax2.plot(exp05["data"]["steps"], bpc_05, label="exp05: Char (2.15 BPC)", color="#1f77b4", linewidth=2.2)
    ax2.plot(exp06["data"]["steps"], bpc_06, label="exp06: Char Expanded (2.12 BPC)", color="#2ca02c", linewidth=2.2)
    ax2.plot(exp07["data"]["steps"], bpc_07, label="exp07: BPE Subword (2.12 BPC)", color="#ff7f0e", linewidth=2.5)

    ax2.axhline(y=1.47/math.log(2), color="green", linestyle="--", linewidth=1.5, label="Karpathy Reference (2.12 BPC)")
    ax2.set_xlabel("Training Steps", fontsize=11)
    ax2.set_ylabel("Normalized Loss (Bits-Per-Character / BPC)", fontsize=11)
    ax2.set_title("(B) Fair BPC Information Density", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Clean Multi-Experiment Comparison (Draft Runs Excluded)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/bpe_vs_char_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig("Day 3/bpe_vs_char_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig("Day 2/bpe_vs_char_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig("runs/multi_exp_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

print("✨ Successfully cleaned up figures! Excluded incomplete draft runs (exp01, exp02, exp03, exp04).")
print("  - Remaining Complete Runs: exp05 (Char 64), exp06 (Char 256), exp07 (BPE 128)")
