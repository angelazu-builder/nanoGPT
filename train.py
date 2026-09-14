#!/opt/miniconda3/bin/python3
import os
import sys
import json
import math
import shutil
import torch
import matplotlib.pyplot as plt
from config import DEFAULT_CONFIG, get_dataset_module
from model import MiniTransformerLM
from eval_gibberish import evaluate_gibberish_rate

# ==============================================================================
# 1. Experiment Management & Directory Creation (Day 3 Folder Structure)
# ==============================================================================
day_dir = "Day 3"
day_runs_dir = os.path.join(day_dir, "runs")
os.makedirs(day_runs_dir, exist_ok=True)

# Check existing experiments across runs/, Day 3/runs/, Day 2/runs/, Day 1/runs/
existing_exps = []
for search_dir in ["runs", day_runs_dir, "Day 2/runs", "Day 1/runs"]:
    if os.path.exists(search_dir):
        for d in os.listdir(search_dir):
            if d.startswith("exp") and os.path.isdir(os.path.join(search_dir, d)):
                existing_exps.append(d)

exp_nums = [int(d.replace("exp", "")) for d in existing_exps if d.replace("exp", "").isdigit()]
next_num = (max(exp_nums) + 1) if exp_nums else 1
exp_id = f"exp{next_num:02d}"
exp_dir = os.path.join(day_runs_dir, exp_id)
os.makedirs(exp_dir, exist_ok=True)

# Also create root runs/ fallback for compatibility
root_exp_dir = os.path.join("runs", exp_id)
os.makedirs(root_exp_dir, exist_ok=True)

print(f"📁 Created Day 3 Experiment Directory: '{exp_dir}'", flush=True)

# ==============================================================================
# 2. Hyperparameters & Configuration
# ==============================================================================
config = DEFAULT_CONFIG.copy()
config["exp_id"] = exp_id

# Dynamic Dataset Import based on tokenizer setting ('char' vs 'bpe')
tokenizer_type = config.get("tokenizer", "char")
ds = get_dataset_module(tokenizer_type)
vocab_size = ds.vocab_size
decode = ds.decode
encode = ds.encode

# Handle batch fetching helper
if hasattr(ds, 'get_bpe_batch'):
    get_batch = ds.get_bpe_batch
else:
    get_batch = ds.get_batch

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
config["device"] = str(device)
config["vocab_size"] = vocab_size

with open(os.path.join(exp_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=4)

print(f"🚀 Starting {exp_id} on Device: {device} (Apple Silicon M3)", flush=True)
print(f"🔤 Tokenizer: '{tokenizer_type.upper()}' | Vocab Size V = {vocab_size:,}", flush=True)
print(f"⚙️ Features: FlashAttention + Cosine LR Decay + Decoupled Weight Decay ({config['weight_decay']})\n", flush=True)

# ==============================================================================
# 3. Model & Optimizer Setup with Weight Decay Decoupling
# ==============================================================================
model = MiniTransformerLM(
    vocab_size=vocab_size, 
    n_embd=config['n_embd'], 
    n_head=config['n_head'], 
    n_layer=config['n_layer'], 
    block_size=config['block_size'],
    dropout=config['dropout'],
    pos_emb_type=config.get('pos_emb_type', 'absolute')
).to(device)

n_params = sum(p.numel() for p in model.parameters())
config["n_params"] = n_params
print(f"Model parameters: {n_params:,} (~{n_params/1e6:.2f}M)\n", flush=True)

# Configure AdamW via model.configure_optimizers (2D weight decay vs 1D bias/norm exclusion)
optimizer = model.configure_optimizers(weight_decay=config['weight_decay'], learning_rate=config['learning_rate'])

# ==============================================================================
# 4. Learning Rate Schedule: Warmup + Cosine Decay
# ==============================================================================
def get_lr(it):
    if it < config['warmup_iters']:
        return config['learning_rate'] * (it + 1) / config['warmup_iters']
    if it > config['max_iters']:
        return config['min_lr']
    decay_ratio = (it - config['warmup_iters']) / (config['max_iters'] - config['warmup_iters'])
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config['min_lr'] + coeff * (config['learning_rate'] - config['min_lr'])

# ==============================================================================
# 5. Tracking, Reporting & Early Stopping
# ==============================================================================
history = {
    "steps": [],
    "train_loss": [],
    "val_loss": [],
    "lr": []
}

records = [] # List of dicts for MD report generation
best_val_loss = float('inf')
patience_counter = 0
best_model_path = os.path.join(exp_dir, "best_model.pt")
report_md_path = os.path.join(day_dir, f"{exp_id}_training_report.md")

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config['eval_iters'])
        for k in range(config['eval_iters']):
            X, Y = get_batch(split, batch_size=config['batch_size'], block_size=config['block_size'], device=device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def save_loss_plot(history, exp_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(history["steps"], history["train_loss"], label="Train Loss", color="blue", linewidth=2)
    plt.plot(history["steps"], history["val_loss"], label="Val Loss", color="orange", linewidth=2)
    plt.axhline(y=config["karpathy_val_target"], color="green", linestyle="--", label=f"Karpathy Target ({config['karpathy_val_target']})")
    
    plt.xlabel("Training Steps")
    plt.ylabel("CrossEntropy Loss")
    plt.title(f"Experiment {exp_id} - Loss Curve & Cosine LR Decay")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    
    plot_path = os.path.join(exp_dir, "loss_curve.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

def log_sample(step, exp_dir):
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    with torch.no_grad():
        top_k_val = config.get('top_k', 40)
        generated_tokens = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=top_k_val)[0].tolist()
    sample_text = decode(generated_tokens)
    
    gib_rate, total_words, invalid_words = evaluate_gibberish_rate(sample_text)

    # Save to samples.txt file
    with open(os.path.join(exp_dir, "samples.txt"), "a", encoding="utf-8") as f:
        f.write(f"\n--- STEP {step} SAMPLE (Gibberish Rate: {gib_rate:.2f}%) ---\n")
        f.write(sample_text + "\n")

    # Real-time stdout print to terminal
    print("\n" + "="*50, flush=True)
    print(f"📖 Real-time Sample Output at Step {step} (Gibberish Rate: {gib_rate:.1f}%):", flush=True)
    print("-"*50, flush=True)
    print(sample_text, flush=True)
    print("="*50 + "\n", flush=True)

    model.train()
    return sample_text, gib_rate

def update_md_report(records, final_summary="Training in progress..."):
    """ Generates a markdown report file in Day 2 folder """
    md_content = f"# 🚀 Day 2 NanoGPT Training & Evaluation Report ({exp_id})\n\n"
    md_content += "## ⚙️ Hyperparameters & Model Configuration\n\n"
    md_content += "| Configuration Key | Value | Description |\n"
    md_content += "| :--- | :--- | :--- |\n"
    md_content += f"| **Experiment ID** | `{exp_id}` | Isolated experiment directory |\n"
    md_content += f"| **Tokenizer Type** | `{tokenizer_type.upper()}` | Subword BPE vs Character-level |\n"
    md_content += f"| **Device** | `{device}` | Hardware accelerator |\n"
    md_content += f"| **Vocabulary Size ($V$)** | `{vocab_size:,}` | Unique vocabulary tokens |\n"
    md_content += f"| **Embedding Dim ($d_{{model}}$)** | `{config['n_embd']}` | Model hidden feature dimension |\n"
    md_content += f"| **Attention Heads ($h$)** | `{config['n_head']}` | Head size = {config['n_embd'] // config['n_head']} |\n"
    md_content += f"| **Transformer Layers ($l$)** | `{config['n_layer']}` | Total stacked Transformer blocks |\n"
    md_content += f"| **Context Length ($T$)** | `{config['block_size']}` | Sequence history window |\n"
    md_content += f"| **Batch Size ($B$)** | `{config['batch_size']}` | Sequences per batch |\n"
    md_content += f"| **Peak Learning Rate** | `{config['learning_rate']}` | Cosine decay schedule peak |\n"
    md_content += f"| **Weight Decay** | `{config['weight_decay']}` | 2D params decayed, 1D excluded |\n"
    md_content += f"| **Attention Kernel** | `FlashAttention (SDPA)` | PyTorch 2.x kernel acceleration |\n"
    md_content += f"| **Position Embedding** | `{config.get('pos_emb_type', 'absolute')}` | Token position encoding |\n"
    md_content += f"| **Sampling Top-K** | `{config.get('top_k', 40)}` | Truncation sampling parameter |\n"
    md_content += f"| **Total Parameters** | `{n_params:,}` | ~{n_params/1e6:.2f}M parameters |\n\n"
    md_content += "---\n\n"
    
    md_content += "## 📊 Step-by-Step Training Log\n\n"
    md_content += "| Step | Learning Rate | Train Loss | Val Loss | Perplexity (PPL) | Gibberish Rate (%) |\n"
    md_content += "| :---: | :---: | :---: | :---: | :---: | :---: |\n"
    for r in records:
        ppl = math.exp(r['val_loss'])
        gib = r.get('gib_rate', 0.0)
        md_content += f"| {r['step']} | {r['lr']:.2e} | {r['train_loss']:.4f} | {r['val_loss']:.4f} | **{ppl:.2f}** | **{gib:.1f}%** |\n"
    
    md_content += "\n---\n\n"
    md_content += "## 📝 Real-Time Text Samples per 300 Iterations\n\n"
    for r in records:
        ppl = math.exp(r['val_loss'])
        gib = r.get('gib_rate', 0.0)
        md_content += f"### 📌 Step {r['step']} (Train Loss: {r['train_loss']:.4f} | Val Loss: {r['val_loss']:.4f} | PPL: {ppl:.2f} | Non-Word Rate: {gib:.1f}%)\n"
        md_content += "```text\n"
        md_content += r['sample_text'].strip() + "\n"
        md_content += "```\n\n"

    md_content += "---\n\n"
    md_content += "## 🏆 Final Summary & Status\n\n"
    md_content += f"{final_summary}\n"

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

# ==============================================================================
# 5.5 Theoretical Minimum Loss Check (Sanity Verification)
# ==============================================================================
theoretical_loss = math.log(vocab_size)
initial_eval = estimate_loss()
loss_0 = initial_eval['train']
print(f"📊 Step 0 Theoretical Loss Check: Initial Train Loss = {loss_0:.4f} vs Theoretical Expected = {theoretical_loss:.4f} (-ln(1/{vocab_size}))", flush=True)
assert abs(loss_0 - theoretical_loss) < 0.8, f"Initial loss {loss_0:.4f} strays too far from theoretical {theoretical_loss:.4f}!"
print("✅ Theoretical Minimum Loss Check Passed! Model weights correctly initialized.\n", flush=True)

# ==============================================================================
# 6. Training Loop
# ==============================================================================
for iter in range(config['max_iters']):
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter % config['eval_interval'] == 0 or iter == config['max_iters'] - 1:
        losses = estimate_loss()
        curr_train_loss = losses['train']
        curr_val_loss = losses['val']
        
        history["steps"].append(iter)
        history["train_loss"].append(curr_train_loss)
        history["val_loss"].append(curr_val_loss)
        history["lr"].append(lr)
        
        with open(os.path.join(exp_dir, "loss_history.json"), "w") as f:
            json.dump(history, f, indent=4)
        
        print(f"Step {iter:4d} | LR: {lr:.2e} | Train Loss: {curr_train_loss:.4f} | Val Loss: {curr_val_loss:.4f}", flush=True)
        sample_text, gib_rate = log_sample(iter, exp_dir)

        records.append({
            "step": iter,
            "lr": lr,
            "train_loss": curr_train_loss,
            "val_loss": curr_val_loss,
            "sample_text": sample_text,
            "gib_rate": gib_rate
        })
        update_md_report(records)
        
        if (best_val_loss - curr_val_loss) > config['min_delta']:
            best_val_loss = curr_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            shutil.copyfile(best_model_path, "best_model.pt")
            shutil.copyfile(best_model_path, os.path.join(root_exp_dir, "best_model.pt"))
            with open("best_config.json", "w") as f:
                json.dump(config, f, indent=4)
            print(f"  └─ 🏆 New Best Val Loss: {best_val_loss:.4f}! Saved checkpoint (synced to ./best_model.pt).", flush=True)
        else:
            patience_counter += 1
            print(f"  └─ ⚠️ No significant improvement (> {config['min_delta']}). Patience ({patience_counter}/{config['patience']})", flush=True)
            if patience_counter >= config['patience']:
                print(f"\n🛑 Early Stopping Triggered at Step {iter}! Best Val Loss: {best_val_loss:.4f}", flush=True)
                break

    xb, yb = get_batch('train', batch_size=config['batch_size'], block_size=config['block_size'], device=device)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

save_loss_plot(history, exp_dir)
best_ppl = math.exp(best_val_loss)
summary_str = f"- **Best Validation Loss**: `{best_val_loss:.4f}`\n- **Best Perplexity (PPL)**: `{best_ppl:.2f}`\n- **Target Baseline (Karpathy 1.47)**: {'✅ Reached' if best_val_loss <= 1.47 else '⚠️ In Progress'}\n- **Report Location**: `Day 2/{exp_id}_training_report.md`"
update_md_report(records, final_summary=summary_str)

print(f"\n🎉 Experiment {exp_id} Complete! All artifacts & report saved to '{exp_dir}' and '{report_md_path}'.", flush=True)
