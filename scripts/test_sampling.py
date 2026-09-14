#!/opt/miniconda3/bin/python3
import os
import torch
from config import get_latest_checkpoint
from dataset import encode, decode, vocab_size, stoi
from model import MiniTransformerLM

# 1. Setup Device & Load Checkpoint
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model_path, cfg = get_latest_checkpoint()

print(f"🔍 Sampling Coherence Inspection (Device: {device})")
print(f"📁 Loading Model Weights: '{model_path}'")
print(f"⚙️ Model Architecture: n_embd={cfg.get('n_embd', 256)}, n_head={cfg.get('n_head', 8)}, n_layer={cfg.get('n_layer', 6)}, block_size={cfg.get('block_size', 64)}\n")

model = MiniTransformerLM(
    vocab_size=vocab_size,
    n_embd=cfg.get('n_embd', 256),
    n_head=cfg.get('n_head', 8),
    n_layer=cfg.get('n_layer', 6),
    block_size=cfg.get('block_size', 64),
    pos_emb_type=cfg.get('pos_emb_type', 'absolute')
).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 2. Define Sampling Grid
sampling_configs = [
    {"temp": 1.0, "top_k": 40, "desc": "T = 1.0, top_k = 40 (High Creativity / Standard Sampling)"},
    {"temp": 0.8, "top_k": 40, "desc": "T = 0.8, top_k = 40 (Balanced Creativity & Stability)"},
    {"temp": 0.7, "top_k": 20, "desc": "T = 0.7, top_k = 20 (Focused Context & Low Tail Noise)"},
    {"temp": 0.6, "top_k": 10, "desc": "T = 0.6, top_k = 10 (Strict High-Confidence Greedy-ish Sampling)"},
]

fixed_prompt = "KING RICHARD III:\n"
fixed_seed = 42

valid_chars = [c for c in fixed_prompt if c in stoi]
encoded_prompt = encode("".join(valid_chars))
idx_base = torch.tensor([encoded_prompt], dtype=torch.long, device=device)

results = []

print("=" * 70)
print(f"🎯 Fixed Prompt: '{fixed_prompt.strip()}' | Random Seed: {fixed_seed}")
print("=" * 70 + "\n")

for cfg_item in sampling_configs:
    temp = cfg_item["temp"]
    top_k = cfg_item["top_k"]
    desc = cfg_item["desc"]
    
    # Fix seed for fair deterministic comparison
    torch.manual_seed(fixed_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fixed_seed)
    
    with torch.no_grad():
        generated_idx = model.generate(idx_base.clone(), max_new_tokens=250, temperature=temp, top_k=top_k)
    
    text = decode(generated_idx[0].tolist())
    results.append({
        "temp": temp,
        "top_k": top_k,
        "desc": desc,
        "text": text
    })
    
    print(f"📌 [{desc}]")
    print("-" * 50)
    print(text)
    print("-" * 50 + "\n")

# Write comparison report to Day 2 folder
day2_dir = "Day 2"
os.makedirs(day2_dir, exist_ok=True)
report_path = os.path.join(day2_dir, "sampling_coherence_comparison.md")

md_content = f"# 🧪 Sampling Coherence Comparison Report\n\n"
md_content += f"- **Model Checkpoint**: `{model_path}`\n"
md_content += f"- **Fixed Prompt**: `{fixed_prompt.strip()}`\n"
md_content += f"- **Fixed Random Seed**: `{fixed_seed}`\n\n"
md_content += "---\n\n"

for r in results:
    md_content += f"## 📌 {r['desc']}\n"
    md_content += "```text\n"
    md_content += r['text'].strip() + "\n"
    md_content += "```\n\n"

with open(report_path, "w", encoding="utf-8") as f:
    f.write(md_content)

print(f"✅ Comparison completed! Summary report saved to '{report_path}'.")
