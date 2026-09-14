#!/opt/miniconda3/bin/python3
import math
import torch
import torch.nn.functional as F
from config import get_latest_checkpoint
from dataset import get_batch, vocab_size, decode, encode
from model import MiniTransformerLM

# 1. Setup & Load Best Saved Checkpoint
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
best_model_path, cfg = get_latest_checkpoint()

print(f"📊 Running Objective Evaluation Suite on Device: {device}")
print(f"📁 Checkpoint Target: '{best_model_path}'")
print(f"⚙️ Model Specs: {cfg.get('n_embd', 256)}d / {cfg.get('n_head', 8)}h / {cfg.get('n_layer', 6)}l\n")

# Instantiate model matching target architecture
model = MiniTransformerLM(
    vocab_size=vocab_size, 
    n_embd=cfg.get('n_embd', 256), 
    n_head=cfg.get('n_head', 8), 
    n_layer=cfg.get('n_layer', 6), 
    block_size=cfg.get('block_size', 64)
).to(device)

try:
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"✅ Loaded trained weights from '{best_model_path}' successfully!")
except Exception as e:
    print(f"⚠️ Could not load '{best_model_path}', evaluating initialized weights. Error: {e}")

model.eval()

# ==============================================================================
# Metric 1: Val Loss & Perplexity (PPL = exp(Val Loss))
# ==============================================================================
eval_iters = 100
losses = torch.zeros(eval_iters)
with torch.no_grad():
    for k in range(eval_iters):
        X, Y = get_batch('val', batch_size=64, block_size=64, device=device)
        _, loss = model(X, Y)
        losses[k] = loss.item()

val_loss = losses.mean().item()
perplexity = math.exp(val_loss)

# Baseline metrics (Karpathy's standard char-level NanoGPT on Shakespeare)
karpathy_val_loss = 1.47
karpathy_ppl = math.exp(karpathy_val_loss)

print("="*60)
print("📈 METRIC 1: Loss & Perplexity (PPL)")
print("="*60)
print(f"  • Our Model Val Loss   : {val_loss:.4f}")
print(f"  • Our Model Perplexity : {perplexity:.2f} (lower is better, random=65.0)")
print(f"  • Karpathy Target PPL  : {karpathy_ppl:.2f} (Target Val Loss: {karpathy_val_loss:.2f})")
print(f"  • Performance Gap      : {(val_loss - karpathy_val_loss):+.4f} Loss units")

# ==============================================================================
# Metric 2: Diversity Sampling (Distinct-2 Bigram Diversity)
# ==============================================================================
prompt_text = "KING RICHARD:"
prompt_tokens = torch.tensor([encode(prompt_text)], dtype=torch.long, device=device)

num_samples = 5
temperature = 0.8
all_bigrams = set()
total_bigrams = 0

print("\n" + "="*60)
print(f"🎲 METRIC 2: Diversity Sampling for Prompt: '{prompt_text}' (Temp={temperature})")
print("="*60)

for i in range(num_samples):
    # Auto-regressive generation with temperature
    out_tokens = prompt_tokens.clone()
    with torch.no_grad():
        for _ in range(100):
            idx_cond = out_tokens[:, -64:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            out_tokens = torch.cat((out_tokens, next_token), dim=1)
    
    sample_str = decode(out_tokens[0].tolist())
    print(f"Sample [{i+1}]: {sample_str[:80]}...")
    
    # Calculate Bigram diversity
    words = sample_str.split()
    for j in range(len(words) - 1):
        bigram = (words[j], words[j+1])
        all_bigrams.add(bigram)
        total_bigrams += 1

distinct_2_ratio = len(all_bigrams) / max(1, total_bigrams)
print(f"\n  • Distinct-2 Bigram Diversity Ratio: {distinct_2_ratio:.2%} (higher means less repetition)")
print("="*60)
