#!/opt/miniconda3/bin/python3
import torch
import torch.nn.functional as F
from config import get_latest_checkpoint, get_dataset_module
from model import MiniTransformerLM

# 1. Setup Device & Load Best Model Checkpoint
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model_path, cfg = get_latest_checkpoint()

tokenizer_type = cfg.get('tokenizer', 'char')
ds = get_dataset_module(tokenizer_type)
vocab_size = ds.vocab_size
encode = ds.encode
decode = ds.decode

print(f"🎭 Mini-Transformer Interactive Playground (Device: {device})")
print(f"📁 Checkpoint Source: '{model_path}'")
print(f"🔤 Tokenizer Mode: '{tokenizer_type.upper()}' (Vocab V = {vocab_size:,})")
print("=" * 60)

# Instantiate model matching architecture
model = MiniTransformerLM(
    vocab_size=vocab_size, 
    n_embd=cfg.get('n_embd', 256), 
    n_head=cfg.get('n_head', 8), 
    n_layer=cfg.get('n_layer', 6), 
    block_size=cfg.get('block_size', 256),
    pos_emb_type=cfg.get('pos_emb_type', 'absolute')
).to(device)

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Successfully loaded model weights from '{model_path}'!\n")
except Exception as e:
    print(f"❌ Error loading '{model_path}': {e}")
    exit(1)

model.eval()

# 2. Interactive Loop
print("💬 Enter a prompt (e.g. 'ROMEO:', 'HAMLET:', 'To be or not to be') or type 'exit' to quit.\n")

def generate_completion(prompt_text, max_new_tokens=200, temperature=0.8, top_k=40):
    try:
        encoded_prompt = encode(prompt_text)
    except Exception:
        encoded_prompt = [0]

    if not encoded_prompt:
        encoded_prompt = [0]

    idx = torch.tensor([encoded_prompt], dtype=torch.long, device=device)
    
    with torch.no_grad():
        generated_idx = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
            
    return decode(generated_idx[0].tolist())

if __name__ == "__main__":
    while True:
        try:
            user_input = input("👉 Enter Prompt: ")
            if user_input.strip().lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            if not user_input.strip():
                continue
                
            temp_str = input("🎚️ Set Temperature (0.1 = strict/deterministic, 0.8 = creative, default 0.8): ")
            try:
                temp = float(temp_str) if temp_str.strip() else 0.8
            except ValueError:
                temp = 0.8

            topk_str = input("🎯 Set Top-K Filter (e.g. 40, 0 or empty for disabled, default 40): ")
            try:
                top_k = int(topk_str) if topk_str.strip() and int(topk_str) > 0 else (40 if not topk_str.strip() else None)
            except ValueError:
                top_k = 40

            print("\n" + "-"*50)
            print(f"🤖 Generating completion (Temp={temp}, Top-K={top_k})...")
            print("-"*50)
            result = generate_completion(user_input, max_new_tokens=250, temperature=temp, top_k=top_k)
            print(result)
            print("-" * 50 + "\n")
        except KeyboardInterrupt:
            print("\nExiting interactive playground. Bye!")
            break
