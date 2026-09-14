import os
import requests
import torch

# 1. Download Tiny Shakespeare Dataset (~1.1MB text)
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "input.txt"

def load_data():
    if not os.path.exists(DATA_PATH):
        print("Downloading Tiny Shakespeare dataset...")
        res = requests.get(DATA_URL)
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            f.write(res.text)
        print("Download complete!")
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    return text

text = load_data()

# 2. Character-level Tokenizer & Vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Build string <-> integer bidirectional mapping
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]              # string to list of integers
decode = lambda l: ''.join([itos[i] for i in l])      # list of integers to string

# 3. Convert full text to 1D Tensor
data = torch.tensor(encode(text), dtype=torch.long)

# Train / Validation Split (90% train, 10% val)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split="train", batch_size=4, block_size=8, device="cpu"):
    """
    Generate a small batch of inputs (X) and targets (Y)
    - X has shape (B, T)
    - Y has shape (B, T) where Y[i, t] is the target token following X[i, t]
    """
    d = train_data if split == "train" else val_data
    # Pick random starting indices in the dataset
    ix = torch.randint(len(d) - block_size, (batch_size,))
    
    # Stack inputs (X) and targets (Y offset by 1 position)
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    
    return x.to(device), y.to(device)

if __name__ == "__main__":
    print(f"Total text length: {len(text)} characters")
    print(f"Vocabulary size V: {vocab_size}")
    print(f"Unique characters: {''.join(chars[:20])}...")
    
    sample_str = "Hello Shakespeare"
    encoded = encode(sample_str)
    decoded = decode(encoded)
    print(f"Original: '{sample_str}'")
    print(f"Encoded:  {encoded}")
    print(f"Decoded:  '{decoded}'")
    
    x, y = get_batch("train", batch_size=2, block_size=4)
    print("\nSample Batch X (Input Tokens, shape B=2, T=4):")
    print(x)
    print("Sample Batch Y (Target Tokens, shape B=2, T=4):")
    print(y)
