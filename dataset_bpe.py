import os
import re
import torch

# Load Tiny Shakespeare Corpus
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
with open(input_file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Try tiktoken first
IS_TIKTOKEN = False
try:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    # Test encoding a small snippet to ensure data file is cached
    _ = enc.encode("Hello world")
    vocab_size = enc.n_vocab
    IS_TIKTOKEN = True
    print(f"✅ BPE Tokenizer loaded using tiktoken (gpt2 vocab_size: {vocab_size:,})")
except Exception as e:
    print(f"⚠️ tiktoken offline/unavailable ({e}). Initializing Lightweight Subword Vocabulary.")

if IS_TIKTOKEN:
    data = torch.tensor(enc.encode(text), dtype=torch.long)
    def encode(s):
        return enc.encode(s)
    def decode(l):
        return enc.decode(l)
else:
    # Lightweight Custom Subword/Word Tokenizer (Vocab size ~5000)
    # Extracts words and whitespace/punctuation tokens
    words = re.findall(r"\w+|[^\w\s]|\s+", text)
    vocab = sorted(list(set(words)))
    vocab_size = len(vocab)
    wtoi = {w: i for i, i_w in enumerate(vocab) for w in [i_w]}
    itow = {i: w for i, w in enumerate(vocab)}

    data = torch.tensor([wtoi[w] for w in words], dtype=torch.long)
    def encode(s):
        tokens = re.findall(r"\w+|[^\w\s]|\s+", s)
        return [wtoi.get(w, 0) for w in tokens]
    def decode(l):
        return "".join([itow.get(i, "") for i in l])

char_len = len(text)
token_len = len(data)
compression_ratio = char_len / token_len
print(f"📊 Dataset Stats: {char_len:,} Chars -> {token_len:,} Subword Tokens (Compression: {compression_ratio:.2f}x)")

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_bpe_batch(split, batch_size=64, block_size=256, device='cpu'):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

if __name__ == "__main__":
    sample_x, sample_y = get_bpe_batch('train', batch_size=4, block_size=16)
    print(f"Sample Batch X shape: {sample_x.shape}")
    print(f"Sample Decoded Text:\n{decode(sample_x[0].tolist())}")
