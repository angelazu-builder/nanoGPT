import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryEmbedding(nn.Module):
    """ Rotary Position Embedding (RoPE) """
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return (
            self.cos_cached[:, :, :seq_len, :],
            self.sin_cached[:, :, :seq_len, :]
        )

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MiniEmbedding(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, dropout=0.1, pos_emb_type="absolute"):
        super().__init__()
        self.pos_emb_type = pos_emb_type
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        if pos_emb_type == "absolute":
            self.position_embedding_table = nn.Embedding(block_size, n_embd)
        else:
            self.position_embedding_table = None
        self.drop = nn.Dropout(dropout)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        if self.pos_emb_type == "absolute":
            pos = torch.arange(T, device=idx.device) # (T,)
            pos_emb = self.position_embedding_table(pos) # (T, n_embd)
            return self.drop(tok_emb + pos_emb)
        return self.drop(tok_emb)


class CausalSelfAttention(nn.Module):
    """ FlashAttention-backed Multi-Head Causal Self-Attention with optional RoPE """
    def __init__(self, n_embd, n_head, block_size, dropout=0.1, pos_emb_type="absolute"):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_size = n_embd // n_head
        self.dropout = dropout
        self.pos_emb_type = pos_emb_type

        # Key, Query, Value projections in a single batch Linear layer
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        if pos_emb_type == "rope":
            self.rotary_emb = RotaryEmbedding(self.head_size, max_position_embeddings=block_size)

    def forward(self, x):
        B, T, C = x.shape # batch size, sequence length, embedding dimensionality (n_embd)

        # Compute query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)

        if self.pos_emb_type == "rope":
            cos, sin = self.rotary_emb(v, seq_len=T)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # PyTorch 2.0 Scaled Dot-Product Attention (FlashAttention kernel under the hood)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C) # Re-assemble all head outputs side by side

        # Output projection
        return self.resid_dropout(self.c_proj(y))


class FeedForward(nn.Module):
    """ A simple 2-layer Linear network with GELU activation """
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """ Transformer block: Communication (SelfAttention) + Computation (FeedForward) """
    def __init__(self, n_embd, n_head, block_size, dropout=0.1, pos_emb_type="absolute"):
        super().__init__()
        self.sa = CausalSelfAttention(n_embd, n_head, block_size, dropout, pos_emb_type)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MiniTransformerLM(nn.Module):
    """ Complete Mini-Transformer Language Model with FlashAttention & RoPE Support """
    def __init__(self, vocab_size, n_embd=256, n_head=8, n_layer=6, block_size=64, dropout=0.1, pos_emb_type="absolute"):
        super().__init__()
        self.block_size = block_size
        self.pos_emb_type = pos_emb_type
        self.embedding = MiniEmbedding(vocab_size, n_embd, block_size, dropout, pos_emb_type)
        self.blocks = nn.Sequential(*[
            TransformerBlock(n_embd, n_head, block_size, dropout, pos_emb_type) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Initialize weights with standard 0.02 normal distribution
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.embedding(idx) # (B, T, n_embd)
        x = self.blocks(x)      # (B, T, n_embd)
        x = self.ln_f(x)        # (B, T, n_embd)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate):
        """ Separate 2D matrix weights (apply weight_decay) from 1D biases & LayerNorms (0 weight_decay) """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"parameters {inter_params} made it into both decay/no_decay sets!"
        assert len(param_dict.keys() - union_params) == 0, f"parameters {param_dict.keys() - union_params} were not separated into decay/no_decay sets!"

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
        return optimizer

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """ Auto-regressive text generation with Temperature & Top-K sampling """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5) # Apply temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == "__main__":
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"⚡ Testing FlashAttention + RoPE MiniTransformerLM on device: {device}")
    
    vocab_size = 65
    model = MiniTransformerLM(vocab_size=vocab_size, n_embd=256, n_head=8, n_layer=6, block_size=64, pos_emb_type="rope").to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total Trainable Parameters (RoPE): {n_params:,} (~{n_params/1e6:.2f}M)")

    # Test forward pass
    dummy_x = torch.randint(0, vocab_size, (2, 32), device=device)
    logits, loss = model(dummy_x, dummy_x)
    print(f"Dummy Forward Pass Logits Shape: {logits.shape}, Loss: {loss.item():.4f}")
