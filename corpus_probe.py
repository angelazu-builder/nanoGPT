import math
import numpy as np
import torch
import dataset as ds

def analyze_corpus_dependencies(input_file="input.txt", max_lag=256):
    """
    Computes character-level n-gram conditional entropy H(X_t | X_{t-1:t-k}) for k in [1, max_lag]
    to measure how much predictive signal remains beyond 64 characters in Tiny Shakespeare.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    vocab = sorted(list(set(text)))
    char2idx = {ch: i for i, ch in enumerate(vocab)}
    data = np.array([char2idx[ch] for ch in text], dtype=np.int32)
    N = len(data)

    print(f"📊 Analyzing Corpus Dependencies on '{input_file}' ({N:,} characters, Vocab size: {len(vocab)})")

    # Marginal entropy H(X_t)
    counts = np.bincount(data)
    probs = counts / N
    h_marginal = -np.sum(probs * np.log2(probs + 1e-12))
    print(f"  - Marginal Character Entropy H(X): {h_marginal:.4f} bits/char")

    # Estimate conditional entropy for key lags using Markov order proxies and empirical sampling
    lags = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    conditional_entropies = {}

    # Sample sub-sequences to evaluate conditional predictability at different context horizons
    sample_size = 50000
    indices = np.random.choice(N - max_lag - 1, size=sample_size, replace=False)

    for lag in lags:
        # Measure empirical N-gram top-1 / top-5 accuracy or frequency match for lag k
        context_matches = {}
        for idx in indices:
            ctx = tuple(data[idx:idx+lag])
            target = data[idx+lag]
            if ctx not in context_matches:
                context_matches[ctx] = []
            context_matches[ctx].append(target)

        # Compute average entropy of P(target | context)
        entropies = []
        for ctx, targets in context_matches.items():
            t_counts = np.bincount(targets)
            t_probs = t_counts / len(targets)
            ent = -np.sum(t_probs * np.log2(t_probs + 1e-12))
            entropies.append((len(targets), ent))

        # Weighted average conditional entropy
        total_w = sum(w for w, e in entropies)
        avg_cond_ent = sum(w * e for w, e in entropies) / total_w
        info_gain = h_marginal - avg_cond_ent
        conditional_entropies[lag] = (avg_cond_ent, info_gain)
        print(f"  - Lag {lag:3d}: Cond Entropy H(X_t|X_{{t-{lag}:t-1}}) = {avg_cond_ent:.4f} bits/char | Info Gain = {info_gain:.4f} bits")

    return conditional_entropies

if __name__ == "__main__":
    np.random.seed(42)
    analyze_corpus_dependencies()
