"""
CE-1: Corrected Corpus Dependency Probe
========================================
Fix: Train/held-out split for n-gram conditional entropy estimation.
     Laplace smoothing (alpha=0.1) for OOV contexts in held-out set.

Scientific question: Does the in-sample collapse (0.0047 bits/char at lag=16)
reflect genuine corpus structure, or resubstitution/memorization bias?

Control:   all 50,000 pairs used for both n-gram table AND entropy scoring
Treatment: 40,000 pairs build n-gram table; 10,000 held-out pairs score entropy
"""

import math
import numpy as np
import json
import os

def analyze_corpus_dependencies_v2(
    input_file="input.txt",
    max_lag=256,
    sample_size=50000,
    train_frac=0.8,
    alpha=0.1,          # Laplace smoothing concentration
    seed=42
):
    np.random.seed(seed)

    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    vocab = sorted(list(set(text)))
    V = len(vocab)
    char2idx = {ch: i for i, ch in enumerate(vocab)}
    data = np.array([char2idx[ch] for ch in text], dtype=np.int32)
    N = len(data)

    print(f"\n{'='*65}")
    print(f"CE-1: Corrected Corpus Dependency Probe")
    print(f"{'='*65}")
    print(f"Corpus: '{input_file}'  |  N={N:,} chars  |  V={V}")
    print(f"Sample size: {sample_size:,}  |  Train frac: {train_frac:.0%}  |  Laplace α={alpha}")
    print(f"{'='*65}\n")

    # Marginal entropy H(X_t)
    counts = np.bincount(data, minlength=V)
    probs = counts / N
    h_marginal = -np.sum(probs * np.log2(probs + 1e-12))
    print(f"Marginal Character Entropy H(X): {h_marginal:.4f} bits/char\n")

    lags = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    results = {}

    # Sample all (context, target) pairs once
    indices = np.random.choice(N - max_lag - 1, size=sample_size, replace=False)
    n_train = int(sample_size * train_frac)
    train_indices = indices[:n_train]
    held_out_indices = indices[n_train:]

    print(f"{'Lag':>5}  {'Insample H':>12}  {'Heldout H':>12}  {'Bias Δ':>10}  {'InfoGain(held)':>14}  {'InfoGain(insample)':>18}")
    print("-" * 80)

    for lag in lags:
        # ── Build n-gram table on TRAIN set ──────────────────────────────────
        ctx_counts_train = {}
        for idx in train_indices:
            ctx = tuple(data[idx:idx + lag])
            tgt = data[idx + lag]
            if ctx not in ctx_counts_train:
                ctx_counts_train[ctx] = np.zeros(V, dtype=np.float32)
            ctx_counts_train[ctx][tgt] += 1.0

        # ── Score entropy on HELD-OUT set (with Laplace smoothing) ───────────
        held_ents = []
        held_weights = []
        n_oov = 0  # held-out contexts not seen in train
        for idx in held_out_indices:
            ctx = tuple(data[idx:idx + lag])
            tgt = data[idx + lag]
            if ctx in ctx_counts_train:
                raw = ctx_counts_train[ctx] + alpha   # Laplace smoothed
            else:
                # Context never seen in train → full uniform prior
                raw = np.full(V, alpha, dtype=np.float32)
                n_oov += 1
            p = raw / raw.sum()
            # Entropy of this smoothed distribution
            ent = -np.sum(p * np.log2(p + 1e-12))
            held_ents.append(ent)
            held_weights.append(1.0)

        h_heldout = float(np.mean(held_ents))  # uniform weighting over held-out

        # ── Score entropy IN-SAMPLE (original biased method) ─────────────────
        ctx_matches_insample = {}
        for idx in indices:
            ctx = tuple(data[idx:idx + lag])
            tgt = data[idx + lag]
            if ctx not in ctx_matches_insample:
                ctx_matches_insample[ctx] = []
            ctx_matches_insample[ctx].append(tgt)

        insample_ents = []
        insample_weights = []
        for ctx, tgts in ctx_matches_insample.items():
            tc = np.bincount(tgts, minlength=V)
            tp = tc / len(tgts)
            e = -np.sum(tp * np.log2(tp + 1e-12))
            insample_ents.append((len(tgts), e))

        total_w = sum(w for w, e in insample_ents)
        h_insample = sum(w * e for w, e in insample_ents) / total_w

        bias_delta = h_insample - h_heldout
        info_gain_heldout   = h_marginal - h_heldout
        info_gain_insample  = h_marginal - h_insample
        pct_explained_held  = 100.0 * info_gain_heldout  / h_marginal
        pct_explained_ins   = 100.0 * info_gain_insample / h_marginal

        print(
            f"{lag:>5}  "
            f"{h_insample:>12.4f}  "
            f"{h_heldout:>12.4f}  "
            f"{bias_delta:>+10.4f}  "
            f"{info_gain_heldout:>8.4f} ({pct_explained_held:5.1f}%)  "
            f"{info_gain_insample:>8.4f} ({pct_explained_ins:5.1f}%)"
            f"  [OOV={n_oov}]"
        )

        results[lag] = {
            "lag": lag,
            "h_marginal": float(h_marginal),
            "h_insample": float(h_insample),
            "h_heldout": float(h_heldout),
            "bias_delta": float(bias_delta),
            "info_gain_heldout": float(info_gain_heldout),
            "info_gain_insample": float(info_gain_insample),
            "pct_explained_heldout": float(pct_explained_held),
            "pct_explained_insample": float(pct_explained_ins),
            "n_oov_heldout": int(n_oov),
        }

    print("\n")
    print("=== CE-1 VERDICT ===")
    lag16 = results.get(16, {})
    if lag16:
        print(f"Lag=16  insample: {lag16['h_insample']:.4f} bits/char  ({lag16['pct_explained_insample']:.1f}% of H explained)")
        print(f"Lag=16  held-out: {lag16['h_heldout']:.4f} bits/char  ({lag16['pct_explained_heldout']:.1f}% of H explained)")
        print(f"Bias Δ at lag=16: {lag16['bias_delta']:+.4f} bits/char")
        if lag16['h_heldout'] < 0.05:
            verdict = "Directionally SAME as original: corpus IS locally dominated at lag≤16. Only the exact number was inflated."
        else:
            verdict = "Original claim DOES NOT hold: entropy at lag=16 is not near-zero on held-out data. Resubstitution bias was the cause."
        print(f"Verdict: {verdict}")

    # Sensitivity check: what if held-out is smaller?
    print("\n--- Sensitivity: OOV rate by lag ---")
    for lag in lags:
        print(f"  lag={lag:3d}: OOV contexts in held-out = {results[lag]['n_oov_heldout']} / {sample_size - n_train}")

    # Save results
    os.makedirs("results", exist_ok=True)
    out_path = "results/ce1_corpus_probe_corrected.json"
    with open(out_path, "w") as f:
        json.dump({
            "h_marginal": float(h_marginal),
            "V": V, "N": N,
            "sample_size": sample_size,
            "train_frac": train_frac,
            "alpha": alpha,
            "lags": results
        }, f, indent=2)
    print(f"\nResults saved to '{out_path}'")
    return results


if __name__ == "__main__":
    analyze_corpus_dependencies_v2()
