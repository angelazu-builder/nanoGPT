# Iteration 1 Protocol and Controls — Retrospective Reconstruction

> **Status:** Reconstructed after results from the original proposal, report, and `experiment_curriculum.py`. This was **not** frozen before training and must not be represented as a prospective preregistration.

## 1. Research question

Does the order in which a small Transformer encounters context lengths `T∈{32,64,128,256}` change its training dynamics and final validation performance?

## 2. Experimental setup

| Component | Iteration 1 setup |
|---|---|
| Model | `MiniTransformerLM`, approximately 4.8M parameters |
| Architecture | `d_model=256`, `n_head=8`, `n_layer=6`, RoPE, maximum block size 256 |
| Dataset | Tiny Shakespeare, fixed train/validation split |
| Tokenizer | Character level, vocabulary size 65 |
| Device | Apple Silicon M3 using MPS |
| Seed | 42 only |
| Optimizer | AdamW, weight decay `0.1` |
| Learning rate | Warmup then cosine decay from `1e-3` to `1e-4` over 2,500 steps |
| Training budget | 2,500 optimizer updates |
| Token throughput | `B×T=4096` target tokens per update |
| Evaluation | One fixed validation batch of 16 sequences at length 256 |

Batch size changed with context length to preserve token throughput:

| Context length | Batch size |
|---:|---:|
| 32 | 128 |
| 64 | 64 |
| 128 | 32 |
| 256 | 16 |

## 3. Experimental arms

| Arm | Schedule |
|---|---|
| Ascending curriculum | `32→64→128→256`, 625 steps per block |
| Shuffled | Same 625-step histogram at each context length, randomly permuted by step |
| Descending / anti-curriculum | `256→128→64→32`, 625 steps per block |
| Fixed-long baseline | `T=256` for all 2,500 steps |

## 4. What was held constant

Across all four arms:

- model class and architecture;
- tokenizer and train/validation split;
- initialization procedure and seed;
- optimizer type and hyperparameters;
- global learning-rate schedule;
- number of optimizer updates;
- target-token throughput per update;
- evaluation batch and evaluation checkpoints;
- maximum model context capacity.

Across the ascending, shuffled, and descending arms only:

- 625 updates at each of the four context lengths;
- total target-token exposure at each context length.

## 5. What varied

- temporal placement of the four context lengths;
- final context length immediately before evaluation;
- which context length coincided with each learning-rate and model-state phase;
- realized training examples, because batches were sampled online and were not paired across arms;
- context exposure of the fixed-long baseline.

## 6. Measurements

- validation loss and bits per character (BPC);
- attention entropy;
- effective representation rank;
- initialization gradient statistics;
- truncated-context evaluation at `T∈{32,64,128,256}`.

## 7. Known validity limitations

This iteration was exploratory and cannot support a clean causal order claim:

1. only one seed was used;
2. training batches were not paired across arms;
3. the original context-horizon evaluation scored different target positions at different horizons;
4. the original corpus probe used a resubstitution-biased estimator;
5. terminal context, temporal order, learning-rate phase, and optimizer history were entangled;
6. fixed token throughput did not imply fixed FLOPs because attention cost depends on context length;
7. rank and entropy diagnostics were interpreted too strongly as mechanisms.

The original report and its figures are preserved for audit, but their mechanism claims are not current conclusions.
