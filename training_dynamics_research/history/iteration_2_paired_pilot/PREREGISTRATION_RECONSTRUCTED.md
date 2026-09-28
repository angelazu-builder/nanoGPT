# Iteration 2 Protocol and Controls — Retrospective Reconstruction

> **Status:** Reconstructed after results from `experiment_paired.py`, saved artifacts, and the corrected paired-pilot report. This was **not** frozen before training and must not be represented as a prospective preregistration.

## 1. Research question

After correcting target alignment and pairing the training data within seed, does context-length order leave a detectable terminal performance difference?

## 2. Experimental setup

| Component | Iteration 2 setup |
|---|---|
| Model | `MiniTransformerLM`, approximately 4.8M parameters |
| Architecture | `d_model=256`, `n_head=8`, `n_layer=6`, RoPE, maximum block size 256 |
| Dataset | Tiny Shakespeare, 90/10 train/validation split |
| Tokenizer | Character level, vocabulary size 65 |
| Seeds | `42,43,44,45,46` |
| Optimizer | AdamW, weight decay `0.1`, gradient clipping `1.0` |
| Learning rate | Warmup then cosine decay from `1e-3` to `1e-4` over 2,500 steps |
| Training budget | 2,500 optimizer updates per arm and seed |
| Token throughput | `B×T=4096` target tokens per update |
| Evaluation checkpoints | Step 1 and every 250 steps through step 2500 |
| Evaluation panel | 16 fixed length-256 sequences per seed; the same panel used by all arms in that seed |
| Scored targets | Fixed final 32 target positions at every evaluation |

Batch size was `128,64,32,16` for context lengths `32,64,128,256`, respectively.

## 3. Experimental arms

| Arm | Schedule |
|---|---|
| Ascending curriculum | `32→64→128→256`, 625 steps per block |
| Shuffled | Same context-length histogram, randomly permuted by step |
| Descending / anti-curriculum | `256→128→64→32`, 625 steps per block |
| Fixed-long baseline | `T=256` for all 2,500 steps |

## 4. Pairing design

Within each seed:

- all arms began from the same seeded initialization;
- each context length had a pre-generated pool of training start indices;
- every arm using a given context length consumed the same context-specific pool;
- start indices were sampled with replacement, so pairing was preserved but windows were not guaranteed to be unique or non-overlapping;
- all arms used the same fixed validation sequences and target positions.

## 5. What was held constant

Across all arms within a seed:

- model architecture and initialization;
- tokenizer and corpus split;
- optimizer, gradient clipping, and global learning-rate schedule;
- number of optimizer updates;
- target-token throughput per update;
- context-specific data pools when the same context length was used;
- validation sequences, scored target positions, evaluation cadence, and metric;
- analysis code and final BPC definition.

Across ascending, shuffled, and descending arms only:

- 625 updates at each context length;
- total target-token exposure at each context length.

## 6. What varied

- temporal order of context lengths among the three matched-histogram arms;
- terminal context immediately before the final evaluation;
- interaction between context length, global learning-rate phase, model state, and AdamW moment history;
- context histogram for the fixed-long baseline.

The treatment therefore remained a bundled schedule effect, not an isolated abstract order effect.

## 7. Primary measurements

- final validation BPC at step 2500;
- paired seed-level differences between curriculum and each comparison arm;
- mean, standard deviation, paired t statistic, and p value across five seeds;
- secondary attention-entropy, effective-rank, gradient-noise, and truncated-context diagnostics.

No formal equivalence test was preregistered or powered. A nonsignificant difference could not establish equivalence.

## 8. Known validity limitations

1. the 16-sequence panel scored only 512 target-token predictions per evaluation;
2. five seeds had limited precision for effects around `0.01–0.02 BPC`;
3. terminal training context matched the main `T=256` evaluation for ascending but not descending;
4. learning-rate phase and optimizer-state history remained inseparable from block order;
5. the fixed-long arm was token-matched, not FLOP-matched;
6. one corpus, model scale, tokenizer, and optimizer limited external validity.

These unresolved confounds motivated Iteration 3's common long-context recovery stage and multi-horizon evaluation.
