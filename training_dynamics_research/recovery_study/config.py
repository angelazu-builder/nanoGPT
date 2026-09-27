"""
Configuration & Constants for Context-Length Recovery Study
Preregistered specification: PREREGISTRATION.md (commit f1bf7b7)
"""

import os

# Device & Hardware
DEVICE = "mps"  # Apple Silicon M3 24GB

# Seeds & Arms
SEEDS = [42, 43, 44]
ARMS = ["ascending", "descending", "nonmonotonic"]

SCHEDULES = {
    "ascending": [32, 64, 128, 256],
    "descending": [256, 128, 64, 32],
    "nonmonotonic": [64, 256, 32, 128],
}

# Training Budget
TARGET_TOKENS_PER_STEP = 4096
BLOCK_STEPS = 500
PRE_RECOVERY_STEPS = 2000
RECOVERY_STEPS = 500
MAX_STEPS = 2500

# Adaptive Extension Budget (Hard Cap)
EXTENSION_STEPS = 500
HARD_CAP_STEPS = 3000

# Batch sizes ensuring B * T = 4096
BATCH_SIZES = {
    32: 128,
    64: 64,
    128: 32,
    256: 16,
}
CONTEXT_LENGTHS = [32, 64, 128, 256]

# Optimizer & LR Schedule
LR = 1e-3
MIN_LR = 1e-4
WARMUP_STEPS = 400
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

# Model Architecture
N_EMBD = 256
N_HEAD = 8
N_LAYER = 6
BLOCK_SIZE = 256
POS_EMB_TYPE = "rope"
DROPOUT = 0.0

# Validation Architecture (Nested Panel)
N_PROCESS_SEQUENCES = 16
N_ANCHOR_SEQUENCES = 32
VAL_CONTEXT = 256
FIXED_TARGET_LEN = 32  # Score the final 32 target positions
VALIDATION_SEED = 20260927

# Process Checkpoints (17 checkpoints)
PROCESS_EVAL_STEPS = [
    250,
    500, 501,
    750,
    1000, 1001,
    1250,
    1500, 1501,
    1750,
    2000, 2001, 2010, 2050,
    2100, 2250, 2500,
]

# Anchor Checkpoints (higher-precision endpoints)
ANCHOR_EVAL_STEPS = [2000, 2500]

# Extended Recovery Checkpoints
EXTENSION_PROCESS_EVAL_STEPS = [2600, 2750, 2900, 3000]
EXTENSION_ANCHOR_EVAL_STEPS = [3000]

# Arm Execution Order per Seed (Section 17)
ARM_ROTATION = {
    42: ["ascending", "descending", "nonmonotonic"],
    43: ["descending", "nonmonotonic", "ascending"],
    44: ["nonmonotonic", "ascending", "descending"],
}

# Decision Thresholds (Section 16 & Section 3)
THRESHOLD_MIN_PRE_GAP = 0.02           # BPC (denominator interpretable threshold)
THRESHOLD_RECOVERY_RATIO = 0.25        # 75% recovery criterion
THRESHOLD_RESIDUAL_PERSISTENT = 0.03   # BPC (minimum effect for mechanism study)
THRESHOLD_PLATEAU_250 = 0.025          # BPC per 250 recovery steps (preserves 0.01 / 100 steps)

# Results Storage
BASE_RESULTS_DIR = "results/recovery_study"
SMOKE_RESULTS_DIR = "results/recovery_study_smoke"
