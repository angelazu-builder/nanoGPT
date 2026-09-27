"""
Manifest generation & verification for Context-Length Recovery Study
Preregistered specification: Section 9 & Section 10
"""

import numpy as np
from .config import (
    CONTEXT_LENGTHS,
    BATCH_SIZES,
    BLOCK_STEPS,
    RECOVERY_STEPS,
    EXTENSION_STEPS,
    N_ANCHOR_SEQUENCES,
    N_PROCESS_SEQUENCES,
    VAL_CONTEXT,
    VALIDATION_SEED,
)


def generate_scheduled_manifest(seed, train_length):
    """
    Generate the scheduled manifest for steps 1..2000.
    Explicit RNG namespace: seed * 10_000 + T.
    Indices sampled with replacement from [0, train_length - T).
    """
    manifest = {}
    for T in CONTEXT_LENGTHS:
        B = BATCH_SIZES[T]
        rng = np.random.RandomState(seed * 10_000 + T)
        manifest[T] = rng.randint(
            0,
            train_length - T,
            size=(BLOCK_STEPS, B),
        )
    return manifest


def generate_recovery_manifest(seed, train_length):
    """
    Generate the standard recovery manifest for steps 2001..2500 (T=256, B=16).
    Explicit RNG namespace: seed * 10_000 + 9_999.
    """
    rng = np.random.RandomState(seed * 10_000 + 9_999)
    return rng.randint(
        0,
        train_length - 256,
        size=(RECOVERY_STEPS, 16),
    )


def generate_extended_recovery_manifest(seed, train_length):
    """
    Generate the extended recovery manifest for steps 2501..3000 (T=256, B=16).
    Explicit RNG namespace: seed * 10_000 + 19_999.
    """
    rng = np.random.RandomState(seed * 10_000 + 19_999)
    return rng.randint(
        0,
        train_length - 256,
        size=(EXTENSION_STEPS, 16),
    )


def generate_validation_manifest(n_sequences=N_ANCHOR_SEQUENCES, context_length=VAL_CONTEXT, seed=VALIDATION_SEED, val_length=None):
    """
    Generate the fixed validation sequence start positions.
    First N_PROCESS_SEQUENCES form the process panel;
    All N_ANCHOR_SEQUENCES form the anchor panel.
    """
    if val_length is None:
        raise ValueError("val_length must be provided")
    rng = np.random.RandomState(seed)
    anchor_indices = rng.randint(
        0,
        val_length - context_length,
        size=n_sequences,
    )
    return anchor_indices


def verify_manifest_invariants(seed, train_length, val_length):
    """
    Required assertions from Section 9:
    1. Scheduled manifest shapes: (500, B) for each T
    2. Recovery manifest shape: (500, 16)
    3. Extended recovery manifest shape: (500, 16)
    4. Valid index bounds [0, length - T)
    5. Distinct RNG outputs across seeds
    """
    sched = generate_scheduled_manifest(seed, train_length)
    rec = generate_recovery_manifest(seed, train_length)
    ext = generate_extended_recovery_manifest(seed, train_length)
    
    for T in CONTEXT_LENGTHS:
        B = BATCH_SIZES[T]
        arr = sched[T]
        assert arr.shape == (500, B), f"Shape mismatch for T={T}: {arr.shape}"
        assert arr.min() >= 0, f"Negative index in scheduled manifest T={T}"
        assert arr.max() < train_length - T, f"Out of bounds in scheduled manifest T={T}"
        
    assert rec.shape == (500, 16), f"Recovery manifest shape mismatch: {rec.shape}"
    assert rec.min() >= 0 and rec.max() < train_length - 256
    
    assert ext.shape == (500, 16), f"Extended manifest shape mismatch: {ext.shape}"
    assert ext.min() >= 0 and ext.max() < train_length - 256
    
    # Assert separation from another seed
    other_seed = seed + 1
    other_sched = generate_scheduled_manifest(other_seed, train_length)
    other_rec = generate_recovery_manifest(other_seed, train_length)
    assert not np.array_equal(sched[256], other_sched[256]), "Manifests should be seed-specific"
    assert not np.array_equal(rec, other_rec), "Recovery manifests should be seed-specific"
    
    val_indices = generate_validation_manifest(val_length=val_length)
    assert len(val_indices) == N_ANCHOR_SEQUENCES
    assert val_indices.min() >= 0 and val_indices.max() < val_length - VAL_CONTEXT
    
    return True
