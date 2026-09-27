"""
Unit tests for Recovery Study schedules, manifests, and invariants
Preregistered specification: Section 9 & Section 17
"""

import unittest
import numpy as np
import torch

from .config import (
    SEEDS,
    ARMS,
    SCHEDULES,
    BLOCK_STEPS,
    PRE_RECOVERY_STEPS,
    RECOVERY_STEPS,
    MAX_STEPS,
    CONTEXT_LENGTHS,
    BATCH_SIZES,
    MIN_LR,
    N_ANCHOR_SEQUENCES,
    N_PROCESS_SEQUENCES,
    VAL_CONTEXT,
)
from .schedules import build_scheduled_contexts, build_run_schedule, get_lr
from .manifests import (
    generate_scheduled_manifest,
    generate_recovery_manifest,
    generate_extended_recovery_manifest,
    generate_validation_manifest,
    verify_manifest_invariants,
)


class TestRecoveryStudyDesign(unittest.TestCase):
    
    def test_scheduled_contexts(self):
        """Verify 2000-step pre-recovery schedules have identical context histograms."""
        for arm in ARMS:
            contexts = build_scheduled_contexts(arm)
            self.assertEqual(len(contexts), PRE_RECOVERY_STEPS)
            # Each context length must appear exactly 500 times
            for T in CONTEXT_LENGTHS:
                count = contexts.count(T)
                self.assertEqual(
                    count, BLOCK_STEPS,
                    f"Arm '{arm}' context T={T} count was {count}, expected {BLOCK_STEPS}"
                )
                
    def test_run_schedule_structure(self):
        """Verify 2500-step run schedule properties."""
        for arm in ARMS:
            schedule = build_run_schedule(arm, total_steps=MAX_STEPS)
            self.assertEqual(len(schedule), MAX_STEPS)
            
            # Steps 1..2000 are pre-recovery
            pre_steps = schedule[:PRE_RECOVERY_STEPS]
            for s in pre_steps:
                self.assertFalse(s["is_recovery"])
                self.assertFalse(s["is_extension"])
                self.assertEqual(s["B"] * s["T"], 4096)
                self.assertTrue(0 <= s["occurrence_idx"] < 500)
                
            # Steps 2001..2500 are recovery
            rec_steps = schedule[PRE_RECOVERY_STEPS:MAX_STEPS]
            for idx, s in enumerate(rec_steps):
                self.assertTrue(s["is_recovery"])
                self.assertFalse(s["is_extension"])
                self.assertEqual(s["T"], 256)
                self.assertEqual(s["B"], 16)
                self.assertEqual(s["occurrence_idx"], idx)
                
    def test_lr_schedule_clamping(self):
        """Verify LR warms up, decays, and clamps at min_lr for extension."""
        lr_0 = get_lr(0)
        self.assertEqual(lr_0, 0.0)
        
        lr_peak = get_lr(400)
        self.assertAlmostEqual(lr_peak, 1e-3, places=6)
        
        lr_end = get_lr(MAX_STEPS)
        self.assertAlmostEqual(lr_end, MIN_LR, places=6)
        
        # Extended steps must remain clamped at min_lr
        lr_ext = get_lr(MAX_STEPS + 250)
        self.assertAlmostEqual(lr_ext, MIN_LR, places=6)
        lr_3000 = get_lr(3000)
        self.assertAlmostEqual(lr_3000, MIN_LR, places=6)
        
    def test_manifest_invariants_and_separation(self):
        """Verify manifest shapes, bounds, and RNG independence."""
        mock_train_len = 100_000
        mock_val_len = 20_000
        
        for seed in SEEDS:
            self.assertTrue(
                verify_manifest_invariants(seed, mock_train_len, mock_val_len)
            )
            
            # Check scheduled draws
            sched = generate_scheduled_manifest(seed, mock_train_len)
            for T in CONTEXT_LENGTHS:
                B = BATCH_SIZES[T]
                self.assertEqual(sched[T].shape, (500, B))
                self.assertTrue((sched[T] >= 0).all())
                self.assertTrue((sched[T] < mock_train_len - T).all())
                
            # Check recovery draws
            rec = generate_recovery_manifest(seed, mock_train_len)
            self.assertEqual(rec.shape, (500, 16))
            self.assertTrue((rec >= 0).all())
            self.assertTrue((rec < mock_train_len - 256).all())
            
    def test_validation_manifest_nesting(self):
        """Verify process panel is exact first 16 slice of anchor panel."""
        mock_val_len = 20_000
        anchor_indices = generate_validation_manifest(val_length=mock_val_len)
        self.assertEqual(len(anchor_indices), N_ANCHOR_SEQUENCES)
        
        process_indices = anchor_indices[:N_PROCESS_SEQUENCES]
        self.assertEqual(len(process_indices), N_PROCESS_SEQUENCES)
        np.testing.assert_array_equal(anchor_indices[:16], process_indices)


if __name__ == "__main__":
    unittest.main()
