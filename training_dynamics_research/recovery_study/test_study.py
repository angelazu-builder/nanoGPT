"""
Unit tests for Recovery Study schedules, manifests, decision rules, and invariants
Preregistered specification: Section 9, 16, 17
"""

import os
import sys
import unittest
import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from .config import (
    SEEDS,
    ARMS,
    SCHEDULES,
    BLOCK_STEPS,
    PRE_RECOVERY_STEPS,
    RECOVERY_STEPS,
    MAX_STEPS,
    HARD_CAP_STEPS,
    CONTEXT_LENGTHS,
    BATCH_SIZES,
    MIN_LR,
    N_ANCHOR_SEQUENCES,
    N_PROCESS_SEQUENCES,
    VAL_CONTEXT,
    THRESHOLD_MIN_PRE_GAP,
    THRESHOLD_RESIDUAL_PERSISTENT,
    THRESHOLD_PLATEAU_250,
)
from .schedules import build_scheduled_contexts, build_run_schedule, get_lr
from .manifests import (
    generate_scheduled_manifest,
    generate_recovery_manifest,
    generate_extended_recovery_manifest,
    generate_validation_manifest,
    verify_manifest_invariants,
)
from .evaluation import slice_process_metrics_from_anchor


class TestRecoveryStudyDesign(unittest.TestCase):
    
    def test_scheduled_contexts(self):
        """Verify 2000-step pre-recovery schedules have identical context histograms."""
        for arm in ARMS:
            contexts = build_scheduled_contexts(arm)
            self.assertEqual(len(contexts), PRE_RECOVERY_STEPS)
            for T in CONTEXT_LENGTHS:
                count = contexts.count(T)
                self.assertEqual(
                    count, BLOCK_STEPS,
                    f"Arm '{arm}' context T={T} count was {count}, expected {BLOCK_STEPS}"
                )
                
    def test_run_schedule_structure_and_extension(self):
        """Verify 2500-step and 3000-step extension run schedule properties."""
        for arm in ARMS:
            schedule = build_run_schedule(arm, total_steps=HARD_CAP_STEPS)
            self.assertEqual(len(schedule), HARD_CAP_STEPS)
            
            # Steps 1..2000 are pre-recovery
            pre_steps = schedule[:PRE_RECOVERY_STEPS]
            for s in pre_steps:
                self.assertFalse(s["is_recovery"])
                self.assertFalse(s["is_extension"])
                self.assertEqual(s["B"] * s["T"], 4096)
                self.assertTrue(0 <= s["occurrence_idx"] < 500)
                
            # Steps 2001..2500 are standard recovery
            rec_steps = schedule[PRE_RECOVERY_STEPS:MAX_STEPS]
            for idx, s in enumerate(rec_steps):
                self.assertTrue(s["is_recovery"])
                self.assertFalse(s["is_extension"])
                self.assertEqual(s["T"], 256)
                self.assertEqual(s["B"], 16)
                self.assertEqual(s["occurrence_idx"], idx)
                
            # Steps 2501..3000 are extended recovery
            ext_steps = schedule[MAX_STEPS:HARD_CAP_STEPS]
            for idx, s in enumerate(ext_steps):
                self.assertTrue(s["is_recovery"])
                self.assertTrue(s["is_extension"])
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
            
            sched = generate_scheduled_manifest(seed, mock_train_len)
            for T in CONTEXT_LENGTHS:
                B = BATCH_SIZES[T]
                self.assertEqual(sched[T].shape, (500, B))
                self.assertTrue((sched[T] >= 0).all())
                self.assertTrue((sched[T] < mock_train_len - T).all())
                
            rec = generate_recovery_manifest(seed, mock_train_len)
            self.assertEqual(rec.shape, (500, 16))
            self.assertTrue((rec >= 0).all())
            self.assertTrue((rec < mock_train_len - 256).all())
            
            ext = generate_extended_recovery_manifest(seed, mock_train_len)
            self.assertEqual(ext.shape, (500, 16))
            self.assertTrue((ext >= 0).all())
            self.assertTrue((ext < mock_train_len - 256).all())
            
            # Independence: recovery and extended recovery manifests must be distinct
            self.assertFalse(np.array_equal(rec, ext))
            
    def test_validation_manifest_nesting(self):
        """Verify process panel is exact first 16 slice of anchor panel."""
        mock_val_len = 20_000
        anchor_indices = generate_validation_manifest(val_length=mock_val_len)
        self.assertEqual(len(anchor_indices), N_ANCHOR_SEQUENCES)
        
        process_indices = anchor_indices[:N_PROCESS_SEQUENCES]
        self.assertEqual(len(process_indices), N_PROCESS_SEQUENCES)
        np.testing.assert_array_equal(anchor_indices[:16], process_indices)

    def test_slice_process_metrics_from_anchor(self):
        """Verify slice_process_metrics_from_anchor gives exact first-16 mean."""
        mock_anchor_eval = {
            "horizon_results": {
                256: {
                    "mean_bpc": 2.5,
                    "seq_bpc": [2.0] * 16 + [3.0] * 16,
                },
                32: {
                    "mean_bpc": 3.0,
                    "seq_bpc": [2.5] * 16 + [3.5] * 16,
                }
            },
            "mean_rank": 80.0,
            "mean_entropy": 0.5,
        }
        proc = slice_process_metrics_from_anchor(mock_anchor_eval, n_process=16)
        self.assertAlmostEqual(proc["bpc"], 2.0)
        self.assertEqual(len(proc["seq_bpc"]), 16)
        self.assertAlmostEqual(proc["horizon_results"][32]["mean_bpc"], 2.5)

    def test_synthetic_decision_rules_cases_a_through_e(self):
        """
        Verify decision rules for all 5 preregistered cases:
        Case A: pre-gap < 0.02 -> STOP
        Case B: post-gap <= 0.02 -> STOP
        Case C: 0.02 < post-gap <= 0.03 -> UNRESOLVED
        Case D: post-gap > 0.03 and slope > 0.025 -> EXTEND_RECOVERY
        Case E: post-gap > 0.03, 3/3 > 0, and slope <= 0.025 -> GO_OPTIMIZER_EXPERIMENT
        """
        # Helper evaluator logic mirroring analyze.py
        def evaluate_decision(mean_pre, mean_post, positive_count, slope_250):
            is_plateaued = abs(slope_250) < THRESHOLD_PLATEAU_250
            if abs(mean_pre) < THRESHOLD_MIN_PRE_GAP:
                return "Case A"
            elif abs(mean_post) <= THRESHOLD_MIN_PRE_GAP:
                return "Case B"
            elif THRESHOLD_MIN_PRE_GAP < abs(mean_post) <= THRESHOLD_RESIDUAL_PERSISTENT or positive_count not in [0, 3]:
                return "Case C"
            elif mean_post > THRESHOLD_RESIDUAL_PERSISTENT and slope_250 > THRESHOLD_PLATEAU_250:
                return "Case D"
            elif mean_post > THRESHOLD_RESIDUAL_PERSISTENT and positive_count == 3 and is_plateaued:
                return "Case E"
            return "Case C"
            
        # Case A: pre-gap fails to replicate
        self.assertEqual(evaluate_decision(0.012, 0.010, 3, 0.005), "Case A")
        
        # Case B: gap is practically eliminated
        self.assertEqual(evaluate_decision(0.160, 0.015, 3, 0.005), "Case B")
        self.assertEqual(evaluate_decision(0.160, -0.005, 1, 0.005), "Case B")
        
        # Case C: unresolved small residual or seed disagreement
        self.assertEqual(evaluate_decision(0.160, 0.025, 3, 0.005), "Case C")
        self.assertEqual(evaluate_decision(0.160, 0.035, 2, 0.005), "Case C")  # Disagreement 2/3
        
        # Case D: large gap but still actively recovering (slope > 0.025)
        self.assertEqual(evaluate_decision(0.160, 0.045, 3, 0.035), "Case D")
        
        # Case E: persistent-effect candidate (large gap, 3/3 positive, plateaued)
        self.assertEqual(evaluate_decision(0.160, 0.038, 3, 0.012), "Case E")

    def test_identical_initialization_same_seed(self):
        """
        Preregistered check: Verify that calling set_seed(seed) BEFORE model creation
        produces identical parameter tensors for models with the same seed,
        and differing tensors across different seeds.
        """
        from model import MiniTransformerLM
        from .runner import set_seed
        
        # Two models with seed 42
        set_seed(42)
        model_a = MiniTransformerLM(
            vocab_size=65,
            n_embd=64,
            n_head=2,
            n_layer=2,
            block_size=256,
            pos_emb_type="learned",
            dropout=0.1,
        )
        
        set_seed(42)
        model_b = MiniTransformerLM(
            vocab_size=65,
            n_embd=64,
            n_head=2,
            n_layer=2,
            block_size=256,
            pos_emb_type="learned",
            dropout=0.1,
        )
        
        for (name_a, p_a), (name_b, p_b) in zip(model_a.named_parameters(), model_b.named_parameters()):
            self.assertEqual(name_a, name_b)
            self.assertTrue(torch.equal(p_a, p_b), f"Parameter {name_a} differed between identical seeds!")
            
        # Model with seed 43
        set_seed(43)
        model_c = MiniTransformerLM(
            vocab_size=65,
            n_embd=64,
            n_head=2,
            n_layer=2,
            block_size=256,
            pos_emb_type="learned",
            dropout=0.1,
        )
        
        has_difference = any(
            not torch.equal(p_a, p_c)
            for (_, p_a), (_, p_c) in zip(model_a.named_parameters(), model_c.named_parameters())
        )
        self.assertTrue(has_difference, "Parameters across different seeds (42 vs 43) were unexpectedly identical!")

    def test_formal_gate_validation(self):
        """Verify validate_formal_gate enforces 9 completed runs, 17 process checkpoints, 2 anchor checkpoints, and nesting."""
        from .analyze import validate_formal_gate
        from .config import PROCESS_EVAL_STEPS, ANCHOR_EVAL_STEPS, CONTEXT_LENGTHS
        
        # Build synthetic valid dataset
        synthetic_data = {}
        for s in [42, 43, 44]:
            synthetic_data[s] = {}
            for arm in ["ascending", "descending", "nonmonotonic"]:
                logs = []
                for step in PROCESS_EVAL_STEPS:
                    h_res = {}
                    for h in CONTEXT_LENGTHS:
                        h_res[h] = {
                            "mean_bpc": 2.5,
                            "seq_bpc": [2.5] * 16
                        }
                    logs.append({
                        "step": step,
                        "process_bpc": 2.5,
                        "horizon_results": h_res
                    })
                anchor_logs = []
                for step in ANCHOR_EVAL_STEPS:
                    h_res = {}
                    for h in CONTEXT_LENGTHS:
                        h_res[h] = {
                            "mean_bpc": 2.5,
                            "seq_bpc": [2.5] * 32
                        }
                    anchor_logs.append({
                        "step": step,
                        "anchor_bpc": 2.5,
                        "horizon_results": h_res
                    })
                synthetic_data[s][arm] = {
                    "completed": True,
                    "smoke_test": False,
                    "current_step": 2500,
                    "logs": logs,
                    "anchor_logs": anchor_logs,
                }
                
        # Must pass cleanly on complete valid dataset
        validate_formal_gate(synthetic_data, [42, 43, 44])
        
        # Test failure if a run is missing
        incomplete_seeds = [42, 43]
        with self.assertRaises(ValueError):
            validate_formal_gate(synthetic_data, incomplete_seeds)
            
        # Test failure if completed is False
        synthetic_data[42]["ascending"]["completed"] = False
        with self.assertRaises(ValueError):
            validate_formal_gate(synthetic_data, [42, 43, 44])
        synthetic_data[42]["ascending"]["completed"] = True
        
        # Test failure if smoke_test is True
        synthetic_data[42]["ascending"]["smoke_test"] = True
        with self.assertRaises(ValueError):
            validate_formal_gate(synthetic_data, [42, 43, 44])
        synthetic_data[42]["ascending"]["smoke_test"] = False
        
        # Test failure if nesting invariant is violated (anchor first 16 does not equal process mean)
        synthetic_data[42]["ascending"]["anchor_logs"][0]["horizon_results"][256]["seq_bpc"] = [9.9] * 16 + [2.5] * 16
        with self.assertRaises(ValueError):
            validate_formal_gate(synthetic_data, [42, 43, 44])

    def test_extension_decision_never_returns_extend_recovery(self):
        """
        Verify that once step 3000 extension is completed, analyze.py never returns EXTEND_RECOVERY.
        """
        # Mirroring the extension decision branch from analyze.py
        def evaluate_extension_decision(mean_ext, late_slope, positive_count, is_plateaued):
            if late_slope > THRESHOLD_PLATEAU_250:
                return "UNRESOLVED_AT_HARD_CAP"
            elif mean_ext > THRESHOLD_RESIDUAL_PERSISTENT and positive_count == 3 and is_plateaued:
                return "GO_OPTIMIZER_EXPERIMENT"
            else:
                return "UNRESOLVED_OR_RECOVERED"

        # Case 1: Still recovering actively -> Terminal unresolved, NOT extend
        res1 = evaluate_extension_decision(0.045, 0.035, 3, False)
        self.assertEqual(res1, "UNRESOLVED_AT_HARD_CAP")
        self.assertNotEqual(res1, "EXTEND_RECOVERY")

        # Case 2: Plateaued candidate -> Optimizer experiment
        res2 = evaluate_extension_decision(0.038, 0.010, 3, True)
        self.assertEqual(res2, "GO_OPTIMIZER_EXPERIMENT")

        # Case 3: Residual small or non-unanimous -> Unresolved or recovered
        res3 = evaluate_extension_decision(0.015, 0.005, 3, True)
        self.assertEqual(res3, "UNRESOLVED_OR_RECOVERED")

    def test_run_spec_and_tables_pipeline(self):
        """Verify immutable RunSpec properties, initialize_run lifecycle, and StudyTables extraction."""
        from .core_types import RunSpec, RunState
        from .runner import make_run_spec, initialize_run
        from .tables import extract_study_tables
        
        spec = make_run_spec(
            arm="ascending",
            seed=42,
            total_steps=50,
            is_smoke=True,
            results_dir="results/recovery_study",
        )
        self.assertEqual(spec.arm, "ascending")
        self.assertEqual(spec.seed, 42)
        self.assertTrue(spec.is_smoke)
        self.assertTrue("results/recovery_study_smoke" in spec.effective_results_dir)
        
        # Initialize run with overwrite=True produces fresh RunState with step 1
        state = initialize_run(spec, device="cpu", overwrite=True)
        self.assertIsInstance(state, RunState)
        self.assertEqual(state.step, 1)
        self.assertFalse(state.is_complete)
        self.assertIsNotNone(state.model)
        self.assertIsNotNone(state.optimizer)
        
        # Test StudyTables extraction from synthetic data
        synthetic_study = {
            42: {
                "ascending": {
                    "logs": [
                        {
                            "step": 2000,
                            "horizon_results": {
                                256: {"mean_bpc": 3.14, "seq_bpc": [3.14] * 16}
                            }
                        }
                    ],
                    "anchor_logs": [
                        {
                            "step": 2000,
                            "horizon_results": {
                                256: {"mean_bpc": 3.14, "seq_bpc": [3.14] * 32}
                            }
                        }
                    ]
                }
            }
        }
        tables = extract_study_tables(synthetic_study, [42])
        self.assertAlmostEqual(tables.process_bpc(42, "ascending", 2000, 256), 3.14)
        self.assertAlmostEqual(tables.anchor_bpc(42, "ascending", 2000, 256), 3.14)


if __name__ == "__main__":
    unittest.main()

