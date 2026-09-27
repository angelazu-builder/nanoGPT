"""
Core types, specifications, and state definitions for Recovery Study.
Enforces typed immutability for experiment specifications and strict unidirectional state flow.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch

from .config import BASE_RESULTS_DIR, SMOKE_RESULTS_DIR


@dataclass(frozen=True)
class RunSpec:
    """Immutable specification for a single experimental run."""
    arm: str
    seed: int
    total_steps: int
    is_smoke: bool = False
    is_extension: bool = False
    results_dir: str = BASE_RESULTS_DIR

    @property
    def effective_results_dir(self) -> str:
        if self.is_smoke and self.results_dir == BASE_RESULTS_DIR:
            return SMOKE_RESULTS_DIR
        return self.results_dir

    @property
    def run_dir(self) -> str:
        return os.path.join(self.effective_results_dir, f"seed_{self.seed}", self.arm)

    @property
    def results_file(self) -> str:
        return os.path.join(self.run_dir, "run_results.json")


@dataclass
class RunState:
    """Mutable execution state encapsulating model, optimizer, step, and evaluation logs."""
    spec: RunSpec
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    step: int = 1
    elapsed_sec: float = 0.0
    logs: List[Dict[str, Any]] = field(default_factory=list)
    anchor_logs: List[Dict[str, Any]] = field(default_factory=list)
    sensitivity_logs: Dict[str, Any] = field(default_factory=dict)
    is_complete: bool = False

    def last_process_log(self) -> Optional[Dict[str, Any]]:
        return self.logs[-1] if self.logs else None

    def last_anchor_log(self) -> Optional[Dict[str, Any]]:
        return self.anchor_logs[-1] if self.anchor_logs else None


@dataclass
class StudyTables:
    """Canonical Tidy Data tables representing the unified source of truth for analysis and plotting."""
    process_horizon_records: List[Dict[str, Any]] = field(default_factory=list)
    anchor_horizon_records: List[Dict[str, Any]] = field(default_factory=list)
    anchor_sequence_records: List[Dict[str, Any]] = field(default_factory=list)
    recovery_contrast_records: List[Dict[str, Any]] = field(default_factory=list)
    transition_shock_records: List[Dict[str, Any]] = field(default_factory=list)
    
    # Pre-indexed lookup maps for O(1) query semantics
    _proc_map: Dict[tuple, float] = field(default_factory=dict)
    _anch_map: Dict[tuple, float] = field(default_factory=dict)

    def process_bpc(self, seed: int, arm: str, step: int, horizon: int) -> float:
        return self._proc_map.get((seed, arm, step, int(horizon)), float("nan"))

    def anchor_bpc(self, seed: int, arm: str, step: int, horizon: int) -> float:
        return self._anch_map.get((seed, arm, step, int(horizon)), float("nan"))
