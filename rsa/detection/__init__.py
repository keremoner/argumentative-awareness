"""
Three-score sequential detection framework for epistemic vigilance.

Mathematical reference: ``three_scores.pdf`` / ``detection_implementation_spec.md``.

This package is intentionally isolated from ``rsa.experimental`` -- it shares
no code with the existing ``SuspicionSwitchListener`` / ``DiscrepancySwitchListener``
and provides a clean primitive-level implementation of the three scores
(``surp2``, ``surp1``, ``sus``) together with a decoupled sequential test
harness.
"""

from .scores import (
    ScoreContext,
    compute_surp2,
    compute_surp1,
    compute_sus,
    make_sus_variant,
    SUS_VARIANTS,
    SUS_VARIANT_FNS,
    SCORE_FNS,
)
from .sequential_test import SequentialTest
from .listener import DetectionListener, SWITCH_TYPES
from .replay import (
    Trajectory,
    TableSpeaker,
    replay_listener,
    splice,
    tau_for_rule,
    regenerate_tables,
    switching_trajectory,
)

__all__ = [
    "ScoreContext",
    "compute_surp2",
    "compute_surp1",
    "compute_sus",
    "make_sus_variant",
    "SUS_VARIANTS",
    "SUS_VARIANT_FNS",
    "SCORE_FNS",
    "SequentialTest",
    "DetectionListener",
    "SWITCH_TYPES",
    "Trajectory",
    "TableSpeaker",
    "replay_listener",
    "splice",
    "tau_for_rule",
    "regenerate_tables",
    "switching_trajectory",
]
