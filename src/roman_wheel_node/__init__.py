"""Roman Wheel Node: exact-copy gate + learning, append-only Bayesian source weighting.

See :mod:`roman_wheel_node.core` for the full documentation.
"""
from .core import (BAYES_VARIANTS, LiveWeights, Result, ThresholdLearner, clamp01, give_feedback,
                   opcode, opcode_learned, roman_wheel_node)
from .dependence import DependenceLearner, FrozenDependence
from .trace import Claim, ClaimTrace, Decision, TamperError, replay
from .policy import Policy, PolicyError, load_policy

__version__ = "1.2.0"
__all__ = ["roman_wheel_node", "LiveWeights", "ThresholdLearner", "give_feedback", "clamp01",
           "opcode", "opcode_learned", "Result", "BAYES_VARIANTS",
           "DependenceLearner", "FrozenDependence", "ClaimTrace", "Claim", "Decision", "TamperError", "replay", "Policy", "PolicyError", "load_policy", "__version__"]
# shadow mode: roman_wheel_node.shadow.ShadowRunner (imported lazily; it is optional for the core node)
