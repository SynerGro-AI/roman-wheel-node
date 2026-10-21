"""Reproducible synthetic benchmark for Roman Wheel Node.

    python -m roman_wheel_node.bench --seeds 20            # full run -> results/
    python -m roman_wheel_node.bench --seeds 20 --check results   # rerun and byte-compare
    python -m roman_wheel_node.bench.generator --list      # the exact claim-stream generator
"""
from .generator import SCENARIOS, CopyGroup, Scenario, Source, stream
from .runner import POLICIES, run, run_one, to_json, to_text

__all__ = ["SCENARIOS", "Scenario", "Source", "CopyGroup", "stream", "POLICIES", "run", "run_one", "to_json", "to_text"]
