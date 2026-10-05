"""Fail-closed preflight run at the start of every phase (P2 onwards).

Runs the regression tests that guard properties every result depends on. Any failure
stops the phase before a single run starts.
"""

from __future__ import annotations

import subprocess
import sys

from .provenance import ROOT

REGRESSION_TESTS = [
    # Model initialisation must depend on the run's own seed only (P1 seeding bug, fixed in ad61131).
    "tests/test_models_theory_metrics.py::test_train_model_initialisation_depends_only_on_seed",
]


def preflight(phase: str) -> dict:
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *REGRESSION_TESTS]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"preflight failed for {phase}; refusing to start.\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return {"phase": phase, "regression_tests": REGRESSION_TESTS, "passed": True,
            "summary": r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""}
