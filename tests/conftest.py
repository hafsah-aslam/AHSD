"""Every test starts from a fixed, logged seed (PYTEST_SEED, default 0).

Each invocation appends its seed to logs/pytest_seeds.log so any failure can be
replayed with `PYTEST_SEED=<seed> python -m pytest`.
"""

import datetime as dt
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch

SEED = int(os.environ.get("PYTEST_SEED", "0"))


def pytest_sessionstart(session):
    log = Path(__file__).resolve().parents[1] / "logs" / "pytest_seeds.log"
    log.parent.mkdir(exist_ok=True)
    with open(log, "a") as f:
        f.write(f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} PYTEST_SEED={SEED}\n")


def pytest_report_header(config):
    return f"PYTEST_SEED={SEED}"


@pytest.fixture(autouse=True)
def _seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    yield
