import math

import pytest
import torch

from vjepa_ac.schedule import cosine_warmup_factor, make_scheduler


def test_warmup_ramp():
    assert cosine_warmup_factor(0, 10, 100) == 0.0
    assert cosine_warmup_factor(5, 10, 100) == pytest.approx(0.5)
    assert cosine_warmup_factor(10, 10, 100) == pytest.approx(1.0)


def test_cosine_decay():
    assert cosine_warmup_factor(55, 10, 100) == pytest.approx(0.5)
    assert cosine_warmup_factor(100, 10, 100) == pytest.approx(0.0, abs=1e-12)
    mid = 0.5 * (1 + math.cos(math.pi * 0.25))
    assert cosine_warmup_factor(32.5, 10, 100) == pytest.approx(mid)


def test_monotonic_after_warmup():
    vals = [cosine_warmup_factor(s, 10, 100) for s in range(10, 101)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))


def test_scheduler_applies_factor():
    lin = torch.nn.Linear(2, 2)
    optim = torch.optim.AdamW(lin.parameters(), lr=1e-3)
    sched = make_scheduler(optim, warmup_steps=4, total_steps=8)
    assert sched.get_last_lr()[0] == pytest.approx(0.0)
    for _ in range(4):
        optim.step()
        sched.step()
    assert sched.get_last_lr()[0] == pytest.approx(1e-3)
