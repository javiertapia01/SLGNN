"""Pruebas unitarias de los contratos puros de S03-R1."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "slgnn_v2"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from recover_S03_retention import (  # noqa: E402
    interpolate_states,
    pareto_names,
    replace_prefixes,
)


def test_interpolation_preserves_endpoints_and_midpoint():
    left = {"weight": torch.tensor([0.0, 2.0])}
    right = {"weight": torch.tensor([2.0, 6.0])}
    assert torch.equal(interpolate_states(left, right, 0.0)["weight"], left["weight"])
    assert torch.equal(interpolate_states(left, right, 1.0)["weight"], right["weight"])
    assert torch.equal(
        interpolate_states(left, right, 0.5)["weight"],
        torch.tensor([1.0, 4.0]),
    )


def test_replace_prefixes_only_changes_requested_modules():
    base = {
        "head_pp_V.weight": torch.tensor([1.0]),
        "head_pw_V.weight": torch.tensor([2.0]),
        "proc_V.weight": torch.tensor([3.0]),
    }
    donor = {key: value + 10.0 for key, value in base.items()}
    result = replace_prefixes(base, donor, ("head_pp_",))
    assert result["head_pp_V.weight"].item() == 11.0
    assert result["head_pw_V.weight"].item() == 2.0
    assert result["proc_V.weight"].item() == 3.0
    assert base["head_pp_V.weight"].item() == 1.0


def test_pareto_names_excludes_dominated_candidate():
    records = [
        {"name": "a", "n60_score": 1.0, "retention": {"gate_score": 2.0}},
        {"name": "b", "n60_score": 2.0, "retention": {"gate_score": 1.0}},
        {"name": "dominated", "n60_score": 3.0, "retention": {"gate_score": 3.0}},
    ]
    assert set(pareto_names(records)) == {"a", "b"}
