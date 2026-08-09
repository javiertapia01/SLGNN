"""Cero fuga entre ejemplos del batch, para v3 y para GNS.

El chequeo es **exacto** (§18): correr un sistema solo y correrlo dentro de un
batch con otro sistema geométricamente superpuesto debe dar bit a bit lo mismo,
salvo el orden de reducción de las sumas dispersas.
"""

import pytest
import torch

from gns_baseline import GNSConfig, GNSControlled
from slgnn_experiments.sampling import StratifiedSampler, TransitionIndex
from slgnn_v3 import ParticleBatch, RouterProfile, SLGNNv3, V3Config, V3State, box_surfaces
from slgnn_v3.graph import assert_no_cross_batch, build_candidate_graph

DTYPE = torch.float64


def _pb(offset):
    q = torch.tensor([[2.0, 2.0, 0.45], [2.0, 2.0, 1.4]], dtype=DTYPE) + offset
    return ParticleBatch.from_arrays(
        q=q, v=torch.tensor([[0.1, 0, -0.3], [0, 0.1, -0.2]], dtype=DTYPE),
        omega=torch.tensor([[0.2, 0, 0], [0, 0.1, 0]], dtype=DTYPE),
        mass=torch.ones(2, dtype=DTYPE), radius=torch.full((2,), 0.5, dtype=DTYPE),
    )


def _small_v3(profile):
    torch.manual_seed(0)
    cfg = V3Config()
    cfg.encoder.hidden = cfg.potential.hidden = 16
    cfg.dissipation.hidden = cfg.impact.hidden = 16
    cfg.router.profile = profile
    return SLGNNv3(cfg).to(DTYPE)


def test_graph_never_crosses_batches_even_when_overlapping():
    a, b = _pb(torch.zeros(3, dtype=DTYPE)), _pb(torch.tensor([0.02, 0.0, 0.0]))
    both = ParticleBatch.concat([a, b])
    edges = build_candidate_graph(both, 0.35, 0.15)
    assert_no_cross_batch(edges, both.batch_id)
    # con dos sistemas de dos partículas cada uno solo caben 2 aristas internas
    assert edges.shape[0] == 2


@pytest.mark.parametrize("profile", [RouterProfile.COMPLIANT, RouterProfile.IMPULSIVE])
def test_v3_batched_equals_solo(profile):
    model = _small_v3(profile)
    box = box_surfaces([0, 0, 0], [5, 5, 5])
    g = torch.tensor([0.0, 0.0, -0.98], dtype=DTYPE)
    a, b = _pb(torch.zeros(3, dtype=DTYPE)), _pb(torch.tensor([0.02, 0.0, 0.0]))

    model.reset_lifecycle()
    solo = model.step(V3State(a, time=0.0), 0.05, box, g)
    model.reset_lifecycle()
    joint = model.step(V3State(ParticleBatch.concat([a, b]), time=0.0), 0.05, box, g)
    assert float((joint.delta_p[:2] - solo.delta_p).detach().abs().max()) <= 1e-13
    assert float((joint.delta_L[:2] - solo.delta_L).detach().abs().max()) <= 1e-13


def test_gns_batched_equals_solo():
    torch.manual_seed(0)
    model = GNSControlled(GNSConfig(hidden=16, n_message_steps=2)).to(DTYPE)
    box = box_surfaces([0, 0, 0], [5, 5, 5])
    g = torch.tensor([0.0, 0.0, -0.98], dtype=DTYPE)
    a, b = _pb(torch.zeros(3, dtype=DTYPE)), _pb(torch.tensor([0.02, 0.0, 0.0]))
    solo = model.step(V3State(a, time=0.0), 0.05, box, g)
    joint = model.step(V3State(ParticleBatch.concat([a, b]), time=0.0), 0.05, box, g)
    assert float((joint.delta_p[:2] - solo.delta_p).detach().abs().max()) <= 1e-13


def test_sampler_composition_is_recorded():
    """La composición real de cada época se registra, no se supone."""
    index = TransitionIndex(
        items=[(0, k) for k in range(10)],
        strata=["free"] * 6 + ["pw"] * 4,
        by_stratum={"free": list(range(6)), "pw": list(range(6, 10))},
    )
    s = StratifiedSampler(index, {"free": 0.5, "pw": 0.5, "mixed": 0.5}, seed=0)
    assert s.dropped_strata == ["mixed"]
    assert sum(s.effective_quotas.values()) == pytest.approx(1.0)
    picks = s.sample(8)
    assert len(picks) == 8
    comp = s.epoch_composition(8, 10)
    assert sum(comp.values()) == pytest.approx(80, abs=2)
