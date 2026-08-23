"""Grafo candidato: aristas no ordenadas, batching seguro y CCD."""

import torch

from slgnn_v3 import ParticleBatch, V3Config
from slgnn_v3.graph import (
    assert_no_cross_batch,
    build_candidate_graph,
    candidate_pairs,
    free_positions,
)

from .conftest import DTYPE, make_particles, random_particles


def test_pairs_are_unordered_and_unique():
    pb = random_particles(10, seed=3, box=(0.6, 2.4))
    e = build_candidate_graph(pb, 0.35, 0.15)
    assert (e[:, 0] < e[:, 1]).all()
    seen = {tuple(row) for row in e.tolist()}
    assert len(seen) == e.shape[0]


def test_no_cross_batch_edges_even_when_geometrically_close():
    """Dos sistemas superpuestos en coordenadas: ni una arista entre ellos."""
    a = make_particles([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]])
    b = make_particles([[0.05, 0.0, 0.0], [0.95, 0.0, 0.0]])
    both = ParticleBatch.concat([a, b])
    e = build_candidate_graph(both, 0.35, 0.15)
    assert_no_cross_batch(e, both.batch_id)
    assert both.batch_id[e[:, 0]].tolist() == both.batch_id[e[:, 1]].tolist()
    # exactamente una arista por sistema, ninguna cruzada
    assert e.shape[0] == 2


def test_cutoff_respects_gap_not_distance():
    """Con radios desiguales, el corte se mide sobre el gap."""
    pb = make_particles([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                        radius=torch.tensor([1.0, 1.8], dtype=DTYPE))
    # gap = 3 - 2.8 = 0.2
    assert candidate_pairs(pb.q, pb.radius, pb.batch_id, 0.25).shape[0] == 1
    assert candidate_pairs(pb.q, pb.radius, pb.batch_id, 0.10).shape[0] == 0


def test_ccd_adds_predicted_crossing():
    """Una partícula rápida que cruzaría entre snapshots entra al grafo."""
    q = torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=DTYPE)
    pb = make_particles(q, v=[[40.0, 0.0, 0.0], [0.0, 0.0, 0.0]], radius=0.5)
    assert build_candidate_graph(pb, 0.35, 0.15).shape[0] == 0
    q_free = free_positions(pb.q, pb.v, dt=0.1)
    e = build_candidate_graph(pb, 0.35, 0.15, q_free=q_free, ccd_margin=0.0)
    assert e.shape[0] == 1


def test_empty_graph_is_valid():
    pb = make_particles([[0.0, 0.0, 0.0]])
    e = build_candidate_graph(pb, 0.35, 0.15)
    assert e.shape == (0, 2)
    assert_no_cross_batch(e, pb.batch_id)
