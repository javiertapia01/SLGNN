"""Cinemática de contacto: punto común, brazos, convención de signo, ventanas."""

import pytest
import torch

from slgnn_v3 import V3Config
from slgnn_v3.contact_kinematics import build_contacts
from slgnn_v3.graph import build_candidate_graph
from slgnn_v3.smoothing import compression, positive_part_c2, quintic_window

from .conftest import DTYPE, build_set, default_box, make_particles, random_particles


def _pair(qa, qb, Ra, Rb, va=None, vb=None, wa=None, wb=None):
    q = torch.tensor([qa, qb], dtype=DTYPE)
    v = torch.tensor([va or [0, 0, 0], vb or [0, 0, 0]], dtype=DTYPE)
    w = torch.tensor([wa or [0, 0, 0], wb or [0, 0, 0]], dtype=DTYPE)
    return make_particles(q, v, w, radius=torch.tensor([Ra, Rb], dtype=DTYPE))


def test_common_point_swap_invariance():
    """§18.3: radios desiguales usan el punto común correcto, invariante al
    intercambio `i <-> j`, a `atol <= 1e-12`."""
    cfg = V3Config()
    pb = _pair([0.0, 0.0, 0.0], [1.3, 0.4, -0.2], 0.9, 0.3)
    cs = build_contacts(pb, torch.tensor([[0, 1]]), None, cfg)
    pb_swap = _pair([1.3, 0.4, -0.2], [0.0, 0.0, 0.0], 0.3, 0.9)
    cs_swap = build_contacts(pb_swap, torch.tensor([[0, 1]]), None, cfg)
    assert torch.allclose(cs.x_c, cs_swap.x_c, atol=1e-12)
    # las normales se invierten y los brazos se intercambian
    assert torch.allclose(cs.n, -cs_swap.n, atol=1e-12)
    assert torch.allclose(cs.r_i, cs_swap.r_j, atol=1e-12)
    assert torch.allclose(cs.r_j, cs_swap.r_i, atol=1e-12)


def test_common_point_is_midpoint_between_surfaces():
    pb = _pair([0.0, 0.0, 0.0], [1.2, 0.0, 0.0], 0.8, 0.5)
    cs = build_contacts(pb, torch.tensor([[0, 1]]), None, V3Config())
    expected = 0.5 * ((0.0 + 0.8) + (1.2 - 0.5))
    assert float(cs.x_c[0, 0]) == pytest.approx(expected, abs=1e-14)


def test_arms_difference_is_exact():
    """`r_i - r_j = q_j - q_i` exactamente: de ahí sale la conservación."""
    pb = random_particles(6, seed=7, box=(0.6, 2.0))
    cs = build_set(pb)
    pp = ~cs.is_wall
    lhs = (cs.r_i - cs.r_j)[pp]
    rhs = (pb.q[cs.j[pp]] - pb.q[cs.i[pp]])
    assert float((lhs - rhs).abs().max()) == 0.0


def test_relative_velocity_sign_convention():
    """`u_n < 0` significa aproximación, en par y en pared."""
    pb = _pair([0.0, 0.0, 0.0], [1.2, 0.0, 0.0], 0.5, 0.5,
               va=[1.0, 0, 0], vb=[0.0, 0, 0])
    cs = build_contacts(pb, torch.tensor([[0, 1]]), None, V3Config())
    assert float(cs.u_n) < 0

    box = default_box()
    approaching = make_particles([[2.0, 2.0, 0.55]], v=[[0.0, 0.0, -1.0]], radius=0.5)
    cs_w = build_set(approaching, box)
    floor = cs_w.surface == 4
    assert float(cs_w.u_n[floor]) < 0


def test_spin_enters_contact_velocity():
    """Sin spin en la velocidad de contacto no hay fricción posible."""
    pb = _pair([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.5, 0.5,
               wa=[0.0, 0.0, 1.0])
    cs = build_contacts(pb, torch.tensor([[0, 1]]), None, V3Config())
    assert float(cs.u_tau.norm()) > 1e-9


def test_wall_arm_points_to_the_wall():
    box = default_box()
    pb = make_particles([[2.0, 2.0, 0.5]], radius=0.5)
    cs = build_set(pb, box)
    floor = (cs.surface == 4).nonzero().flatten()
    assert torch.allclose(cs.r_i[floor], torch.tensor([[0.0, 0.0, -0.5]], dtype=DTYPE))


# --- suavizados -----------------------------------------------------------

def test_positive_part_exactly_zero_in_separation():
    """§18.1: compresión exactamente cero para `g >= 0`, no `log(2)/beta`."""
    g = torch.tensor([1.0, 0.5, 1e-12, 0.0], dtype=DTYPE)
    delta = compression(g, eps=0.02)
    assert float(delta.abs().max()) == 0.0


def test_positive_part_c2_continuity():
    eps = 0.02
    for x0 in (0.0, eps):
        x = torch.tensor([x0], dtype=DTYPE, requires_grad=True)
        y = positive_part_c2(x, eps)
        (d1,) = torch.autograd.grad(y.sum(), x, create_graph=True)
        (d2,) = torch.autograd.grad(d1.sum(), x)
        h = 1e-7
        left = positive_part_c2(torch.tensor([x0 - h], dtype=DTYPE), eps)
        right = positive_part_c2(torch.tensor([x0 + h], dtype=DTYPE), eps)
        assert float((right - left).abs()) < 1e-6          # continua
        if x0 == 0.0:
            assert float(d1.detach().abs()) < 1e-12 and float(d2.detach().abs()) < 1e-12
        else:
            assert float(d1.detach()) == pytest.approx(1.0, abs=1e-10)
            assert float(d2.detach().abs()) < 1e-9


def test_positive_part_is_identity_above_eps():
    x = torch.tensor([0.05, 1.0], dtype=DTYPE)
    assert torch.allclose(positive_part_c2(x, 0.02), x)


def test_quintic_window_endpoints():
    x = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0], dtype=DTYPE)
    w = quintic_window(x, 0.0, 1.0)
    assert float(w[0]) == 1.0 and float(w[1]) == 1.0
    assert float(w[3]) == 0.0 and float(w[4]) == 0.0
    assert 0.0 < float(w[2]) < 1.0


def test_activation_is_one_in_contact_zero_when_separated():
    cfg = V3Config()
    pb = _pair([0.0, 0.0, 0.0], [0.9, 0.0, 0.0], 0.5, 0.5)     # gap = -0.1
    cs = build_contacts(pb, torch.tensor([[0, 1]]), None, cfg)
    assert float(cs.activation) == pytest.approx(1.0)
    pb2 = _pair([0.0, 0.0, 0.0], [1.2, 0.0, 0.0], 0.5, 0.5)    # gap = +0.2
    cs2 = build_contacts(pb2, torch.tensor([[0, 1]]), None, cfg)
    assert float(cs2.activation) == 0.0


def test_keys_are_canonical_and_stable():
    pb = random_particles(6, seed=11, box=(0.6, 2.0))
    cs = build_set(pb, default_box())
    keys = cs.keys()
    pp = ~cs.is_wall
    assert (keys[pp, 1] < keys[pp, 2]).all()
    assert (keys[cs.is_wall, 3] == 1).all()
    assert (keys[pp, 3] == 0).all()
