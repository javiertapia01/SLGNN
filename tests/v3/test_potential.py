"""Cabeza `V`: `U(0) = 0`, monotonía y repulsión por construcción."""

import pytest
import torch

from slgnn_v3 import V3Config
from slgnn_v3.potential import PotentialHead, conservative_force

from .conftest import DTYPE, build_set, make_particles, small_model


def _head(seed=0, hidden=16):
    torch.manual_seed(seed)
    cfg = V3Config()
    cfg.potential.hidden = hidden
    cfg.encoder.hidden = hidden
    return PotentialHead(cfg.potential, cfg.encoder).to(DTYPE), cfg


def test_U_zero_at_zero_compression():
    """`U(0) = 0` con tolerancia 1e-12: el cero energético está en separación."""
    head, cfg = _head()
    h = torch.randn(32, cfg.encoder.hidden, dtype=DTYPE)
    U = head.energy(torch.zeros(32, dtype=DTYPE), h)
    assert float(U.detach().abs().max()) <= 1e-12


def test_dU_ddelta_non_negative():
    """`dU/d(delta) >= -1e-10` para compresiones y contextos muestreados."""
    head, cfg = _head(seed=1)
    delta = torch.linspace(0.0, 0.4, 40, dtype=DTYPE).requires_grad_(True)
    h = torch.randn(40, cfg.encoder.hidden, dtype=DTYPE)
    U = head.energy(delta, h)
    (g,) = torch.autograd.grad(U.sum(), delta)
    assert float(g.min()) >= -1e-10


def test_dU_ddelta_equals_normal_force():
    """El teorema fundamental del cálculo, verificado sobre la cuadratura."""
    head, cfg = _head(seed=2)
    delta = torch.linspace(0.01, 0.3, 20, dtype=DTYPE).requires_grad_(True)
    h = torch.randn(20, cfg.encoder.hidden, dtype=DTYPE)
    U = head.energy(delta, h)
    (g,) = torch.autograd.grad(U.sum(), delta)
    f = head.normal_force(delta.detach(), h)
    assert float((g - f).detach().abs().max()) <= 1e-9


def test_normal_force_non_negative():
    head, cfg = _head(seed=3)
    delta = torch.rand(64, dtype=DTYPE) * 0.5
    h = 5.0 * torch.randn(64, cfg.encoder.hidden, dtype=DTYPE)
    assert float(head.normal_force(delta, h).detach().min()) >= 0.0


def test_force_is_repulsive_on_a_real_pair():
    """Dos esferas solapadas deben empujarse, no atraerse."""
    model = small_model(seed=5)
    q = torch.tensor([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]], dtype=DTYPE)
    pb = make_particles(q, radius=0.5).requires_grad_q()
    cs = build_set(pb, cfg=model.cfg)
    h_node, h_edge, _ = model.encoder(cs, pb)
    h_V = model.proc_V(cs, h_node, h_edge)
    V, _ = model.head_V.total_potential(cs, h_V, torch.ones_like(cs.gap))
    F = conservative_force(V, pb.q, create_graph=False)
    assert float(F[0, 0].detach()) < 0        # i empujada hacia -x
    assert float(F[1, 0].detach()) > 0        # j empujada hacia +x
    assert float((F[0] + F[1]).detach().abs().max()) <= 1e-12   # acción-reacción


def test_no_force_when_separated():
    """Vuelo libre: `delta = 0` exacto, luego fuerza exactamente nula.

    Esta es la prueba que `softplus(-g)` del legacy no pasa.
    """
    model = small_model(seed=6)
    q = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=DTYPE)
    pb = make_particles(q, radius=0.5).requires_grad_q()
    cs = build_set(pb, cfg=model.cfg)
    assert cs.n_contacts == 1 and float(cs.delta.detach()) == 0.0
    h_node, h_edge, _ = model.encoder(cs, pb)
    V, _ = model.head_V.total_potential(cs, model.proc_V(cs, h_node, h_edge),
                                        torch.ones_like(cs.gap))
    F = conservative_force(V, pb.q, create_graph=False)
    assert float(F.detach().abs().max()) == 0.0


def test_potential_receives_no_velocity():
    """`V` no puede depender de `v` ni de `omega`: si dependiera, no sería
    conservativo. Se comprueba cambiando las velocidades y midiendo `V`."""
    model = small_model(seed=7)
    q = torch.tensor([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]], dtype=DTYPE)
    outs = []
    for scale in (0.0, 5.0):
        pb = make_particles(q, v=scale * torch.ones(2, 3, dtype=DTYPE),
                            omega=scale * torch.ones(2, 3, dtype=DTYPE), radius=0.5)
        cs = build_set(pb, cfg=model.cfg)
        h_node, h_edge, _ = model.encoder(cs, pb)
        V, _ = model.head_V.total_potential(cs, model.proc_V(cs, h_node, h_edge),
                                            torch.ones_like(cs.gap))
        outs.append(float(V.detach()))
    assert outs[0] == pytest.approx(outs[1], abs=1e-14)


def test_conservative_force_requires_grad_q():
    model = small_model(seed=8)
    q = torch.tensor([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]], dtype=DTYPE)
    pb = make_particles(q, radius=0.5)
    cs = build_set(pb, cfg=model.cfg)
    h_node, h_edge, _ = model.encoder(cs, pb)
    V, _ = model.head_V.total_potential(cs, model.proc_V(cs, h_node, h_edge),
                                        torch.ones_like(cs.gap))
    with pytest.raises(RuntimeError, match="hoja diferenciable"):
        conservative_force(V, pb.q, create_graph=False)


def test_finite_differences_vs_autograd_force():
    """Diferencias finitas contra autograd, error relativo <= 1e-5."""
    model = small_model(seed=9)
    q0 = torch.tensor([[0.0, 0.0, 0.0], [0.88, 0.1, 0.0]], dtype=DTYPE)

    def energy(qq):
        pb = make_particles(qq, radius=0.5)
        cs = build_set(pb, cfg=model.cfg)
        h_node, h_edge, _ = model.encoder(cs, pb)
        return model.head_V.total_potential(
            cs, model.proc_V(cs, h_node, h_edge), torch.ones_like(cs.gap)
        )[0]

    pb = make_particles(q0, radius=0.5).requires_grad_q()
    cs = build_set(pb, cfg=model.cfg)
    h_node, h_edge, _ = model.encoder(cs, pb)
    V, _ = model.head_V.total_potential(cs, model.proc_V(cs, h_node, h_edge),
                                        torch.ones_like(cs.gap))
    F = conservative_force(V, pb.q, create_graph=False)

    h = 1e-6
    fd = torch.zeros_like(q0)
    for i in range(2):
        for k in range(3):
            dq = torch.zeros_like(q0)
            dq[i, k] = h
            fd[i, k] = -(energy(q0 + dq) - energy(q0 - dq)).detach() / (2 * h)
    assert float((F - fd).detach().norm() / (fd.norm() + 1e-30)) <= 1e-5
