"""Integrador híbrido: ecuación central, vuelo libre, gravedad y no-doble-conteo.

Cubre el Nivel 0 de §16.2: las pruebas analíticas que deben pasar **antes** de
entrenar nada.
"""

import pytest
import torch

from slgnn_v3 import ParticleBatch, RouterProfile, V3State, box_surfaces, half_space
from slgnn_v3.integrator import advance_position, free_velocity, gravity_force

from .conftest import DTYPE, default_box, make_particles, small_model


def test_free_particle_no_gravity_is_exact():
    """Nivel 0.1: sin contacto ni gravedad, el paso es traslación uniforme."""
    model = small_model(seed=1)
    v0 = torch.tensor([[0.3, -0.2, 0.1]], dtype=DTYPE)
    pb = make_particles([[2.5, 2.5, 2.5]], v=v0, radius=0.5)
    res = model.step(V3State(pb, time=0.0), 0.1, default_box())
    assert float(res.delta_p.detach().abs().max()) == 0.0
    assert float(res.delta_L.detach().abs().max()) == 0.0
    assert torch.allclose(res.next_state.particles.v, v0)
    assert torch.allclose(res.next_state.particles.q, pb.q + 0.1 * v0, atol=1e-14)


def test_free_particle_with_gravity_is_exact():
    """Nivel 0.2: `Delta p = dt m g` exactamente, sin aprender nada."""
    model = small_model(seed=2)
    g = torch.tensor([0.0, -0.98, 0.0], dtype=DTYPE)
    pb = make_particles([[2.5, 2.5, 2.5]], v=[[0.0, 0.5, 0.0]], radius=0.5, mass=2.0)
    dt = 0.1
    res = model.step(V3State(pb, time=0.0), dt, default_box(), g)
    expected = dt * pb.mass.unsqueeze(-1) * g
    assert torch.allclose(res.delta_p, expected, atol=1e-14)
    assert float(res.delta_L.detach().abs().max()) == 0.0


def test_gravity_counted_exactly_once():
    """D-006: la gravedad es fuerza externa; `V` no la contiene."""
    model = small_model(seed=3)
    g = torch.tensor([0.0, -0.98, 0.0], dtype=DTYPE)
    pb = make_particles([[2.5, 2.5, 2.5]], radius=0.5)
    res = model.step(V3State(pb, time=0.0), 0.1, default_box(), g)
    assert res.diagnostics.energies["V_gravity"] == "disabled"
    assert torch.allclose(res.forces, gravity_force(pb, g), atol=1e-14)


def test_central_discrete_equation_holds():
    """`M (nu_{k+1} - nu_k) = dt F_reg + J^T Lambda`, exactamente."""
    for profile in (RouterProfile.COMPLIANT, RouterProfile.IMPULSIVE):
        model = small_model(profile=profile, seed=4)
        pb = make_particles(
            [[2.0, 2.0, 0.45], [2.0, 2.0, 1.4], [2.6, 2.0, 1.9]],
            v=[[0.1, 0, -0.4], [0, 0.2, -0.5], [-0.3, 0, 0.1]],
            omega=[[0.2, 0, 0], [0, 0.1, 0], [0, 0, 0.3]], radius=0.5,
        )
        g = torch.tensor([0.0, 0.0, -0.98], dtype=DTYPE)
        dt = 0.05
        res = model.step(V3State(pb, time=0.1), dt, default_box(), g)
        p = res.next_state.particles
        lhs_p = pb.mass.unsqueeze(-1) * (p.v - pb.v)
        lhs_L = pb.inertia.unsqueeze(-1) * (p.omega - pb.omega)
        assert float((lhs_p - res.delta_p).detach().abs().max()) <= 1e-14
        assert float((lhs_L - res.delta_L).detach().abs().max()) <= 1e-14


def test_position_uses_post_impulse_velocity():
    model = small_model(profile=RouterProfile.IMPULSIVE, seed=5)
    pb = make_particles([[2.0, 2.0, 0.45]], v=[[0.0, 0.0, -2.0]], radius=0.5)
    dt = 0.05
    res = model.step(V3State(pb, time=0.0), dt, default_box())
    p = res.next_state.particles
    assert torch.allclose(p.q, pb.q + dt * p.v, atol=1e-14)
    assert float(res.impulses.numel()) > 0


def test_time_advances_through_rollout():
    """§22.12: fijar `t = 0` durante el rollout está prohibido."""
    model = small_model(seed=6)
    pb = make_particles([[2.5, 2.5, 2.5]], v=[[0.1, 0.0, 0.0]], radius=0.5)
    outs = model.rollout(V3State(pb, time=0.0), 0.05, default_box(), 4,
                         eval_mode=True, create_graph=False)
    assert [round(o.next_state.time, 10) for o in outs] == [0.05, 0.10, 0.15, 0.20]


def test_three_sphere_chain_simultaneous_contacts():
    """Nivel 0.5: cadena de tres partículas, dos contactos simultáneos, un
    único solve acoplado."""
    model = small_model(profile=RouterProfile.IMPULSIVE, seed=7)
    q = torch.tensor([[1.0, 2.5, 2.5], [1.98, 2.5, 2.5], [2.96, 2.5, 2.5]], dtype=DTYPE)
    pb = make_particles(q, v=[[2.0, 0, 0], [0, 0, 0], [0, 0, 0]], radius=0.5)
    res = model.step(V3State(pb, time=0.0), 0.05, default_box())
    assert res.diagnostics.router["n_impulsive"] == 2
    assert res.diagnostics.solver["solver_component_size_max"] == 2.0
    assert res.diagnostics.balance["internal_linear_momentum_error"] <= 1e-12


def test_batched_systems_do_not_interact():
    """Nivel 0.8: dos sistemas geométricamente próximos, cero aristas cruzadas.

    El resultado de cada sistema debe ser idéntico al de correrlo solo.
    """
    model = small_model(seed=8)
    a = make_particles([[2.0, 2.0, 0.45], [2.0, 2.0, 1.4]],
                       v=[[0, 0, -0.3], [0, 0, -0.1]], radius=0.5)
    b = make_particles([[2.05, 2.0, 0.45], [2.05, 2.0, 1.4]],
                       v=[[0, 0, -0.3], [0, 0, -0.1]], radius=0.5)
    box = default_box()
    g = torch.tensor([0.0, 0.0, -0.98], dtype=DTYPE)

    model.reset_lifecycle()
    solo_a = model.step(V3State(a, time=0.0), 0.05, box, g)
    model.reset_lifecycle()
    both = ParticleBatch.concat([a, b])
    joint = model.step(V3State(both, time=0.0), 0.05, box, g)
    assert float((joint.delta_p[:2] - solo_a.delta_p).detach().abs().max()) <= 1e-12


def test_double_backward_through_the_step():
    """Doble backward: gradientes finitos y no nulos en parámetros activos."""
    model = small_model(seed=9)
    pb = make_particles([[2.0, 2.0, 0.44], [2.0, 2.0, 1.36]],
                        v=[[0, 0, -0.5], [0, 0, 0.2]], radius=0.5)
    res = model.step(V3State(pb, time=0.0), 0.05, default_box(), create_graph=True)
    params = [p for p in model.head_V.parameters()]
    grads = torch.autograd.grad(res.delta_p.pow(2).sum(), params, create_graph=True)
    second = torch.autograd.grad(sum(g.pow(2).sum() for g in grads), params,
                                 allow_unused=True)
    assert all(g is None or torch.isfinite(g).all() for g in second)
    assert any(g is not None and float(g.abs().sum()) > 0 for g in second)


def test_step_result_is_not_just_acceleration():
    """§22.2: v3 no expone un decoder directo de aceleración."""
    model = small_model(seed=10)
    pb = make_particles([[2.0, 2.0, 0.45]], v=[[0, 0, -0.5]], radius=0.5)
    res = model.step(V3State(pb, time=0.0), 0.05, default_box())
    for field in ("delta_p_regular", "delta_p_impulse", "delta_L_regular",
                  "delta_L_impulse", "forces", "torques", "impulses"):
        assert hasattr(res, field)
    a, alpha = res.regular_accelerations(pb.mass, pb.inertia)
    assert a.shape == (1, 3) and alpha.shape == (1, 3)
