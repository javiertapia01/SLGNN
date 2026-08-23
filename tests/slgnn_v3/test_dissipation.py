"""Cabeza `Psi`: convexidad, `psi(0) = 0` y pasividad."""

import pytest
import torch

from slgnn_v3 import RouterProfile, V3Config, V3State
from slgnn_v3.config import DissipationConfig
from slgnn_v3.contact_operator import JT_times_contact_vector
from slgnn_v3.dissipation import DissipationHead
from slgnn_v3.encoder import dissipation_context_features
from slgnn_v3.surfaces import SurfaceSet

from .conftest import DTYPE, build_set, default_box, make_particles, small_model


def _head(seed=0, hidden=16):
    torch.manual_seed(seed)
    cfg = V3Config()
    cfg.dissipation.hidden = hidden
    cfg.encoder.hidden = hidden
    return DissipationHead(cfg.dissipation, cfg.encoder).to(DTYPE), cfg


def test_psi_zero_at_zero():
    head, cfg = _head()
    c1, c2 = head.coefficients(torch.randn(32, cfg.encoder.hidden, dtype=DTYPE))
    psi = head.psi(torch.zeros(32, dtype=DTYPE), c1, c2)
    assert float(psi.detach().abs().max()) <= 1e-12


def test_coefficients_non_negative():
    head, cfg = _head(seed=1)
    c1, c2 = head.coefficients(8.0 * torch.randn(256, cfg.encoder.hidden, dtype=DTYPE))
    assert float(c1.detach().min()) >= 0.0 and float(c2.detach().min()) >= 0.0


def test_psi_first_and_second_derivative_non_negative():
    """`psi' >= -1e-10` y `psi'' >= -1e-10` por autograd."""
    head, cfg = _head(seed=2)
    s = torch.linspace(0.0, 3.0, 64, dtype=DTYPE).requires_grad_(True)
    c1, c2 = head.coefficients(torch.randn(64, cfg.encoder.hidden, dtype=DTYPE))
    psi = head.psi(s, c1, c2)
    (d1,) = torch.autograd.grad(psi.sum(), s, create_graph=True)
    (d2,) = torch.autograd.grad(d1.sum(), s)
    assert float(d1.detach().min()) >= -1e-10
    assert float(d2.detach().min()) >= -1e-10


def test_d_equals_psi_prime():
    head, cfg = _head(seed=3)
    s = torch.linspace(0.1, 2.0, 32, dtype=DTYPE).requires_grad_(True)
    c1, c2 = head.coefficients(torch.randn(32, cfg.encoder.hidden, dtype=DTYPE))
    (d1,) = torch.autograd.grad(head.psi(s, c1, c2).sum(), s)
    assert torch.allclose(d1, head.d(s.detach(), c1, c2), atol=1e-12)


def test_relative_power_is_non_positive():
    """`sum_alpha lambda^Psi . u <= 1e-10`: pasividad medida, no penalizada."""
    model = small_model(seed=4)
    torch.manual_seed(11)
    q = torch.tensor([[0.0, 0.0, 0.0], [0.92, 0.0, 0.0]], dtype=DTYPE)
    for sign in (-1.0, 1.0):
        pb = make_particles(q, v=[[sign * 1.5, 0.2, 0.0], [0.0, 0.0, 0.0]],
                            omega=[[0.3, 0.0, 0.1], [0.0, -0.2, 0.0]], radius=0.5)
        cs = build_set(pb, cfg=model.cfg)
        h_node, h_edge, _ = model.encoder(cs, pb)
        context = dissipation_context_features(cs)
        h_psi = model.proc_Psi(cs, h_node, h_edge, context)
        lam, _, diag = model.head_Psi(cs, h_psi, torch.ones_like(cs.gap) * cs.activation)
        assert float((lam * cs.u).detach().sum()) <= 1e-10


def test_dissipation_inactive_when_separated():
    """Sin contacto no hay amortiguamiento: la activación unilateral es cero."""
    model = small_model(seed=5)
    q = torch.tensor([[0.0, 0.0, 0.0], [1.25, 0.0, 0.0]], dtype=DTYPE)
    pb = make_particles(q, v=[[3.0, 0.0, 0.0], [0.0, 0.0, 0.0]], radius=0.5)
    cs = build_set(pb, cfg=model.cfg)
    assert float(cs.activation.max()) == 0.0
    h_node, h_edge, _ = model.encoder(cs, pb)
    lam, psi, _ = model.head_Psi(
        cs, model.proc_Psi(cs, h_node, h_edge, dissipation_context_features(cs)),
        torch.ones_like(cs.gap) * cs.activation,
    )
    assert float(lam.detach().abs().max()) == 0.0
    assert float(psi.detach()) == 0.0


def test_damping_opposes_approach():
    """Durante el acercamiento el impulso disipativo separa a las partículas."""
    model = small_model(seed=6)
    q = torch.tensor([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]], dtype=DTYPE)
    pb = make_particles(q, v=[[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]], radius=0.5)
    cs = build_set(pb, cfg=model.cfg)
    h_node, h_edge, _ = model.encoder(cs, pb)
    lam, _, _ = model.head_Psi(
        cs, model.proc_Psi(cs, h_node, h_edge, dissipation_context_features(cs)),
        torch.ones_like(cs.gap) * cs.activation,
    )
    assert float((lam * cs.n).detach().sum()) > 0     # a lo largo de n, de i hacia j


def _tangential_model(seed=7):
    return small_model(
        profile=RouterProfile.COMPLIANT,
        seed=seed,
        dissipation=DissipationConfig(
            tangential=True, state_independent_coefficients=True
        ),
    )


def _head_output(model, pb):
    cs = build_set(pb, cfg=model.cfg)
    h_node, h_edge, _ = model.encoder(cs, pb)
    h_psi = model.proc_Psi(
        cs, h_node, h_edge, dissipation_context_features(cs)
    )
    return cs, model.head_Psi(cs, h_psi, cs.activation)


def test_tangential_channel_opposes_slip_and_is_passive():
    """Eq. 6.16-6.17: `lambda_tau` se opone a `u_tau` y no crea energía."""
    model = _tangential_model()
    pb = make_particles(
        [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]],
        v=[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        radius=0.5,
    )
    cs, (lam, psi, diag) = _head_output(model, pb)
    assert float((lam * cs.u).sum().detach()) < 0.0
    assert float(diag["relative_power_tau"]) < 0.0
    assert float(diag["Psi_tau"]) > 0.0
    assert float(psi.detach()) > 0.0
    # No hay aproximación normal: toda la respuesta es tangencial.
    assert float((lam * cs.n).sum(dim=-1).detach().abs().max()) <= 1e-12


def test_tangential_force_produces_spin_through_JT():
    """Una fuerza tangencial genera torque sin un decoder cartesiano libre."""
    model = _tangential_model(seed=8)
    pb = make_particles(
        [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]],
        v=[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        radius=0.5,
    )
    cs, (lam, _, _) = _head_output(model, pb)
    force, torque = JT_times_contact_vector(cs, lam, pb.n)
    assert float(torque.detach().abs().max()) > 1e-8
    power = (force * pb.v).sum() + (torque * pb.omega).sum()
    assert float(power.detach()) <= 1e-10


def test_tangential_full_step_changes_spin_and_conserves_total_momentum():
    model = _tangential_model(seed=9)
    pb = make_particles(
        [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]],
        v=[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        radius=0.5,
    )
    result = model.step(V3State(pb), 0.02, SurfaceSet([]))
    assert float(result.delta_L.detach().abs().max()) > 1e-8
    assert float(result.delta_p.detach().sum(dim=0).abs().max()) <= 1e-12
    total_L = (
        torch.linalg.cross(pb.q, result.delta_p, dim=-1) + result.delta_L
    ).sum(dim=0)
    assert float(total_L.detach().abs().max()) <= 1e-10


def test_dissipation_parameter_context_is_velocity_independent():
    """Los coeficientes convexos no pueden depender ocultamente de `s`."""
    model = _tangential_model(seed=10)
    q = [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]]
    a = make_particles(q, v=[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], radius=0.5)
    b = make_particles(
        q,
        v=[[3.0, -4.0, 2.0], [-2.0, 5.0, -1.0]],
        omega=[[7.0, 0.0, 1.0], [0.0, -3.0, 2.0]],
        radius=0.5,
    )
    ca, cb = build_set(a, cfg=model.cfg), build_set(b, cfg=model.cfg)
    na, ea, _ = model.encoder(ca, a)
    nb, eb, _ = model.encoder(cb, b)
    ha = model.proc_Psi(ca, na, ea, dissipation_context_features(ca))
    hb = model.proc_Psi(cb, nb, eb, dissipation_context_features(cb))
    assert torch.allclose(ha, hb, atol=1e-12, rtol=0.0)


def test_rotational_direct_channel_still_fails_explicitly():
    cfg = V3Config()
    cfg.dissipation.rotational = True
    with pytest.raises(NotImplementedError, match="rotacional"):
        DissipationHead(cfg.dissipation, cfg.encoder)


def test_tangential_channel_rejects_velocity_conditioned_coefficients():
    with pytest.raises(ValueError, match="state_independent_coefficients=True"):
        small_model(
            dissipation=DissipationConfig(
                tangential=True, state_independent_coefficients=False
            )
        )
