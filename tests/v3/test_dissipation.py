"""Cabeza `Psi`: convexidad, `psi(0) = 0` y pasividad."""

import pytest
import torch

from slgnn_v3 import V3Config
from slgnn_v3.dissipation import DissipationHead

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
        from slgnn_v3.encoder import kinematic_features
        kin = kinematic_features(cs, pb)
        h_psi = model.proc_Psi(cs, h_node, h_edge, kin)
        lam, _, diag = model.head_Psi(cs, h_psi, torch.ones_like(cs.gap) * cs.activation)
        assert float((lam * cs.u).detach().sum()) <= 1e-10


def test_dissipation_inactive_when_separated():
    """Sin contacto no hay amortiguamiento: la activación unilateral es cero."""
    model = small_model(seed=5)
    q = torch.tensor([[0.0, 0.0, 0.0], [1.25, 0.0, 0.0]], dtype=DTYPE)
    pb = make_particles(q, v=[[3.0, 0.0, 0.0], [0.0, 0.0, 0.0]], radius=0.5)
    cs = build_set(pb, cfg=model.cfg)
    assert float(cs.activation.max()) == 0.0
    from slgnn_v3.encoder import kinematic_features
    h_node, h_edge, _ = model.encoder(cs, pb)
    lam, psi, _ = model.head_Psi(
        cs, model.proc_Psi(cs, h_node, h_edge, kinematic_features(cs, pb)),
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
    from slgnn_v3.encoder import kinematic_features
    h_node, h_edge, _ = model.encoder(cs, pb)
    lam, _, _ = model.head_Psi(
        cs, model.proc_Psi(cs, h_node, h_edge, kinematic_features(cs, pb)),
        torch.ones_like(cs.gap) * cs.activation,
    )
    assert float((lam * cs.n).detach().sum()) > 0     # a lo largo de n, de i hacia j


def test_tangential_channel_refuses_to_pretend():
    cfg = V3Config()
    cfg.dissipation.tangential = True
    with pytest.raises(NotImplementedError, match="tangencial"):
        DissipationHead(cfg.dissipation, cfg.encoder)
