import torch

from slgnn.config import SLGNNConfig
from slgnn.graph import contact_velocities, neighbor_pairs, pair_geometry


def _cfg():
    return SLGNNConfig()


def test_neighbor_pairs_radius():
    q = torch.tensor(
        [[0.0, 0, 0], [1.2, 0, 0], [5.0, 0, 0]], dtype=torch.float64
    )
    idx = neighbor_pairs(q, r_list=2.0)
    assert idx.tolist() == [[0, 1]]


def test_pair_geometry_two_spheres():
    cfg = _cfg()
    q = torch.tensor([[0.0, 0, 0], [1.2, 0, 0]], dtype=torch.float64)
    radii = torch.tensor([0.5, 0.5], dtype=torch.float64)
    ps = pair_geometry(q, radii, neighbor_pairs(q, cfg.r_list), cfg)
    assert torch.allclose(ps.d, torch.tensor([1.2], dtype=torch.float64))
    assert torch.allclose(ps.e[0], torch.tensor([1.0, 0, 0], dtype=torch.float64), atol=1e-8)
    assert torch.allclose(ps.g, torch.tensor([0.2], dtype=torch.float64))
    # punto común de contacto: brazos proporcionales a radios, l_i + l_j = d (§9)
    assert torch.allclose(ps.l_i + ps.l_j, ps.d)
    assert torch.allclose(ps.l_i, torch.tensor([0.6], dtype=torch.float64))


def test_contact_velocity_pure_spin():
    """Dos esferas girando con el mismo omega_z: deslizamiento tangencial puro.

    v_i^c = W ẑ x (l_i x̂) = W l_i ŷ ; v_j^c = W ẑ x (-l_j x̂) = -W l_j ŷ
    => u = v_j^c - v_i^c = -W(l_i + l_j) ŷ = -W d ŷ ; u_n = 0.
    """
    cfg = _cfg()
    q = torch.tensor([[0.0, 0, 0], [1.0, 0, 0]], dtype=torch.float64)
    radii = torch.tensor([0.5, 0.5], dtype=torch.float64)
    ps = pair_geometry(q, radii, neighbor_pairs(q, cfg.r_list), cfg)
    v = torch.zeros(2, 3, dtype=torch.float64)
    big_w = 2.0
    omega = torch.tensor([[0, 0, big_w], [0, 0, big_w]], dtype=torch.float64)
    u_c, u_n, u_tau = contact_velocities(v, omega, ps)
    assert torch.allclose(u_n, torch.zeros(1, dtype=torch.float64), atol=1e-10)
    assert torch.allclose(
        u_c[0], torch.tensor([0.0, -big_w * 1.0, 0.0], dtype=torch.float64), atol=1e-8
    )
    assert torch.allclose(u_tau, u_c, atol=1e-10)


def test_contact_velocity_head_on():
    cfg = _cfg()
    q = torch.tensor([[0.0, 0, 0], [1.1, 0, 0]], dtype=torch.float64)
    radii = torch.tensor([0.5, 0.5], dtype=torch.float64)
    ps = pair_geometry(q, radii, neighbor_pairs(q, cfg.r_list), cfg)
    v = torch.tensor([[1.0, 0, 0], [-1.0, 0, 0]], dtype=torch.float64)
    omega = torch.zeros(2, 3, dtype=torch.float64)
    _, u_n, u_tau = contact_velocities(v, omega, ps)
    # acercamiento => u_n < 0 (§9)
    assert u_n.item() < 0
    assert torch.allclose(u_n, torch.tensor([-2.0], dtype=torch.float64), atol=1e-7)
    # residuo O(eps/d) del unitario regularizado e = r/(d + eps)
    assert u_tau.norm() < 1e-6
