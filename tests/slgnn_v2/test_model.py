"""Tests de las garantías por construcción de SLGNN-v2 (§37)."""

import torch

from slgnn import BoxSDF, Particles, SLGNN, SLGNNConfig


def _small_cfg(**kw):
    return SLGNNConfig(hidden=16, layers=1, h_mat=4, **kw)


def _model(seed=0, **kw):
    torch.manual_seed(seed)
    return SLGNN(_small_cfg(**kw)).double()


def _randomize_history_heads(model):
    """El canal H parte en cero por diseño; para testear sus simetrías se
    re-aleatoriza la última capa."""
    for head in (model.head_pp_H, model.head_pw_H):
        torch.nn.init.normal_(head[-1].weight, std=0.5)
        torch.nn.init.normal_(head[-1].bias, std=0.5)


def _cluster(n=8, seed=1):
    torch.manual_seed(seed)
    q = 1.15 * torch.randn(n, 3, dtype=torch.float64)
    v = torch.randn(n, 3, dtype=torch.float64)
    w = torch.randn(n, 3, dtype=torch.float64)
    p = Particles.uniform(n, m=1.0, radius=0.5, dtype=torch.float64)
    return q, v, w, p


def _rotation(seed=3):
    torch.manual_seed(seed)
    aa = torch.randn(3, dtype=torch.float64)
    skew = torch.tensor(
        [[0, -aa[2], aa[1]], [aa[2], 0, -aa[0]], [-aa[1], aa[0], 0]],
        dtype=torch.float64,
    )
    return torch.linalg.matrix_exp(skew)


def test_shapes_and_finiteness():
    model = _model()
    q, v, w, p = _cluster()
    out = model(q, v, w, p, g_vec=torch.tensor([0.0, 0, -0.196], dtype=torch.float64))
    assert out.a.shape == (8, 3) and out.alpha.shape == (8, 3)
    assert torch.isfinite(out.a).all() and torch.isfinite(out.alpha).all()
    assert out.diagnostics["n_edges"] > 0


def test_single_free_particle_pure_gravity():
    model = _model()
    p = Particles.uniform(1, m=1.0, radius=0.5, dtype=torch.float64)
    q = torch.zeros(1, 3, dtype=torch.float64)
    v = torch.zeros(1, 3, dtype=torch.float64)
    w = torch.zeros(1, 3, dtype=torch.float64)
    g = torch.tensor([0.0, 0.0, -0.196], dtype=torch.float64)
    out = model(q, v, w, p, g_vec=g)
    assert torch.allclose(out.a[0], g, atol=1e-12)
    assert torch.allclose(out.alpha, torch.zeros_like(out.alpha), atol=1e-12)


def test_history_starts_at_zero():
    model = _model()
    q, v, w, p = _cluster()
    out = model(q, v, w, p)
    assert torch.allclose(out.diagnostics["f_H"], torch.zeros_like(out.diagnostics["f_H"]))
    assert torch.allclose(out.diagnostics["tau_H"], torch.zeros_like(out.diagnostics["tau_H"]))


def test_internal_momentum_conservation():
    """Sin gravedad ni pared: suma de fuerzas = 0 y momento angular total
    (orbital + spin) conservado (§37), incluyendo canal H re-aleatorizado."""
    model = _model()
    _randomize_history_heads(model)
    q, v, w, p = _cluster()
    out = model(q, v, w, p)
    f_total = (
        out.diagnostics["f_cons"] + out.diagnostics["f_R"] + out.diagnostics["f_H"]
    )
    tau_total = out.diagnostics["tau_R"] + out.diagnostics["tau_H"]
    assert f_total.sum(dim=0).abs().max() < 1e-10
    # el balance angular hereda un residuo O(eps/d) del unitario regularizado
    # e_ij = r/(d + eps); con fuerzas O(1) el residuo esperado es ~1e-8
    torque_about_origin = torch.linalg.cross(q, f_total, dim=-1).sum(dim=0)
    assert (torque_about_origin + tau_total.sum(dim=0)).abs().max() < 1e-7


def test_so3_equivariance():
    """Rotar simultáneamente partículas, velocidades, spins, gravedad y pared
    debe rotar las aceleraciones (§11, §37). Se testea con caja posada."""
    model = _model()
    _randomize_history_heads(model)
    torch.manual_seed(7)
    n = 6
    q = 1.5 + torch.rand(n, 3, dtype=torch.float64) * 3.0   # dentro de [0,6]^3
    v = torch.randn(n, 3, dtype=torch.float64)
    w = torch.randn(n, 3, dtype=torch.float64)
    p = Particles.uniform(n, m=1.0, radius=0.5, dtype=torch.float64)
    g = torch.tensor([0.0, 0.0, -0.196], dtype=torch.float64)
    rot = _rotation()

    wall1 = BoxSDF([0.0] * 3, [6.0] * 3)
    wall2 = BoxSDF([0.0] * 3, [6.0] * 3, pose=(rot, torch.zeros(3)))

    out1 = model(q, v, w, p, wall=wall1, g_vec=g)
    out2 = model(q @ rot.T, v @ rot.T, w @ rot.T, p, wall=wall2, g_vec=g @ rot.T)

    assert torch.allclose(out2.a, out1.a @ rot.T, atol=1e-9)
    assert torch.allclose(out2.alpha, out1.alpha @ rot.T, atol=1e-9)


def test_rayleigh_dissipates_on_approach():
    """Potencia del canal de Rayleigh no positiva con pared estática (§37)."""
    model = _model()
    q = torch.tensor([[0.0, 0, 0], [1.2, 0, 0]], dtype=torch.float64)
    v = torch.tensor([[1.0, 0, 0], [-1.0, 0, 0]], dtype=torch.float64)
    w = torch.zeros(2, 3, dtype=torch.float64)
    p = Particles.uniform(2, m=1.0, radius=0.5, dtype=torch.float64)
    out = model(q, v, w, p)
    power = (v * out.diagnostics["f_R"]).sum() + (w * out.diagnostics["tau_R"]).sum()
    assert power.item() <= 1e-12
    assert out.diagnostics["R_pp"].item() > 0  # hay disipación activa


def test_rayleigh_wall_dissipates():
    model = _model()
    wall = BoxSDF([0.0] * 3, [6.0] * 3)
    q = torch.tensor([[0.7, 3.0, 3.0]], dtype=torch.float64)  # gap 0.2 a la cara x=0
    v = torch.tensor([[-1.0, 0.5, 0.0]], dtype=torch.float64)  # acercándose
    w = torch.zeros(1, 3, dtype=torch.float64)
    p = Particles.uniform(1, m=1.0, radius=0.5, dtype=torch.float64)
    out = model(q, v, w, p, wall=wall)
    power = (v * out.diagnostics["f_R"]).sum() + (w * out.diagnostics["tau_R"]).sum()
    assert power.item() <= 1e-12
    assert out.diagnostics["R_pW"].item() > 0


def test_trainability_double_backward():
    """El doble backward (fuerzas por autograd dentro del grafo de la pérdida)
    entrena. El target debe conservar momento (repulsión antisimétrica): un
    target que no lo conserve es inalcanzable por construcción (§37)."""
    model = _model(seed=11)
    q0 = torch.tensor([[0.0, 0, 0], [1.3, 0, 0]], dtype=torch.float64)
    v0 = torch.tensor([[0.5, 0, 0], [-0.5, 0, 0]], dtype=torch.float64)
    w0 = torch.zeros(2, 3, dtype=torch.float64)
    p = Particles.uniform(2, m=1.0, radius=0.5, dtype=torch.float64)
    target = torch.tensor([[-0.3, 0, 0], [0.3, 0, 0]], dtype=torch.float64)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    losses = []
    for _ in range(40):
        opt.zero_grad()
        out = model(q0, v0, w0, p)
        loss = (out.a - target).pow(2).sum()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]
