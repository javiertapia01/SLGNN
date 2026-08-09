"""`GNSControlled`: contrato de interfaz, capacidad y ausencia de estructura.

El baseline debe ser un competidor honesto: mismo protocolo, presupuesto de
parámetros comparable y ajustado a conciencia. Estos tests fijan lo primero;
lo segundo se reporta en cada manifiesto.
"""

import pytest
import torch

from gns_baseline import GNSConfig, GNSControlled
from gns_baseline.encoder import to_directed
from slgnn_v3 import ParticleBatch, SLGNNv3, V3Config, V3State, box_surfaces

DTYPE = torch.float64


def _pb(n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = 0.6 + 3.0 * torch.rand(n, 3, generator=g, dtype=DTYPE)
    return ParticleBatch.from_arrays(
        q=q, v=0.4 * torch.randn(n, 3, generator=g, dtype=DTYPE),
        omega=0.4 * torch.randn(n, 3, generator=g, dtype=DTYPE),
        mass=torch.ones(n, dtype=DTYPE), radius=torch.full((n,), 0.5, dtype=DTYPE),
    )


def test_step_interface_matches_v3():
    model = GNSControlled(GNSConfig(hidden=16, n_message_steps=2)).to(DTYPE)
    res = model.step(V3State(_pb(), time=0.3), 0.05,
                     box_surfaces([0, 0, 0], [5, 5, 5]),
                     torch.tensor([0.0, 0.0, -0.98], dtype=DTYPE))
    for field in ("next_state", "delta_p", "delta_L", "delta_p_regular",
                  "delta_p_impulse", "impulses", "diagnostics"):
        assert hasattr(res, field)
    assert res.delta_p.shape == (4, 3)
    assert res.next_state.time == pytest.approx(0.35)


def test_central_equation_still_holds_for_the_baseline():
    """Aunque no haya solver, el estado siguiente debe ser consistente con el
    incremento de momento que el modelo dice haber aplicado."""
    model = GNSControlled(GNSConfig(hidden=16, n_message_steps=2)).to(DTYPE)
    pb = _pb()
    res = model.step(V3State(pb, time=0.0), 0.05, box_surfaces([0, 0, 0], [5, 5, 5]))
    p = res.next_state.particles
    assert torch.allclose(pb.mass.unsqueeze(-1) * (p.v - pb.v), res.delta_p, atol=1e-14)
    assert torch.allclose(pb.inertia.unsqueeze(-1) * (p.omega - pb.omega),
                          res.delta_L, atol=1e-14)


def test_baseline_declares_what_it_does_not_have():
    """No se rellenan con ceros los diagnósticos que este modelo no produce."""
    model = GNSControlled(GNSConfig(hidden=16, n_message_steps=2)).to(DTYPE)
    res = model.step(V3State(_pb(), time=0.0), 0.05, box_surfaces([0, 0, 0], [5, 5, 5]))
    disabled = res.diagnostics.disabled_fields()
    for expected in ("energies.available", "impact.available", "solver.available",
                     "balance.conservation"):
        assert expected in disabled


def test_baseline_does_not_conserve_momentum_by_construction():
    """Es la diferencia que la comparación quiere aislar: sin `J^T`, la suma de
    incrementos internos no tiene por qué anularse. Si diera cero, el baseline
    tendría gratis la estructura de v3 y la comparación no mediría nada."""
    torch.manual_seed(1)
    model = GNSControlled(GNSConfig(hidden=16, n_message_steps=2)).to(DTYPE)
    for p in model.decoder.parameters():
        torch.nn.init.normal_(p, std=0.5)
    from slgnn_v3.surfaces import SurfaceSet
    res = model.step(V3State(_pb(6, seed=3), time=0.0), 0.05, SurfaceSet([]))
    assert float(res.delta_p.detach().sum(dim=0).abs().max()) > 1e-6


def test_directed_expansion_doubles_edges():
    e = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    d = to_directed(e)
    assert d.shape == (2, 4)
    assert set(map(tuple, d.t().tolist())) == {(0, 1), (1, 0), (1, 2), (2, 1)}


def test_parameter_budget_is_comparable_to_v3():
    """El presupuesto se reporta siempre; aquí se fija que la configuración
    por defecto queda en el mismo orden de magnitud que v3."""
    v3_cfg = V3Config()
    v3_cfg.encoder.hidden = v3_cfg.potential.hidden = 48
    v3_cfg.dissipation.hidden = v3_cfg.impact.hidden = 48
    n_v3 = SLGNNv3(v3_cfg).n_parameters()[0]
    n_gns = GNSControlled(GNSConfig(hidden=88, n_message_steps=4)).n_parameters()[0]
    assert 0.5 <= n_gns / n_v3 <= 2.0, f"v3={n_v3} gns={n_gns}"


def test_gradients_reach_every_block():
    model = GNSControlled(GNSConfig(hidden=16, n_message_steps=2)).to(DTYPE)
    res = model.step(V3State(_pb(), time=0.0), 0.05, box_surfaces([0, 0, 0], [5, 5, 5]))
    (res.delta_p.pow(2).sum() + res.delta_L.pow(2).sum()).backward()
    blocks = {n.split(".")[0] for n, p in model.named_parameters()
              if p.grad is not None and float(p.grad.abs().sum()) > 0}
    assert {"encoder", "processor", "decoder"} <= blocks
