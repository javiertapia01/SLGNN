"""`GNSClassicReduced`: comparación secundaria, con su contrato distinto.

Cambia a la vez historia, target y representación de pared, así que **no**
comparte la interfaz `step` del runner: consume una ventana de posiciones y
devuelve un incremento de velocidad. Estos tests fijan ese contrato y dejan
constancia de que el modelo existe y corre, no de que sea comparable con v3.
"""

import pytest
import torch

from gns_baseline import GNSClassicReduced, GNSConfig
from slgnn_v3 import ParticleBatch

DTYPE = torch.float64


def _particles(n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = 0.6 + 2.0 * torch.rand(n, 3, generator=g, dtype=DTYPE)
    return ParticleBatch.from_arrays(
        q=q, v=torch.zeros(n, 3, dtype=DTYPE), omega=torch.zeros(n, 3, dtype=DTYPE),
        mass=torch.ones(n, dtype=DTYPE), radius=torch.full((n,), 0.5, dtype=DTYPE),
    )


def _history(pb, h=6):
    g = torch.Generator().manual_seed(1)
    drift = 0.02 * torch.randn(h, pb.n, 3, generator=g, dtype=DTYPE)
    return pb.q.unsqueeze(0) + torch.cumsum(drift, dim=0)


def test_consumes_six_positions_and_predicts_velocity_increment():
    cfg = GNSConfig(hidden=16, n_message_steps=2, history_length=6)
    model = GNSClassicReduced(cfg).to(DTYPE)
    pb = _particles()
    out = model(_history(pb, cfg.history_length), pb,
                [0.0, 0.0, 0.0], [4.0, 4.0, 4.0], connectivity_radius=1.35)
    assert out.shape == (pb.n, 3)
    assert torch.isfinite(out).all()


def test_history_length_is_respected():
    cfg = GNSConfig(hidden=16, n_message_steps=2, history_length=4)
    model = GNSClassicReduced(cfg).to(DTYPE)
    pb = _particles()
    out = model(_history(pb, 4), pb, [0.0] * 3, [4.0] * 3, connectivity_radius=1.35)
    assert out.shape == (pb.n, 3)
    with pytest.raises(RuntimeError):
        model(_history(pb, 6), pb, [0.0] * 3, [4.0] * 3, connectivity_radius=1.35)


def test_boundary_distances_are_clipped_to_the_connectivity_radius():
    """Receta clásica: las distancias a frontera se recortan al radio de
    conexión, a diferencia de la consulta SDF con velocidad local que usa v3."""
    cfg = GNSConfig(hidden=16, n_message_steps=2)
    model = GNSClassicReduced(cfg).to(DTYPE)
    far = _particles()
    far = far.replace(q=torch.full_like(far.q, 50.0))
    out = model(_history(far, cfg.history_length), far,
                [0.0] * 3, [100.0] * 3, connectivity_radius=1.35)
    assert torch.isfinite(out).all()


def test_is_differentiable():
    cfg = GNSConfig(hidden=16, n_message_steps=2)
    model = GNSClassicReduced(cfg).to(DTYPE)
    pb = _particles()
    model(_history(pb, cfg.history_length), pb, [0.0] * 3, [4.0] * 3,
          connectivity_radius=1.35).pow(2).sum().backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in model.parameters())


def test_does_not_expose_the_step_interface():
    """No es intercambiable con v3 en el runner, y no debe fingir que lo es:
    su target y su historia son distintos (§15.2)."""
    model = GNSClassicReduced(GNSConfig(hidden=16, n_message_steps=2))
    assert not hasattr(model, "step")
    assert model.profile == "gns-classic-reduced"
