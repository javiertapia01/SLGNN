"""Utilidades comunes de la suite estructural de v3.

Todos los tests físicos corren en `float64` (§18): las tolerancias que exige
la especificación —`1e-12` en el punto común, `1e-10` en la identidad adjunta—
no son alcanzables en `float32`.
"""

from __future__ import annotations

import math

import pytest
import torch

from slgnn_v3 import (
    ParticleBatch,
    SLGNNv3,
    V3Config,
    V3State,
    box_surfaces,
)
from slgnn_v3.contact_kinematics import build_contacts
from slgnn_v3.graph import build_candidate_graph

DTYPE = torch.float64


@pytest.fixture(autouse=True)
def _float64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(DTYPE)
    yield
    torch.set_default_dtype(prev)


def make_particles(q, v=None, omega=None, radius=0.5, mass=1.0, batch_id=None,
                   type_id=None):
    q = torch.as_tensor(q, dtype=DTYPE)
    n = q.shape[0]
    zeros = torch.zeros(n, 3, dtype=DTYPE)
    r = torch.as_tensor(radius, dtype=DTYPE)
    m = torch.as_tensor(mass, dtype=DTYPE)
    return ParticleBatch.from_arrays(
        q=q,
        v=zeros.clone() if v is None else torch.as_tensor(v, dtype=DTYPE),
        omega=zeros.clone() if omega is None else torch.as_tensor(omega, dtype=DTYPE),
        mass=m.expand(n).clone() if m.ndim == 0 else m,
        radius=r.expand(n).clone() if r.ndim == 0 else r,
        batch_id=batch_id, type_id=type_id,
    )


def random_particles(n=8, seed=0, box=(0.6, 4.4), radius=0.5, spin=True):
    g = torch.Generator().manual_seed(seed)
    lo, hi = box
    q = lo + (hi - lo) * torch.rand(n, 3, generator=g, dtype=DTYPE)
    v = 0.5 * torch.randn(n, 3, generator=g, dtype=DTYPE)
    w = (0.5 * torch.randn(n, 3, generator=g, dtype=DTYPE)) if spin else torch.zeros(n, 3, dtype=DTYPE)
    return make_particles(q, v, w, radius=radius)


def default_box():
    return box_surfaces([0.0, 0.0, 0.0], [5.0, 5.0, 5.0])


def build_set(particles, surfaces=None, cfg=None, t=0.0):
    """Construye el `ContactSet` completo tal como lo hace el modelo."""
    cfg = cfg or V3Config()
    q = particles.q
    wall = None
    if surfaces is not None:
        wall = surfaces.query(q, particles.radius, particles.batch_id, t,
                              cfg.graph.pw_gap_off)
    edges = build_candidate_graph(particles, cfg.graph.pp_gap_off, cfg.graph.skin)
    return build_contacts(particles, edges, wall, cfg)


def small_model(profile=None, hidden=16, seed=0, **cfg_kw) -> SLGNNv3:
    torch.manual_seed(seed)
    cfg = V3Config(**cfg_kw)
    cfg.encoder.hidden = hidden
    cfg.potential.hidden = hidden
    cfg.dissipation.hidden = hidden
    cfg.impact.hidden = hidden
    if profile is not None:
        cfg.router.profile = profile
    return SLGNNv3(cfg).to(DTYPE)


def random_rotation(seed=0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(3, 3, generator=g, dtype=DTYPE)
    q_, r_ = torch.linalg.qr(a)
    q_ = q_ * torch.sign(torch.diagonal(r_)).unsqueeze(0)
    if torch.det(q_) < 0:
        q_[:, 0] = -q_[:, 0]
    return q_


def rel_error(a: torch.Tensor, b: torch.Tensor, floor: float = 1e-12) -> float:
    """Error relativo con piso absoluto.

    El piso importa: en el MVP normal `Delta L` es **exactamente cero** (no hay
    canal tangencial), así que `||b|| = 0` y un cociente puro reportaría un
    error infinito a partir de polvo numérico de 1e-19.
    """
    a, b = a.detach(), b.detach()
    return float((a - b).norm() / max(float(b.norm()), floor))
