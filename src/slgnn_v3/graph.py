"""Grafo candidato partícula–partícula, seguro para batching (§4.4).

Reglas no negociables:

- cada par físico se almacena **una sola vez**, con `i < j`;
- se filtra por `batch_id` **antes** de medir vecinos: nunca hay una arista
  entre dos ejemplos concatenados, por muy próximos que estén en coordenadas;
- la construcción discreta corre sin gradiente, pero distancias, gaps,
  normales y puntos de contacto se recalculan diferenciablemente aguas abajo
  (`contact_kinematics`);
- se distingue arista candidata (grafo), contacto regular activo (ventana y
  router) y candidato impulsivo (detección con posición libre);
- los candidatos de cruce se incluyen evaluando la **posición libre
  provisional**, no solo la penetración ya observada.
"""

from __future__ import annotations

import torch

from .state import ParticleBatch


def candidate_pairs(
    q: torch.Tensor,
    radius: torch.Tensor,
    batch_id: torch.Tensor,
    gap_cut: float,
    q_free: torch.Tensor | None = None,
    gap_cut_free: float | None = None,
) -> torch.Tensor:
    """Pares `{i, j}` con `i < j`, mismo batch y `gap <= gap_cut`.

    Si se pasa `q_free`, se añaden también los pares cuyo gap en la posición
    libre cae por debajo de `gap_cut_free` (CCD aproximado): así una partícula
    rápida que atravesaría a otra entre snapshots entra al grafo antes de
    penetrar.

    Devuelve `[E, 2]` long, sin gradiente.
    """
    with torch.no_grad():
        n = q.shape[0]
        if n < 2:
            return torch.zeros(0, 2, dtype=torch.long, device=q.device)
        rsum = radius.unsqueeze(0) + radius.unsqueeze(1)
        same_batch = batch_id.unsqueeze(0) == batch_id.unsqueeze(1)
        mask = (torch.cdist(q, q) - rsum <= gap_cut) & same_batch
        if q_free is not None and gap_cut_free is not None:
            mask = mask | ((torch.cdist(q_free, q_free) - rsum <= gap_cut_free) & same_batch)
        return torch.triu(mask, diagonal=1).nonzero()


def build_candidate_graph(
    particles: ParticleBatch,
    gap_off: float,
    skin: float,
    q_free: torch.Tensor | None = None,
    ccd_margin: float = 0.0,
) -> torch.Tensor:
    """Neighbor list con capa de seguridad `skin`, más candidatos de cruce."""
    return candidate_pairs(
        particles.q,
        particles.radius,
        particles.batch_id,
        gap_cut=gap_off + skin,
        q_free=q_free,
        gap_cut_free=ccd_margin if q_free is not None else None,
    )


def free_positions(q: torch.Tensor, v_free: torch.Tensor, dt: float) -> torch.Tensor:
    """`q* = q + dt v*`: posición libre provisional tras las fuerzas regulares."""
    return q + dt * v_free


def assert_no_cross_batch(edges: torch.Tensor, batch_id: torch.Tensor) -> None:
    """Verificación barata usada por los tests y por el modo debug."""
    if edges.numel() == 0:
        return
    if not bool((batch_id[edges[:, 0]] == batch_id[edges[:, 1]]).all()):
        bad = (batch_id[edges[:, 0]] != batch_id[edges[:, 1]]).nonzero().flatten()
        raise AssertionError(
            f"{bad.numel()} aristas cruzan ejemplos del batch; primera: "
            f"{edges[bad[0]].tolist()}"
        )
