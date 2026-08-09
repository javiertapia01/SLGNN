"""Targets primarios: incrementos de momento (§1.4 instrucciones, §13.1 v3).

    Delta p_i^k = m_i (v_i^{k+1} - v_i^k)
    Delta L_i^k = I_i (omega_i^{k+1} - omega_i^k)

Estos objetos son válidos tanto para una fuerza integrada sobre el paso como
para un impulso concentrado, que es exactamente la razón por la que v3 los usa
en lugar de una aceleración instantánea. La aceleración se conserva como
métrica secundaria/ablation (`accelerations`), nunca como semántica principal.

La construcción vive aquí y **solo** aquí: v3 y GNS consumen el mismo tensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import Trajectory


@dataclass
class TransitionTargets:
    """Targets de todas las transiciones (k, k+1) de una trayectoria."""

    delta_p: torch.Tensor      # [T-1, N, 3]
    delta_L: torch.Tensor      # [T-1, N, 3]
    q_next: torch.Tensor       # [T-1, N, 3]
    v_next: torch.Tensor       # [T-1, N, 3]
    omega_next: torch.Tensor   # [T-1, N, 3]
    dt: float
    name: str = ""

    @property
    def n_transitions(self) -> int:
        return int(self.delta_p.shape[0])


def build_targets(tr: Trajectory) -> TransitionTargets:
    """Incrementos de momento por transición. No filtra ni suaviza nada."""
    m = tr.mass.unsqueeze(0).unsqueeze(-1)   # [1, N, 1]
    inertia = tr.inertia.unsqueeze(0).unsqueeze(-1)
    return TransitionTargets(
        delta_p=m * (tr.v[1:] - tr.v[:-1]),
        delta_L=inertia * (tr.omega[1:] - tr.omega[:-1]),
        q_next=tr.q[1:],
        v_next=tr.v[1:],
        omega_next=tr.omega[1:],
        dt=tr.dt,
        name=tr.name,
    )


def accelerations(tr: Trajectory) -> tuple[torch.Tensor, torch.Tensor]:
    """Aceleraciones lineal y angular por diferencias hacia adelante.

    Métrica secundaria y objetivo de ablation (§15.7.8 de la formulación), no
    la semántica de target de v3.
    """
    return (tr.v[1:] - tr.v[:-1]) / tr.dt, (tr.omega[1:] - tr.omega[:-1]) / tr.dt


def momentum_totals(tr: Trajectory) -> tuple[torch.Tensor, torch.Tensor]:
    """Momento lineal y angular totales por snapshot, [T,3] cada uno.

    El momento angular incluye la parte orbital respecto del origen más el
    spin: `L = sum_i q_i x m_i v_i + sum_i I_i omega_i`.
    """
    m = tr.mass.unsqueeze(0).unsqueeze(-1)
    inertia = tr.inertia.unsqueeze(0).unsqueeze(-1)
    p = (m * tr.v).sum(dim=1)
    L = torch.linalg.cross(tr.q, m * tr.v, dim=-1).sum(dim=1) + (inertia * tr.omega).sum(dim=1)
    return p, L
