"""Integrador híbrido fuerza–impulso (§11 de las instrucciones, §9.3 de v3).

La ecuación discreta central de SLGNN-v3 es

    M (nu_{k+1} - nu_k) = dt F_reg,k + J_k^T Lambda_k,

implementada como split semiimplícito:

    nu*      = nu_k + dt M^-1 F_reg,k          (fuerzas regulares)
    nu_{k+1} = nu*   + M^-1 J^T Lambda         (impulsos)
    q_{k+1}  = q_k + dt K nu_{k+1}             (posición con velocidad POST-impulso)

Aquí viven solo el álgebra del paso y el contenedor de resultados; la
orquestación de cabezas está en `model.py`.

No hay corrección posicional para esconder penetraciones. Si algún día se
añade una proyección de emergencia, debe venir desactivada por defecto y
reportar su magnitud y energía artificial por separado (§11 final).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .diagnostics import StepDiagnostics
from .state import ParticleBatch, V3State


@dataclass
class StepResult:
    """Salida de un paso completo. **No** es `(a, alpha)`.

    Las aceleraciones, si hacen falta, se derivan de la contribución regular y
    se etiquetan como tales: la semántica principal de v3 es el incremento de
    momento (§3.2).
    """

    next_state: V3State
    delta_p_regular: torch.Tensor    # [N, 3]
    delta_p_impulse: torch.Tensor    # [N, 3]
    delta_L_regular: torch.Tensor    # [N, 3]
    delta_L_impulse: torch.Tensor    # [N, 3]
    forces: torch.Tensor             # [N, 3] fuerza regular total
    torques: torch.Tensor            # [N, 3] torque regular total
    impulses: torch.Tensor           # [C_I] impulsos normales del solver
    diagnostics: StepDiagnostics = field(default_factory=StepDiagnostics)
    contacts: object | None = None   # ContactSet del paso, para pérdidas y métricas
    mode: torch.Tensor | None = None # [C] modo del router

    @property
    def delta_p(self) -> torch.Tensor:
        return self.delta_p_regular + self.delta_p_impulse

    @property
    def delta_L(self) -> torch.Tensor:
        return self.delta_L_regular + self.delta_L_impulse

    def regular_accelerations(self, mass: torch.Tensor, inertia: torch.Tensor):
        """Aceleraciones **de la rama regular únicamente**, para compatibilidad
        diagnóstica. No es el target de v3."""
        return (
            self.forces / mass.unsqueeze(-1),
            self.torques / inertia.unsqueeze(-1),
        )


def free_velocity(
    particles: ParticleBatch, force: torch.Tensor, torque: torch.Tensor, dt: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """`nu* = nu_k + dt M^-1 F_reg`."""
    v_free = particles.v + dt * force / particles.mass.unsqueeze(-1)
    w_free = particles.omega + dt * torque / particles.inertia.unsqueeze(-1)
    return v_free, w_free


def apply_impulse(
    particles: ParticleBatch,
    v_free: torch.Tensor,
    w_free: torch.Tensor,
    impulse_force: torch.Tensor,
    impulse_torque: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`nu_{k+1} = nu* + M^-1 J^T Lambda`."""
    v = v_free + impulse_force / particles.mass.unsqueeze(-1)
    w = w_free + impulse_torque / particles.inertia.unsqueeze(-1)
    return v, w


def advance_position(q: torch.Tensor, v_next: torch.Tensor, dt: float) -> torch.Tensor:
    """`q_{k+1} = q_k + dt v_{k+1}`, con la velocidad **post-impulso**."""
    return q + dt * v_next


def gravity_force(particles: ParticleBatch, g: torch.Tensor | None) -> torch.Tensor:
    """`F_ext = m_i g`, contada una sola vez y registrada por separado (D-006)."""
    if g is None:
        return torch.zeros_like(particles.q)
    g = torch.as_tensor(g, dtype=particles.dtype, device=particles.device)
    return particles.mass.unsqueeze(-1) * g
