"""Diagnósticos obligatorios de cada forward (§12 de las instrucciones).

Regla que define este módulo: **un campo no implementado se marca como
desactivado, nunca se rellena con un número que parezca un resultado válido.**
`DISABLED` es un centinela explícito y `StepDiagnostics.disabled_fields()`
lista qué falta, para que un informe no pueda presentar como medido algo que
no existe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

DISABLED = "disabled"


def _num(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return float(x.detach()) if x.numel() == 1 else x.detach()
    return x


@dataclass
class StepDiagnostics:
    """Salidas diagnósticas de un paso completo."""

    energies: dict[str, Any] = field(default_factory=dict)
    dissipation: dict[str, Any] = field(default_factory=dict)
    regular: dict[str, Any] = field(default_factory=dict)
    impact: dict[str, Any] = field(default_factory=dict)
    router: dict[str, Any] = field(default_factory=dict)
    geometry: dict[str, Any] = field(default_factory=dict)
    solver: dict[str, Any] = field(default_factory=dict)
    balance: dict[str, Any] = field(default_factory=dict)
    wall: dict[str, Any] = field(default_factory=dict)

    def sections(self) -> dict[str, dict]:
        return {
            "energies": self.energies, "dissipation": self.dissipation,
            "regular": self.regular, "impact": self.impact, "router": self.router,
            "geometry": self.geometry, "solver": self.solver,
            "balance": self.balance, "wall": self.wall,
        }

    def disabled_fields(self) -> list[str]:
        return [
            f"{name}.{k}" for name, sec in self.sections().items()
            for k, v in sec.items() if v == DISABLED
        ]

    def scalars(self) -> dict[str, float]:
        """Aplana a escalares para logging; ignora tensores no escalares."""
        out: dict[str, float] = {}
        for name, sec in self.sections().items():
            for k, v in sec.items():
                if v == DISABLED:
                    continue
                v = _num(v)
                if isinstance(v, (int, float)):
                    out[f"{name}.{k}"] = float(v)
        return out


def internal_momentum_error(
    dp: torch.Tensor, dL: torch.Tensor, q: torch.Tensor
) -> dict[str, float]:
    """Error de conservación de momento **interno**, `[N,3]` de entrada.

    `dp` y `dL` deben contener **solo** contribuciones partícula–partícula. Un
    contacto con la pared transfiere momento a un cuerpo que no es grado de
    libertad del sistema, y una pared móvil además inyecta energía: incluirlos
    aquí convertiría física correcta en un "error" espurio.

    Se reporta como error medido, no se impone como penalización: la
    conservación ya está garantizada por aplicar un único vector por contacto
    mediante `J^T`. Este número existe para detectar errores numéricos o de
    implementación, no para entrenarse contra él.
    """
    lin = dp.sum(dim=0)
    ang = (torch.linalg.cross(q, dp, dim=-1) + dL).sum(dim=0)
    scale = dp.norm(dim=-1).sum().clamp_min(1e-30)
    ang_scale = (
        torch.linalg.cross(q, dp, dim=-1).norm(dim=-1).sum() + dL.norm(dim=-1).sum()
    ).clamp_min(1e-30)
    return {
        "internal_linear_momentum_error": float(lin.norm()),
        "internal_angular_momentum_error": float(ang.norm()),
        "internal_linear_momentum_error_relative": float(lin.norm() / scale),
        "internal_angular_momentum_error_relative": float(ang.norm() / ang_scale),
    }


def wall_transfer(dp_wall: torch.Tensor, dL_wall: torch.Tensor) -> dict[str, float]:
    """Momento transferido a/desde la pared. No es un error: es física."""
    return {
        "wall_linear_impulse": float(dp_wall.sum(dim=0).norm()),
        "wall_angular_impulse": float(dL_wall.sum(dim=0).norm()),
    }
