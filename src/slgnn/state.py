"""Contenedores de estado para SLGNN-v2."""

from dataclasses import dataclass, field

import torch


@dataclass
class Particles:
    """Propiedades conocidas de las partículas (§3): no se aprenden."""

    m: torch.Tensor         # [N] masas
    radii: torch.Tensor     # [N] radios
    type_ids: torch.Tensor  # [N] long, tipo tau_i
    props: torch.Tensor     # [N,P] propiedades materiales mu_i (puede ser P=0)

    @property
    def inertia(self) -> torch.Tensor:
        """Esfera maciza isotrópica: I = (2/5) m R² (§3)."""
        return 0.4 * self.m * self.radii**2

    @staticmethod
    def uniform(n: int, m: float, radius: float, n_props: int = 0,
                dtype=torch.float32) -> "Particles":
        return Particles(
            m=torch.full((n,), m, dtype=dtype),
            radii=torch.full((n,), radius, dtype=dtype),
            type_ids=torch.zeros(n, dtype=torch.long),
            props=torch.zeros(n, n_props, dtype=dtype),
        )


@dataclass
class SLGNNOutput:
    """Salida del forward: aceleraciones y diagnósticos para las pérdidas."""

    a: torch.Tensor          # [N,3] aceleración translacional
    alpha: torch.Tensor      # [N,3] aceleración angular
    diagnostics: dict = field(default_factory=dict)
