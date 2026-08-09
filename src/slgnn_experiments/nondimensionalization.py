"""Adimensionalización única y coherente (§3.4 de las instrucciones, §13.3 de v3).

Una sola transformación derivada de tres escalas `L0`, `M0`, `T0`. Todo lo
demás se deriva de ellas: no se normalizan independientemente magnitudes que
deben satisfacer la misma ecuación mecánica.

    q' = q/L0            v' = v/(L0/T0)        omega' = omega*T0
    m' = m/M0            I' = I/(M0 L0^2)      Lambda' = Lambda/(M0 L0/T0)
    F' = F/(M0 L0/T0^2)  V' = V/(M0 L0^2/T0^2) Psi' = Psi/(M0 L0^2/T0^3)

Las escalas de target son `P0 = M0 L0 / T0` (momento lineal) y
`L0 P0 = M0 L0^2 / T0` (momento angular).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

from .data import DEFAULT_DENSITY, DEFAULT_DIAMETER, Trajectory


@dataclass(frozen=True)
class Scales:
    """Escalas físicas de referencia. Se guardan en cada checkpoint y manifiesto."""

    L0: float
    M0: float
    T0: float

    # --- escalas derivadas -------------------------------------------------
    @property
    def V0(self) -> float:
        """Velocidad."""
        return self.L0 / self.T0

    @property
    def W0(self) -> float:
        """Velocidad angular."""
        return 1.0 / self.T0

    @property
    def P0(self) -> float:
        """Momento lineal; escala de `Delta p` y del impulso `Lambda`."""
        return self.M0 * self.L0 / self.T0

    @property
    def LP0(self) -> float:
        """Momento angular; escala de `Delta L`."""
        return self.L0 * self.P0

    @property
    def F0(self) -> float:
        """Fuerza."""
        return self.M0 * self.L0 / self.T0**2

    @property
    def E0(self) -> float:
        """Energía."""
        return self.M0 * self.L0**2 / self.T0**2

    @property
    def PSI0(self) -> float:
        """Potencia (dimensión del potencial disipativo)."""
        return self.M0 * self.L0**2 / self.T0**3

    @property
    def I0(self) -> float:
        """Inercia."""
        return self.M0 * self.L0**2

    # --- conversiones ------------------------------------------------------
    def length(self, x: float) -> float:
        return x / self.L0

    def gravity(self, g: float) -> float:
        """Aceleración -> adimensional (dividida por L0/T0^2)."""
        return g * self.T0**2 / self.L0

    def time(self, t: float) -> float:
        return t / self.T0

    def nondim(self, tr: Trajectory) -> Trajectory:
        """Adimensionaliza una trayectoria completa. Idempotencia prohibida:
        aplicar dos veces levanta un error en vez de escalar dos veces."""
        if tr.dimensionless:
            raise ValueError(
                f"{tr.name}: la trayectoria ya está adimensionalizada. "
                "La transformación se aplica exactamente una vez."
            )
        return Trajectory(
            q=tr.q / self.L0,
            v=tr.v / self.V0,
            omega=tr.omega / self.W0,
            mass=tr.mass / self.M0,
            radius=tr.radius / self.L0,
            inertia=tr.inertia / self.I0,
            dt=tr.dt / self.T0,
            t0=tr.t0 / self.T0,
            name=tr.name,
            dataset_key=tr.dataset_key,
            schema_variant=tr.schema_variant,
            velocity_from_dem=tr.velocity_from_dem,
            dimensionless=True,
        )

    def redim_momentum(self, dp: torch.Tensor) -> torch.Tensor:
        return dp * self.P0

    def as_dict(self) -> dict[str, float]:
        d = asdict(self)
        d.update(
            V0=self.V0, W0=self.W0, P0=self.P0, LP0=self.LP0,
            F0=self.F0, E0=self.E0, PSI0=self.PSI0, I0=self.I0,
        )
        return d


def default_scales() -> Scales:
    """Escalas oficiales de v3 para el dataset Dynami-CAL.

    `L0` = diámetro de partícula (5 mm), `M0` = masa de una partícula,
    `T0 = 1e-3 s`. Con `dt = 1e-4 s` el paso adimensional es `dt' = 0.1` y las
    velocidades típicas quedan O(1) — el rango donde `softplus`/`sigmoid`
    tienen gradiente útil. Ver `docs/v3/DECISIONS.md` D-013 para por qué esto
    difiere de `T0 = 0.01 s` del legacy.
    """
    d = DEFAULT_DIAMETER
    m = DEFAULT_DENSITY * (math.pi / 6.0) * d**3
    return Scales(L0=d, M0=m, T0=1e-3)
