"""Señales de confianza (§7.7).

Dos señales, ambas baratas y ambas significativas en este banco de pruebas:

- `gamma_ood`: omega fuera del rango cubierto por la rampa, o descriptores de
  carga fuera de la envolvente observada;
- `sigma_spec`: discrepancia entre la rama ascendente y la descendente de la
  biblioteca a esa omega.

La segunda es lo interesante: **la histéresis medida en el experimento H se
convierte directamente en la incertidumbre epistémica del espectro**, sin
ensambles ni entrenamiento adicional. Si H concluye que `omega ↦ E^coll` es una
función, `sigma_spec` es pequeña y el MPC opera libre; si no lo es, la misma
cifra que lo demuestra es la que restringe al controlador.
"""

from dataclasses import dataclass, field

import numpy as np

from .coarse import MacroFeatures
from .library import Regime, SpectrumLibrary


@dataclass
class Envelope:
    """Envolvente de los descriptores de carga observados en los datos."""

    omega: tuple[float, float]
    com_r: tuple[float, float]
    ke: tuple[float, float]

    @classmethod
    def from_features(cls, feats: list[MacroFeatures]) -> "Envelope":
        def rng(values):
            v = np.array([x for x in values if np.isfinite(x)])
            return (float(v.min()), float(v.max())) if v.size else (np.nan, np.nan)

        return cls(
            omega=rng(f.omega for f in feats),
            com_r=rng(f.com_r for f in feats),
            ke=rng(f.ke for f in feats),
        )


def _outside(value: float, bounds: tuple[float, float]) -> float:
    """Distancia relativa fuera de un intervalo; 0 si está dentro."""
    lo, hi = bounds
    if not np.isfinite(value) or not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0
    span = max(hi - lo, 1e-30)
    if value < lo:
        return (lo - value) / span
    if value > hi:
        return (value - hi) / span
    return 0.0


@dataclass
class ConfidenceReport:
    gamma_ood: float
    sigma_spec: float
    sigma_rate: float
    triggered: bool
    reasons: list = field(default_factory=list)


@dataclass
class ConfidenceMonitor:
    """Vigila la consulta a la biblioteca y cablea la acción de §7.7.

    La acción está deliberadamente cableada aunque el detector sea crudo: si
    `gamma_ood` se dispara, el MPC restringe omega al rango cubierto y reduce
    `delta_omega_max`. Un interruptor tosco que existe es preferible a uno fino
    que no.
    """

    envelope: Envelope
    ood_threshold: float = 0.02
    sigma_threshold: float = 0.15   # décadas de Wasserstein entre ramas
    shrink: float = 0.25            # factor sobre delta_omega_max al dispararse

    @classmethod
    def from_library(cls, lib: SpectrumLibrary, **kwargs) -> "ConfidenceMonitor":
        return cls(envelope=Envelope.from_features(lib.nodes), **kwargs)

    def evaluate(self, regime: Regime, state_com_r: float = np.nan) -> ConfidenceReport:
        reasons = []
        gamma = _outside(regime.omega, self.envelope.omega)
        if regime.ood or gamma > self.ood_threshold:
            reasons.append(f"omega={regime.omega:.3f} fuera de {self.envelope.omega}")
        gamma_load = _outside(state_com_r, self.envelope.com_r)
        if gamma_load > self.ood_threshold:
            reasons.append(f"com_r={state_com_r:.5f} fuera de {self.envelope.com_r}")
        gamma = max(gamma, gamma_load)

        sigma = regime.sigma_spec if np.isfinite(regime.sigma_spec) else 0.0
        if sigma > self.sigma_threshold:
            reasons.append(f"histéresis sigma_spec={sigma:.3f} décadas")

        return ConfidenceReport(
            gamma_ood=float(gamma), sigma_spec=float(sigma),
            sigma_rate=float(regime.sigma_rate if np.isfinite(regime.sigma_rate) else 0.0),
            triggered=bool(reasons), reasons=reasons,
        )

    def restrict(self, bounds: tuple[float, float], delta_max: float,
                 report: ConfidenceReport):
        """Acción cableada: recortar omega al rango cubierto y frenar su tasa."""
        if not report.triggered:
            return bounds, delta_max
        lo, hi = self.envelope.omega
        safe = (max(bounds[0], lo), min(bounds[1], hi))
        if safe[0] > safe[1]:
            safe = (lo, hi)
        return safe, delta_max * self.shrink
