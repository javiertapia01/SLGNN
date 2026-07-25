"""Frontera de unidades del gemelo (§6 de las instrucciones).

Regla única, y no negociable porque su violación no rompe nada — solo desplaza
silenciosamente las tasas de rotura del PBM:

    `C_phi` recibe adimensional y entrega dimensional (SI).
    Ninguna otra frontera del paquete `twin` convierte unidades.

`slgnn.data.Scales` adimensionaliza para la red; `Scaling` es su contraparte
para el camino de vuelta. Se construye una vez desde los metadatos del dataset
y se serializa junto a todo resultado.
"""

from dataclasses import asdict, dataclass

from slgnn.data import Scales, default_scales


@dataclass(frozen=True)
class Scaling:
    """Escalas de referencia para volver de adimensional a SI."""

    L: float  # longitud de referencia [m]
    T: float  # tiempo de referencia [s]
    M: float  # masa de referencia [kg]

    @property
    def velocity(self) -> float:
        return self.L / self.T

    @property
    def energy(self) -> float:
        return self.M * self.L**2 / self.T**2

    @property
    def power(self) -> float:
        return self.energy / self.T

    @property
    def rate(self) -> float:
        """Inversa del tiempo: convierte tasas adimensionales a s^-1."""
        return 1.0 / self.T

    @property
    def spectral_rate(self) -> float:
        """Tasa por unidad de masa: eventos/(s·kg) desde eventos/(T·M)."""
        return 1.0 / (self.T * self.M)

    @classmethod
    def from_slgnn(cls, scales: Scales) -> "Scaling":
        return cls(L=scales.L0, T=scales.T0, M=scales.M0)

    @classmethod
    def from_dataset(cls) -> "Scaling":
        """Escalas del dataset Dynami-CAL, las mismas que usa el entrenamiento."""
        return cls.from_slgnn(default_scales())

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(energy=self.energy, power=self.power, velocity=self.velocity)
        return d


def nondim_omega_fn(omega_fn, scaling: Scaling):
    """Envuelve un omega(t) en SI para que opere en tiempo/frecuencia adimensional.

    El integrador y las SDF del rollout trabajan en unidades adimensionales, de
    modo que una `RotatingCylinderSDF` construida con geometría adimensional
    necesita omega' (t') = omega(t' · T) · T.
    """

    def _omega(t_nondim: float) -> float:
        return omega_fn(t_nondim * scaling.T) * scaling.T

    return _omega
