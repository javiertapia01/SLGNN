"""Configuración de los baselines GNS."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass
class GNSConfig:
    """Presupuesto de capacidad comparable al de v3, siempre reportado."""

    hidden: int = 64
    n_message_steps: int = 4
    n_material_types: int = 4
    material_dim: int = 8
    activation: str = "silu"

    # Grafo: mismos cortes que v3 para que el vecindario sea idéntico.
    pp_gap_off: float = 0.35
    pw_gap_off: float = 0.35
    skin: float = 0.15

    # Normalización de la salida: el decoder predice incrementos escalados.
    output_scale: float = 1.0

    # Solo para la configuración clásica reducida.
    history_length: int = 6

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict | None) -> "GNSConfig":
        raw = dict(raw or {})
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(
                f"Claves desconocidas en la configuración de GNS: {sorted(unknown)}"
            )
        return cls(**raw)
