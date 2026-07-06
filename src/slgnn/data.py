"""Loader del dataset Dynami-CAL (M2), robusto a las 3 variantes de esquema.

Ver data/DATA_NOTES.md: el parser lee SIEMPRE por nombre de cabecera. Si falta
`Diameter` usa el valor por defecto del dataset (0.005 m); si falta
`Particle_ID` usa el orden de fila como identidad (verificado estable).
`Orientation:0-2` se descarta (no fiable según el PDF de origen).
"""

import csv
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

_FILE_RE = re.compile(r"data_at_timestep_(\d+)\.csv$")

_KNOWN_COLUMNS = {
    "Diameter", "Density", "Particle_ID",
    "Velocity:0", "Velocity:1", "Velocity:2",
    "Angular_velocity:0", "Angular_velocity:1", "Angular_velocity:2",
    "Orientation:0", "Orientation:1", "Orientation:2",
    "coordinates:0", "coordinates:1", "coordinates:2",
}


@dataclass
class Trajectory:
    q: torch.Tensor       # [T,N,3] posiciones
    v: torch.Tensor       # [T,N,3] velocidades
    omega: torch.Tensor   # [T,N,3] velocidades angulares
    m: torch.Tensor       # [N] masas
    radii: torch.Tensor   # [N] radios
    dt: float


def _timestep_files(case_dir: Path) -> list[Path]:
    found = []
    for p in case_dir.iterdir():
        match = _FILE_RE.search(p.name)
        if match:
            found.append((int(match.group(1)), p))
    found.sort(key=lambda kv: kv[0])
    steps = [k for k, _ in found]
    if steps != list(range(len(steps))):
        raise ValueError(f"Timesteps no contiguos en {case_dir}: {steps[:5]}...")
    return [p for _, p in found]


def _read_frame(path: Path, default_diameter: float):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        unknown = set(reader.fieldnames or []) - _KNOWN_COLUMNS
        if unknown:
            raise ValueError(f"Cabecera no reconocida en {path}: {sorted(unknown)}")
        rows = list(reader)
    has_diameter = "Diameter" in rows[0]
    has_id = "Particle_ID" in rows[0]

    def col(name):
        return np.array([float(r[name]) for r in rows])

    q = np.stack([col("coordinates:0"), col("coordinates:1"), col("coordinates:2")], -1)
    v = np.stack([col("Velocity:0"), col("Velocity:1"), col("Velocity:2")], -1)
    w = np.stack(
        [col("Angular_velocity:0"), col("Angular_velocity:1"), col("Angular_velocity:2")],
        -1,
    )
    density = col("Density")
    diameter = col("Diameter") if has_diameter else np.full(len(rows), default_diameter)
    if has_id:
        order = np.argsort(col("Particle_ID"))
        q, v, w, density, diameter = (
            q[order], v[order], w[order], density[order], diameter[order]
        )
    return q, v, w, diameter, density


def load_case(case_dir, dt: float, default_diameter: float = 0.005,
              cache: bool = True, dtype=torch.float32) -> Trajectory:
    """Carga una carpeta CASE completa a tensores [T, N, 3]."""
    case_dir = Path(case_dir)
    cache_path = case_dir / "_slgnn_cache.npz"
    if cache and cache_path.exists():
        z = np.load(cache_path)
        q, v, w, diameter, density = z["q"], z["v"], z["w"], z["diameter"], z["density"]
    else:
        files = _timestep_files(case_dir)
        if not files:
            raise FileNotFoundError(f"Sin archivos data_at_timestep_*.csv en {case_dir}")
        frames = [_read_frame(p, default_diameter) for p in files]
        q = np.stack([fr[0] for fr in frames])
        v = np.stack([fr[1] for fr in frames])
        w = np.stack([fr[2] for fr in frames])
        diameter, density = frames[0][3], frames[0][4]
        if cache:
            np.savez_compressed(
                cache_path, q=q, v=v, w=w, diameter=diameter, density=density
            )
    radii = diameter / 2.0
    m = density * (math.pi / 6.0) * diameter**3
    to = lambda arr: torch.as_tensor(arr, dtype=dtype)
    return Trajectory(q=to(q), v=to(v), omega=to(w), m=to(m), radii=to(radii), dt=dt)


def finite_difference_accelerations(x: torch.Tensor, dt: float) -> torch.Tensor:
    """Targets a_k = (v_{k+1} - v_k)/dt, consistente con el update de velocidad
    del Euler semiimplícito (§31). Devuelve [T-1, N, 3]."""
    return (x[1:] - x[:-1]) / dt


@dataclass
class Scales:
    """Adimensionalización (§36): longitud L0, tiempo T0, masa M0."""

    L0: float
    T0: float
    M0: float

    def nondim(self, tr: Trajectory) -> Trajectory:
        return replace(
            tr,
            q=tr.q / self.L0,
            v=tr.v * (self.T0 / self.L0),
            omega=tr.omega * self.T0,
            m=tr.m / self.M0,
            radii=tr.radii / self.L0,
            dt=tr.dt / self.T0,
        )

    def gravity(self, g: float = 9.81) -> float:
        return g * self.T0**2 / self.L0

    def length(self, x: float) -> float:
        return x / self.L0


def default_scales() -> Scales:
    """Escalas del dataset Dynami-CAL: L0 = diámetro de partícula (0.005 m),
    T0 = 0.01 s (100 pasos de modelo), M0 = masa de una partícula."""
    d = 0.005
    m = 4000.0 * (math.pi / 6.0) * d**3
    return Scales(L0=d, T0=0.01, M0=m)
