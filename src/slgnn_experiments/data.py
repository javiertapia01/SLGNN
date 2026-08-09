"""Lectura del dataset Dynami-CAL, neutral respecto de la arquitectura.

Reimplementa el loader contra el esquema ya verificado en `data/DATA_NOTES.md`,
con las garantías que exige la §3.3 de las instrucciones de implementación:

- lectura **siempre por nombre de cabecera** (3 variantes reales de esquema);
- orden de fila estable como identidad de partícula cuando falta `Particle_ID`;
- `dt` y timestamps reales por dataset (no todos comparten `dt`);
- `q`, `v` y `omega` originales del DEM, sin filtrar;
- masa, radio e inercia por partícula;
- `Orientation:*` descartada explícitamente;
- caché **versionada por esquema**: si `_SCHEMA_VERSION` cambia, las cachés
  viejas se ignoran en vez de devolver silenciosamente un tensor incompatible.

No se hardcodea que todos los datasets compartan el mismo wrapper de
directorio: el registro `DATASETS` guarda la ruta real de cada uno (el archivo
homogéneo tiene un nivel `DATA/` extra que los demás no tienen).
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

# Subir esta versión invalida todas las cachés en disco.
_SCHEMA_VERSION = 1
_CACHE_NAME = f"_slgnn_v3_cache_v{_SCHEMA_VERSION}.npz"

_FILE_RE = re.compile(r"data_at_timestep_(\d+)\.csv$")

_KNOWN_COLUMNS = frozenset({
    "Diameter", "Density", "Particle_ID",
    "Velocity:0", "Velocity:1", "Velocity:2",
    "Angular_velocity:0", "Angular_velocity:1", "Angular_velocity:2",
    "Orientation:0", "Orientation:1", "Orientation:2",
    "coordinates:0", "coordinates:1", "coordinates:2",
})

DEFAULT_DIAMETER = 0.005   # m, constante verificada en los 6 archivos
DEFAULT_DENSITY = 4000.0   # kg/m^3


@dataclass(frozen=True)
class DatasetSpec:
    """Ruta real y metadatos físicos de un archivo del dataset."""

    key: str
    subpath: str            # relativo a data/extracted, incluye el wrapper real
    cases: tuple[str, ...]
    dt: float               # segundos entre snapshots grabados
    gravity: float          # m/s^2, 0.0 si el caso no tiene gravedad
    gravity_axis: str | None
    geometry: str           # "box" | "cylinder" | "none"
    box_min: tuple[float, float, float] | None = None
    box_max: tuple[float, float, float] | None = None
    notes: str = ""


# El eje de gravedad es -y en los datasets de caja y cilindro: verificado
# empíricamente contra los datos (data/DATA_NOTES.md, sección "Pendiente").
DATASETS: dict[str, DatasetSpec] = {
    "two_spheres": DatasetSpec(
        key="two_spheres",
        subpath="Benchmark_2Spheres_Oblique_Collision",
        cases=("1x", "2x", "4x"),
        dt=1e-4,
        gravity=0.0,
        gravity_axis=None,
        geometry="none",
        notes="Colision oblicua sin gravedad ni paredes. 100 snapshots, 2 particulas.",
    ),
    "one_sphere_wall": DatasetSpec(
        key="one_sphere_wall",
        subpath="Benchmark_1Sphere_Multiple_Wall_Collision",
        cases=("10", "30", "45", "60", "90"),
        dt=1e-4,
        gravity=0.0,
        gravity_axis=None,
        geometry="box",
        # Pared plana en z=0; se modela como semiespacio via una sola cara.
        box_min=(-1.0, -1.0, 0.0),
        box_max=(1.0, 1.0, 1.0),
        notes="Angulo de impacto en grados como nombre de caso. 200 snapshots, 1 particula.",
    ),
    "sixty_homogeneous": DatasetSpec(
        key="sixty_homogeneous",
        subpath="60Spheres_Homogeneous_Interaction_Inside_Cuboidal_Enclosure/DATA",
        cases=tuple(f"CASE{i:02d}" for i in range(1, 10)),
        dt=1e-4,
        gravity=0.0,
        gravity_axis=None,
        geometry="box",
        box_min=(0.0, 0.0, 0.0),
        box_max=(0.03, 0.03, 0.03),
        notes="Wrapper DATA/ extra. CASE08-09 no documentados en el PDF fuente.",
    ),
    "sixty_gravity": DatasetSpec(
        key="sixty_gravity",
        subpath="60Spheres_Gravity_Inside_Cuboidal_Enclosure",
        cases=tuple(f"CASE{i:02d}" for i in range(1, 8)),
        dt=1e-4,
        gravity=9.81,
        gravity_axis="y",
        geometry="box",
        box_min=(0.0, 0.0, 0.0),
        box_max=(0.03, 0.03, 0.03),
        notes="Sin wrapper DATA/. CASE07 es extrapolacion: nunca en seleccion.",
    ),
    "rotating_cylinder": DatasetSpec(
        key="rotating_cylinder",
        subpath="Extrapolation_2073Spheres_Gravity_Inside_Rotating_Cylinder",
        cases=("CASE08",),
        dt=1e-3,
        gravity=9.81,
        gravity_axis="y",
        geometry="cylinder",
        notes="Solo inferencia. dt de grabacion 1e-3, no 1e-4.",
    ),
}


@dataclass
class Trajectory:
    """Una trayectoria DEM completa, en unidades físicas o adimensionales.

    `dimensionless` documenta en qué sistema están los tensores: la
    adimensionalización se aplica una sola vez y se registra, nunca se aplica
    dos veces por accidente.
    """

    q: torch.Tensor          # [T, N, 3] posiciones del centro
    v: torch.Tensor          # [T, N, 3] velocidades lineales (DEM originales)
    omega: torch.Tensor      # [T, N, 3] velocidades angulares (DEM originales)
    mass: torch.Tensor       # [N]
    radius: torch.Tensor     # [N]
    inertia: torch.Tensor    # [N] esfera maciza homogénea, 2/5 m R^2
    dt: float                # entre snapshots consecutivos
    t0: float = 0.0          # tiempo real del snapshot 0
    name: str = ""
    dataset_key: str = ""
    schema_variant: str = "?"
    velocity_from_dem: bool = True
    dimensionless: bool = False

    @property
    def n_steps(self) -> int:
        return int(self.q.shape[0])

    @property
    def n_particles(self) -> int:
        return int(self.q.shape[1])

    def times(self) -> torch.Tensor:
        """Timestamps reales [T]. Nunca se sustituye por t = 0."""
        return self.t0 + self.dt * torch.arange(
            self.n_steps, dtype=self.q.dtype, device=self.q.device
        )

    def to(self, dtype=None, device=None) -> "Trajectory":
        cast = lambda x: x.to(dtype=dtype or x.dtype, device=device or x.device)
        return Trajectory(
            q=cast(self.q), v=cast(self.v), omega=cast(self.omega),
            mass=cast(self.mass), radius=cast(self.radius), inertia=cast(self.inertia),
            dt=self.dt, t0=self.t0, name=self.name, dataset_key=self.dataset_key,
            schema_variant=self.schema_variant,
            velocity_from_dem=self.velocity_from_dem,
            dimensionless=self.dimensionless,
        )


def dataset_root(repo_root: Path | str) -> Path:
    return Path(repo_root) / "data" / "extracted"


def resolve_case_dir(dataset_key: str, case: str, repo_root: Path | str) -> Path:
    """Ruta real de un caso. Falla con mensaje explícito si no existe."""
    if dataset_key not in DATASETS:
        raise KeyError(
            f"Dataset desconocido {dataset_key!r}. Conocidos: {sorted(DATASETS)}"
        )
    spec = DATASETS[dataset_key]
    if case not in spec.cases:
        raise KeyError(f"Caso {case!r} no listado para {dataset_key!r}: {spec.cases}")
    path = dataset_root(repo_root) / spec.subpath / case
    if not path.is_dir():
        raise FileNotFoundError(
            f"No existe {path}. Descomprime data/raw/{spec.subpath.split('/')[0]}.zip"
        )
    return path


def list_cases(dataset_key: str, repo_root: Path | str) -> list[str]:
    """Casos del registro que existen realmente en disco."""
    spec = DATASETS[dataset_key]
    base = dataset_root(repo_root) / spec.subpath
    return [c for c in spec.cases if (base / c).is_dir()]


def _timestep_files(case_dir: Path) -> list[Path]:
    found: list[tuple[int, Path]] = []
    for p in case_dir.iterdir():
        match = _FILE_RE.search(p.name)
        if match:
            found.append((int(match.group(1)), p))
    if not found:
        raise FileNotFoundError(f"Sin data_at_timestep_*.csv en {case_dir}")
    found.sort(key=lambda kv: kv[0])
    steps = [k for k, _ in found]
    if steps != list(range(len(steps))):
        missing = sorted(set(range(steps[-1] + 1)) - set(steps))[:5]
        raise ValueError(f"Timesteps no contiguos en {case_dir}; faltan {missing}...")
    return [p for _, p in found]


def _schema_variant(fieldnames: list[str]) -> str:
    has_d = "Diameter" in fieldnames
    has_id = "Particle_ID" in fieldnames
    if has_d and has_id:
        return "A"
    if has_id:
        return "B"
    if has_d:
        return "C"
    return "D"  # ni Diameter ni Particle_ID: no observada, pero manejable


def _read_frame(path: Path, default_diameter: float):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        names = list(reader.fieldnames or [])
        unknown = set(names) - _KNOWN_COLUMNS
        if unknown:
            raise ValueError(
                f"Cabecera no reconocida en {path}: {sorted(unknown)}. "
                "Actualiza _KNOWN_COLUMNS y revisa data/DATA_NOTES.md antes de seguir."
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV vacío: {path}")
    variant = _schema_variant(names)

    def col(name: str) -> np.ndarray:
        return np.array([float(r[name]) for r in rows], dtype=np.float64)

    q = np.stack([col("coordinates:0"), col("coordinates:1"), col("coordinates:2")], -1)
    v = np.stack([col("Velocity:0"), col("Velocity:1"), col("Velocity:2")], -1)
    w = np.stack([col("Angular_velocity:0"), col("Angular_velocity:1"),
                  col("Angular_velocity:2")], -1)
    density = col("Density") if "Density" in names else np.full(len(rows), DEFAULT_DENSITY)
    diameter = col("Diameter") if "Diameter" in names else np.full(len(rows), default_diameter)

    if "Particle_ID" in names:
        # Orden canónico por ID: la identidad de partícula no puede depender del
        # orden en que MFiX volcó las filas de un snapshot concreto.
        order = np.argsort(col("Particle_ID"), kind="stable")
        q, v, w, density, diameter = (
            q[order], v[order], w[order], density[order], diameter[order]
        )
    # Sin Particle_ID la identidad es el índice de fila (verificado estable,
    # DATA_NOTES.md §5): se preserva el orden de lectura tal cual.
    return q, v, w, diameter, density, variant


def load_case(
    dataset_key: str,
    case: str,
    repo_root: Path | str,
    *,
    cache: bool = True,
    dtype: torch.dtype = torch.float64,
    max_steps: int | None = None,
) -> Trajectory:
    """Carga una carpeta CASE a una `Trajectory` en unidades físicas (SI)."""
    spec = DATASETS[dataset_key]
    case_dir = resolve_case_dir(dataset_key, case, repo_root)
    cache_path = case_dir / _CACHE_NAME

    if cache and cache_path.exists():
        z = np.load(cache_path, allow_pickle=False)
        q, v, w = z["q"], z["v"], z["w"]
        diameter, density = z["diameter"], z["density"]
        variant = str(z["variant"].item()) if "variant" in z else "?"
    else:
        files = _timestep_files(case_dir)
        frames = [_read_frame(p, DEFAULT_DIAMETER) for p in files]
        variants = {fr[5] for fr in frames}
        if len(variants) > 1:
            raise ValueError(f"Esquema inconsistente dentro de {case_dir}: {variants}")
        variant = variants.pop()
        q = np.stack([fr[0] for fr in frames])
        v = np.stack([fr[1] for fr in frames])
        w = np.stack([fr[2] for fr in frames])
        diameter, density = frames[0][3], frames[0][4]
        if cache:
            np.savez_compressed(
                cache_path, q=q, v=v, w=w, diameter=diameter, density=density,
                variant=np.array(variant),
            )

    if max_steps is not None:
        q, v, w = q[:max_steps], v[:max_steps], w[:max_steps]

    radius = diameter / 2.0
    mass = density * (math.pi / 6.0) * diameter**3
    inertia = 0.4 * mass * radius**2  # esfera maciza homogénea, I = 2/5 m R^2

    to = lambda arr: torch.as_tensor(np.ascontiguousarray(arr), dtype=dtype)
    tr = Trajectory(
        q=to(q), v=to(v), omega=to(w),
        mass=to(mass), radius=to(radius), inertia=to(inertia),
        dt=spec.dt, t0=0.0, name=f"{dataset_key}/{case}",
        dataset_key=dataset_key, schema_variant=variant,
    )
    validate_trajectory(tr)
    return tr


def slice_frames(tr: Trajectory, start: int = 0, stop: int | None = None) -> Trajectory:
    """Ventana `[start, stop)` de una trayectoria, conservando el tiempo real.

    Necesario porque la física no está donde empieza el archivo: la auditoría
    temporal muestra que en `sixty_gravity/CASE01` el primer contacto aparece
    cerca del snapshot 190, así que un prefijo `[0, 120)` no contiene ningún
    contacto y un micro-overfit sobre él solo aprendería caída libre.
    """
    stop = tr.n_steps if stop is None else min(stop, tr.n_steps)
    if not 0 <= start < stop:
        raise ValueError(f"{tr.name}: ventana [{start}, {stop}) inválida "
                         f"para {tr.n_steps} snapshots")
    out = Trajectory(
        q=tr.q[start:stop], v=tr.v[start:stop], omega=tr.omega[start:stop],
        mass=tr.mass, radius=tr.radius, inertia=tr.inertia, dt=tr.dt,
        t0=tr.t0 + start * tr.dt, name=f"{tr.name}[{start}:{stop}]",
        dataset_key=tr.dataset_key, schema_variant=tr.schema_variant,
        velocity_from_dem=tr.velocity_from_dem, dimensionless=tr.dimensionless,
    )
    validate_trajectory(out)
    return out


def validate_trajectory(tr: Trajectory) -> None:
    """Invariantes que deben cumplirse antes de que un tensor entre al modelo."""
    T, N = tr.n_steps, tr.n_particles
    for name, x, shape in (
        ("q", tr.q, (T, N, 3)), ("v", tr.v, (T, N, 3)), ("omega", tr.omega, (T, N, 3)),
        ("mass", tr.mass, (N,)), ("radius", tr.radius, (N,)), ("inertia", tr.inertia, (N,)),
    ):
        if tuple(x.shape) != shape:
            raise ValueError(f"{tr.name}: {name} tiene shape {tuple(x.shape)}, esperado {shape}")
        if not torch.isfinite(x).all():
            raise ValueError(f"{tr.name}: {name} contiene NaN/Inf")
    for name, x in (("mass", tr.mass), ("radius", tr.radius), ("inertia", tr.inertia)):
        if not bool((x > 0).all()):
            raise ValueError(f"{tr.name}: {name} debe ser estrictamente positivo")
    if tr.dt <= 0:
        raise ValueError(f"{tr.name}: dt = {tr.dt} no es positivo")
