"""Carga de datos y armado del banco de pruebas, compartido por los experimentos.

Mismo papel que `slgnn.experiment` para el track de entrenamiento: las
decisiones ya verificadas contra los datos reales (eje de gravedad, perfil de
omega, geometría del cilindro) viven en un solo lugar en vez de duplicarse en
cada script y arriesgar que uno quede desincronizado del otro.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from slgnn.data import default_scales, load_case
from slgnn.sdf import BoxSDF

from .coarse import CoarseConfig, calibrate_dissipation, coarse_grain_windows
from .cylinder import DT_RECORD, PROFILES, make_cylinder_sdf
from .library import DOWN, FLAT, UP, SpectrumLibrary
from .macro import PBMConfig
from .units import Scaling

_AXIS = {"x": 0, "y": 1, "z": 2}


def load_config(path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_scaling() -> Scaling:
    return Scaling.from_slgnn(default_scales())


def _gravity(cfg, scales, dtype=torch.float64) -> np.ndarray:
    g = np.zeros(3)
    g[_AXIS[cfg["data"]["gravity_axis"]]] = -scales.gravity(float(cfg["data"]["gravity"]))
    return g


def coarse_config(cfg, kappa: float | None = None) -> CoarseConfig:
    c = dict(cfg["coarse"])
    c.pop("kappa_diss", None)
    out = CoarseConfig(**c)
    return replace(out, kappa_diss=kappa) if kappa is not None else out


def pbm_config(cfg) -> PBMConfig:
    p = dict(cfg["pbm"])
    p["feed_fractions"] = tuple(p["feed_fractions"])
    return PBMConfig(**p)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def load_cylinder(cfg, root: Path | None = None):
    """Trayectoria adimensional del cilindro + SDF con la cinemática verificada."""
    root = root or project_root()
    scales = default_scales()
    scaling = Scaling.from_slgnn(scales)
    path = root / cfg["data"]["root"] / cfg["data"]["cylinder_case"]
    traj = scales.nondim(load_case(path, dt=float(cfg["data"]["cylinder_dt"]),
                                   dtype=torch.float64))
    profile = cfg["data"].get("omega_profile", "pdf_literal")
    if profile not in PROFILES:
        raise ValueError(f"omega_profile desconocido: {profile}")
    sdf = make_cylinder_sdf(scaling, profile)
    return traj, sdf, _gravity(cfg, scales), scaling, profile


def load_calibration_case(cfg, root: Path | None = None):
    """Caja estática con gravedad: el único caso donde `p_in ≡ 0` es exacto."""
    root = root or project_root()
    scales = default_scales()
    scaling = Scaling.from_slgnn(scales)
    path = root / cfg["data"]["root"] / cfg["data"]["calibration_case"]
    traj = scales.nondim(load_case(path, dt=float(cfg["data"]["calibration_dt"]),
                                   dtype=torch.float64))
    side = scales.length(float(cfg["data"]["calibration_box_m"]))
    return traj, BoxSDF([0.0] * 3, [side] * 3), _gravity(cfg, scales), scaling


def resolve_kappa(cfg, root: Path | None = None, n_windows: int = 4,
                  window: int = 300) -> dict:
    """Devuelve `kappa_diss`, calibrándolo si el config lo deja en null.

    La transferencia de la caja estática al cilindro es defendible —comparten
    los parámetros de contacto (DATA_NOTES.md §5)— pero **es una transferencia**:
    todo `p_diss` que la use es un valor calibrado, no medido.
    """
    fixed = cfg["coarse"].get("kappa_diss")
    if fixed is not None:
        return {"kappa": float(fixed), "source": "config", "residual": float("nan")}
    traj, wall, g, scaling = load_calibration_case(cfg, root)
    windows = [(a, a + window) for a in range(0, n_windows * window, window)]
    result = calibrate_dissipation(traj, wall, scaling, coarse_config(cfg), windows,
                                   g_vec=g)
    result["source"] = f"calibrado en {cfg['data']['calibration_case']}"
    return result


# ---------------------------------------------------------------------------
# Segmentación de la rampa
# ---------------------------------------------------------------------------

def ramp_windows(n_steps: int, dt_nondim: float, scaling: Scaling, width_s: float):
    """Ventanas contiguas de ancho fijo en segundos."""
    width = max(int(round(width_s / (dt_nondim * scaling.T))), 3)
    return [(a, min(a + width, n_steps)) for a in range(0, n_steps - 2, width)
            if min(a + width, n_steps) - a >= 3]


def branch_from_profile(profile: str, t_mid_s: float, dt_s: float = 1e-3) -> str:
    """Rama de la rampa a partir del perfil analítico, no de una regresión.

    El perfil se conoce en forma cerrada, así que clasificar por su derivada es
    exacto y no depende del ancho de ventana.
    """
    fn = PROFILES[profile]
    lo, hi = fn(max(t_mid_s - dt_s, 0.0)), fn(t_mid_s + dt_s)
    if abs(hi - lo) < 1e-12:
        return FLAT
    return UP if hi > lo else DOWN


def coarse_ramp(traj, sdf, scaling, cfg_coarse: CoarseConfig, g_vec, width_s: float,
                profile: str, events=None):
    """`C_phi` sobre toda la rampa, en ventanas de ancho `width_s`."""
    n_steps = int(traj.q.shape[0])
    windows = ramp_windows(n_steps, float(traj.dt), scaling, width_s)
    feats = coarse_grain_windows(traj, sdf, scaling, cfg_coarse, windows,
                                 g_vec=g_vec, events=events)
    for f in feats:
        f.branch = branch_from_profile(profile, 0.5 * (f.t_start + f.t_end))
        f.meta["window_width_s"] = width_s
    return feats


def build_library(features, min_events: int = 1) -> SpectrumLibrary:
    return SpectrumLibrary(min_events=min_events).fit(features)


def positive_omega_features(features):
    """Subconjunto con omega > 0 — el régimen de molienda hacia adelante.

    El cilindro invierte el giro después de t = 1 s (ver `twin.cylinder`). Para
    el lazo de control eso no es una acción admisible, así que la biblioteca de
    control se construye solo sobre la mitad de avance.
    """
    return [f for f in features if f.omega > 0]
