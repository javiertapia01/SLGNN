"""Utilidades compartidas entre scripts de entrenamiento y evaluación.

Centraliza la carga de datos/config para que decisiones ya verificadas contra
el dataset real (p. ej. el eje de gravedad) vivan en un solo lugar, en vez de
duplicarse y arriesgar que un script quede desincronizado del otro.
"""

from pathlib import Path

import torch

from .data import default_scales, finite_difference_accelerations, load_case
from .model import SLGNN
from .config import SLGNNConfig
from .sdf import BoxSDF
from .state import Particles

_AXIS = {"x": 0, "y": 1, "z": 2}


def load_split(cfg, root: Path):
    """Carga y adimensionaliza train/val, y arma la pared y el vector gravedad.

    El eje de gravedad (`cfg["data"]["gravity_axis"]`) fue verificado
    empíricamente contra los datos reales (las partículas sedimentan en −y,
    no en −z como asumiría un default ingenuo) — ver DATA_NOTES.md /
    Informe_Estrategia_Entrenamiento_SLGNN.md.
    """
    scales = default_scales()
    base = root / "data" / "extracted" / cfg["data"]["dataset"]
    dt = float(cfg["data"]["dt"])

    def _load(case):
        return scales.nondim(load_case(base / case, dt=dt))

    train = [_load(c) for c in cfg["data"]["train_cases"]]
    val = _load(cfg["data"]["val_case"])

    box_min = [scales.length(x) for x in cfg["data"]["box_min"]]
    box_max = [scales.length(x) for x in cfg["data"]["box_max"]]
    wall = BoxSDF(box_min, box_max)

    g_mag = scales.gravity(float(cfg["data"]["gravity"]))
    g_vec = torch.zeros(3, dtype=train[0].q.dtype)
    g_vec[_AXIS[cfg["data"]["gravity_axis"]]] = -g_mag

    particles = Particles.uniform(
        train[0].q.shape[1],
        m=train[0].m[0].item(),
        radius=train[0].radii[0].item(),
        dtype=train[0].q.dtype,
    )
    return scales, train, val, wall, g_vec, particles


def load_case_by_name(cfg, root: Path, case_name: str):
    """Carga (y adimensionaliza) cualquier CASE del dataset del config, no
    solo train/val — para evaluar sobre casos held-out como CASE07."""
    scales = default_scales()
    base = root / "data" / "extracted" / cfg["data"]["dataset"]
    dt = float(cfg["data"]["dt"])
    return scales.nondim(load_case(base / case_name, dt=dt))


def compute_sigmas(train, dt):
    """Escalas de normalización de las pérdidas, derivadas de los datos (§34, §36)."""
    a = torch.cat([finite_difference_accelerations(tr.v, tr.dt).reshape(-1) for tr in train])
    al = torch.cat([finite_difference_accelerations(tr.omega, tr.dt).reshape(-1) for tr in train])
    v = torch.cat([tr.v.reshape(-1) for tr in train])
    w = torch.cat([tr.omega.reshape(-1) for tr in train])
    eps = 1e-6
    return {
        "sigma_a": float(a.std()) + eps,
        "sigma_alpha": float(al.std()) + eps,
        "sigma_q": 1.0,
        "sigma_v": float(v.std()) + eps,
        "sigma_w": float(w.std()) + eps,
    }


def build_model(cfg):
    # float32: el doble backward funciona igual y es ~2x más rápido/liviano que
    # float64 en CPU, lo importante para rollouts largos. Las garantías físicas
    # que exigen float64 se verifican aparte en los tests.
    fields = SLGNNConfig().__dict__
    overrides = {k: v for k, v in (cfg.get("model") or {}).items() if k in fields}
    return SLGNN(SLGNNConfig(**overrides))


def asdict_config(c: SLGNNConfig):
    return dict(c.__dict__)


def load_checkpoint(path: Path):
    """Carga un checkpoint guardado por scripts/train.py y reconstruye el modelo."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = SLGNN(SLGNNConfig(**ck["model_config"]))
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck
