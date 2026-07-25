"""Métricas, comparaciones y reportes (§5).

La distancia entre espectros es 1-Wasserstein sobre el eje de **log-energía**,
que es el eje natural: los bins son log-espaciados y lo que importa de un
espectro es en qué década cae la masa de eventos, no su separación lineal en
joules.
"""

import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

from .coarse import MacroFeatures, Spectrum


def enable_utf8_stdout() -> None:
    """La consola de Windows usa cp1252 y no puede imprimir ≡, ∈, ↦ ni ². Los
    reportes de los experimentos están llenos de esos símbolos, así que se
    reconfigura la salida en vez de empobrecer el texto."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):  # pragma: no cover - depende del entorno
            pass


def wasserstein(a: Spectrum, b: Spectrum) -> float:
    """1-Wasserstein entre espectros normalizados, en décadas de energía.

    Requiere el mismo binning: comparar espectros con bordes distintos sería
    comparar dos objetos diferentes, y el error sería silencioso.
    """
    if a.edges.shape != b.edges.shape or not np.allclose(a.edges, b.edges):
        raise ValueError("los espectros deben compartir el binning")
    pa, pb = a.normalized(), b.normalized()
    if pa.sum() == 0 or pb.sum() == 0:
        return float("nan")
    widths = np.diff(np.log10(a.edges))
    cdf = np.cumsum(pa - pb)
    return float(np.abs(cdf[:-1] * widths[:-1]).sum())


def relative_rate_error(a: Spectrum, b: Spectrum) -> float:
    """Error relativo de la tasa total. Complementa a Wasserstein: la distancia
    entre distribuciones normalizadas es ciega a la magnitud."""
    ra, rb = a.total_rate(), b.total_rate()
    denom = max(abs(rb), 1e-30)
    return float(abs(ra - rb) / denom)


def spectrum_distance(a: Spectrum, b: Spectrum) -> dict:
    return {
        "wasserstein_decades": wasserstein(a, b),
        "rate_rel_error": relative_rate_error(a, b),
        "mean_energy_rel_error": abs(a.mean_energy() - b.mean_energy())
        / max(b.mean_energy(), 1e-30),
    }


def features_table(feats: list[MacroFeatures], columns=None) -> str:
    """Tabla de texto de una lista de `MacroFeatures`, para logs y reportes."""
    columns = columns or [
        ("t_ini[s]", lambda f: f"{f.t_start:8.4f}"),
        ("t_fin[s]", lambda f: f"{f.t_end:8.4f}"),
        ("rama", lambda f: f"{f.branch:>5}"),
        ("w[rad/s]", lambda f: f"{f.omega:8.3f}"),
        ("n_ev", lambda f: f"{f.n_events:6d}"),
        ("tasa[1/s.kg]", lambda f: f"{f.spectrum.total_rate():11.3e}"),
        ("E_med[J]", lambda f: f"{f.spectrum.mean_energy():9.3e}"),
        ("P_in[W]", lambda f: f"{f.p_in_balance:9.3e}"),
        ("KE[J]", lambda f: f"{f.ke:9.3e}"),
        ("com_r[m]", lambda f: f"{f.com_r:8.5f}"),
    ]
    header = " ".join(f"{name:>11}" for name, _ in columns)
    lines = [header, "-" * len(header)]
    for f in feats:
        lines.append(" ".join(f"{fn(f):>11}" for _, fn in columns))
    return "\n".join(lines)


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def save_report(path, payload: dict) -> Path:
    """Serializa un resultado con su `Scaling` incluido (§6)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return path


def ascii_profile(x, y, width=64, height=12, label="") -> str:
    """Gráfico de línea en texto — evita depender de matplotlib para un log."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size == 0:
        return f"{label}: sin datos"
    grid = [[" "] * width for _ in range(height)]
    xs = np.clip(((x - x.min()) / max(np.ptp(x), 1e-30) * (width - 1)).astype(int),
                 0, width - 1)
    ys = np.clip(((y - y.min()) / max(np.ptp(y), 1e-30) * (height - 1)).astype(int),
                 0, height - 1)
    for i, j in zip(xs, ys):
        grid[height - 1 - j][i] = "*"
    body = "\n".join("|" + "".join(row) for row in grid)
    return (f"{label}\n{body}\n+{'-' * width}\n"
            f" x: [{x.min():.4g}, {x.max():.4g}]   y: [{y.min():.4g}, {y.max():.4g}]")
