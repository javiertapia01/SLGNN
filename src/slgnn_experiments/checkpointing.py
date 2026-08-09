"""Manifiestos y checkpoints versionados (§19).

Un checkpoint que no dice con qué escalas, qué índices y qué versión de
esquema se produjo no es reproducible. La carga **falla con explicación** ante
una incompatibilidad en vez de cargar pesos en un modelo distinto.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

CHECKPOINT_SCHEMA = 1


def git_state(repo_root: Path) -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root,
                                      text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
                                             cwd=repo_root, text=True,
                                             stderr=subprocess.DEVNULL).strip())
        return sha, dirty
    except Exception:
        return "unknown", False


@dataclass
class RunManifest:
    """Todo lo que hace falta para repetir una corrida."""

    run_id: str
    model: str
    profile: str
    seed: int
    dataset: str
    train_cases: list[str]
    val_case: str | None
    test_cases: list[str]
    n_transitions_available: int
    index_hash: str
    scales: dict[str, float]
    n_parameters: int
    n_trainable: int
    budget_updates: int
    budget_examples: int
    sampler_quotas: dict[str, float]
    sampler_composition: dict[str, int]
    solver_backend: str
    solver_tolerances: dict[str, float]
    git_sha: str = ""
    git_dirty: bool = False
    started_at: str = ""
    duration_seconds: float = 0.0
    python: str = field(default_factory=lambda: sys.version.split()[0])
    torch: str = field(default_factory=lambda: torch.__version__)
    platform: str = field(default_factory=platform.platform)
    device: str = "cpu"
    dtype: str = "float64"
    notes: str = ""

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


class RunDirectory:
    """Estructura de una corrida: `config_resolved.yaml`, `manifest.json`,
    `metrics.jsonl`, `summary.json`, `best.pt`, `last.pt`, `stdout.log`,
    `environment.txt`."""

    def __init__(self, root: Path, run_id: str):
        self.path = Path(root) / run_id
        self.path.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self._metrics = self.path / "metrics.jsonl"

    def log_metrics(self, record: dict[str, Any]) -> None:
        with open(self._metrics, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=float) + "\n")

    def write_config(self, resolved: dict) -> None:
        import yaml
        (self.path / "config_resolved.yaml").write_text(
            yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    def write_summary(self, summary: dict) -> None:
        (self.path / "summary.json").write_text(
            json.dumps(summary, indent=2, default=float), encoding="utf-8"
        )

    def tee_stdout(self):
        """Duplica `stdout` a `stdout.log` mientras dure el contexto.

        §19 exige `stdout.log` en cada corrida: sin él, un mensaje del tipo
        "estrato sin datos, cuota redistribuida" se pierde y el manifiesto no
        basta para reconstruir qué pasó.
        """
        return _Tee(self.path / "stdout.log")

    def write_environment(self) -> None:
        (self.path / "environment.txt").write_text(
            f"python={sys.version}\ntorch={torch.__version__}\n"
            f"platform={platform.platform()}\n", encoding="utf-8",
        )

    def save_checkpoint(self, name: str, model, optimizer=None, scheduler=None,
                        config: dict | None = None, scales: dict | None = None,
                        sampler_state: dict | None = None, extra: dict | None = None):
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "model_class": type(model).__name__,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "config": config or {},
            "scales": scales or {},
            "sampler_state": sampler_state or {},
            "extra": extra or {},
        }
        torch.save(payload, self.path / name)


class _Tee:
    def __init__(self, path: Path):
        self.path = path
        self._file = None
        self._stdout = None

    def __enter__(self):
        self._file = open(self.path, "a", encoding="utf-8")
        self._stdout = sys.stdout
        outer = self

        class _Writer:
            def write(self, data):
                outer._stdout.write(data)
                outer._file.write(data)

            def flush(self):
                outer._stdout.flush()
                outer._file.flush()

        sys.stdout = _Writer()
        return self

    def __exit__(self, *exc):
        sys.stdout = self._stdout
        self._file.close()
        return False


def load_checkpoint(path: Path, expected_class: str | None = None) -> dict:
    """Carga con validación de esquema y de clase de modelo."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    schema = ck.get("schema")
    if schema != CHECKPOINT_SCHEMA:
        raise ValueError(
            f"{path}: esquema de checkpoint {schema}, esperado {CHECKPOINT_SCHEMA}. "
            "El formato cambió de forma incompatible; vuelve a entrenar o migra "
            "explícitamente en vez de cargar pesos en un modelo distinto."
        )
    if expected_class and ck.get("model_class") != expected_class:
        raise ValueError(
            f"{path}: el checkpoint es de {ck.get('model_class')!r} y se está "
            f"cargando en {expected_class!r}."
        )
    return ck


def index_hash(items: list[tuple[int, int]]) -> str:
    """Hash estable de la lista de transiciones usadas."""
    import hashlib

    h = hashlib.sha256()
    for t, k in items:
        h.update(f"{t}:{k};".encode())
    return h.hexdigest()[:16]


def make_run_id(model: str, profile: str, experiment: str, seed: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    tag = profile.replace("v3-", "").lower() if profile else "na"
    return f"{experiment}__{model}-{tag}__seed{seed}__{stamp}"
