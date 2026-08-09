"""Splits declarativos, con `CASE07` protegido de toda selección.

`CASE07` del dataset con gravedad es el caso de extrapolación (energía
cinética inicial ~3x el más energético de entrenamiento). Usarlo para elegir
hiperparámetros o para seleccionar el mejor checkpoint invalida la
comparación: `assert_case07_protected` lo impide con un error, no con un
comentario (§16.1, §22.14).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .data import DATASETS, Trajectory, list_cases, load_case, slice_frames
from .nondimensionalization import Scales

PROTECTED = {("sixty_gravity", "CASE07")}


@dataclass
class Split:
    dataset: str
    train_cases: tuple[str, ...]
    val_case: str | None
    test_cases: tuple[str, ...] = ()

    def all_cases(self) -> tuple[str, ...]:
        out = list(self.train_cases)
        if self.val_case:
            out.append(self.val_case)
        return tuple(out) + tuple(self.test_cases)


@dataclass
class LoadedSplit:
    split: Split
    train: list[Trajectory]
    val: Trajectory | None
    test: list[Trajectory] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "dataset": self.split.dataset,
            "train_cases": list(self.split.train_cases),
            "val_case": self.split.val_case,
            "test_cases": list(self.split.test_cases),
            "n_train_trajectories": len(self.train),
            "n_train_transitions": sum(t.n_steps - 1 for t in self.train),
        }


def assert_case07_protected(split: Split) -> None:
    for case in list(split.train_cases) + ([split.val_case] if split.val_case else []):
        if (split.dataset, case) in PROTECTED:
            raise ValueError(
                f"{split.dataset}/{case} es el caso de extrapolación y no puede "
                "aparecer en entrenamiento ni en validación. Solo puede evaluarse "
                "una vez, al final (§16.1 y §22.14 de las instrucciones)."
            )


def load_split(
    split: Split, repo_root: Path | str, scales: Scales,
    dtype=None, max_steps: int | None = None,
    frame_start: int = 0, frame_stop: int | None = None,
) -> LoadedSplit:
    """Carga y adimensionaliza. La transformación se aplica una sola vez."""
    assert_case07_protected(split)
    import torch

    dtype = dtype or torch.float64

    def one(case: str) -> Trajectory:
        tr = load_case(split.dataset, case, repo_root, dtype=dtype, max_steps=max_steps)
        if frame_start or frame_stop is not None:
            tr = slice_frames(tr, frame_start, frame_stop)
        return scales.nondim(tr)

    return LoadedSplit(
        split=split,
        train=[one(c) for c in split.train_cases],
        val=one(split.val_case) if split.val_case else None,
        test=[one(c) for c in split.test_cases],
    )


def split_from_config(cfg: dict) -> Split:
    """Construye un `Split` desde el bloque `data` de un YAML."""
    dataset = cfg["dataset"]
    if dataset not in DATASETS:
        raise KeyError(f"Dataset desconocido {dataset!r}: {sorted(DATASETS)}")
    return Split(
        dataset=dataset,
        train_cases=tuple(cfg.get("train_cases", ())),
        val_case=cfg.get("val_case"),
        test_cases=tuple(cfg.get("test_cases", ())),
    )
