"""Infraestructura experimental neutral compartida por SLGNN-v3 y los baselines GNS.

Este paquete no conoce ninguna arquitectura. Su única responsabilidad es que
los modelos que se comparan reciban **exactamente** los mismos datos, targets,
splits, muestreo y métricas. Nada aquí importa `slgnn`, `slgnn_v3` ni
`gns_baseline`; la dependencia va siempre en el otro sentido.

Módulos:

- `data`                 lectura del dataset Dynami-CAL por nombre de cabecera
- `nondimensionalization` escalas L0/M0/T0 y conversión coherente
- `targets`              incrementos de momento Delta p / Delta L
- `contact_labels`       clasificación geométrica de régimen por transición
- `splits`               train/val/test declarativos, con CASE07 protegido
- `sampling`             sampler estratificado por régimen de contacto
- `metrics`              métricas por régimen y agregación por semilla
- `runner`               bucle de entrenamiento/evaluación común
- `checkpointing`        manifiestos y checkpoints versionados
"""

from .data import (
    DATASETS,
    Trajectory,
    dataset_root,
    list_cases,
    load_case,
    resolve_case_dir,
)
from .nondimensionalization import Scales, default_scales
from .targets import TransitionTargets, build_targets

__all__ = [
    "DATASETS",
    "Scales",
    "Trajectory",
    "TransitionTargets",
    "build_targets",
    "dataset_root",
    "default_scales",
    "list_cases",
    "load_case",
    "resolve_case_dir",
]
