"""Ejecuta la batería completa de benchmarks pequeños del MVP (fases 7 y 8).

    python scripts/slgnn_v3/run_all_benchmarks.py

Corre, para cada experimento, la selección de `lr` con presupuesto idéntico
por familia y luego tres semillas de `v3-C`, `v3-I` y `GNSControlled`.
`CASE07` solo se evalúa en el experimento de 60 esferas y solo al final.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable

# `--final-test` en los tres: los casos de test (`4x`, los ángulos oblicuos y
# `CASE07`) son extrapolación genuina y nunca intervienen en la selección de
# `lr` ni del mejor checkpoint. Se evalúan una sola vez, al cerrar la fase.
JOBS = [
    ("two_spheres", ["--tune-lr", "--final-test"]),
    ("one_sphere_wall", ["--tune-lr", "--final-test"]),
    ("gravity60_small", ["--tune-lr", "--final-test"]),
]


def main() -> int:
    for name, extra in JOBS:
        cfg = REPO_ROOT / "configs/experiments" / f"{name}.yaml"
        cmd = [PY, str(REPO_ROOT / "scripts/slgnn_v3/compare.py"),
               "--experiment", str(cfg), "--seeds", "0", "1", "2", *extra]
        print(f"\n{'='*70}\n[bench] {name}\n{'='*70}", flush=True)
        r = subprocess.run(cmd, cwd=REPO_ROOT)
        if r.returncode != 0:
            print(f"[bench] {name} FALLÓ con código {r.returncode}", flush=True)
            return r.returncode
    print("\n[bench] batería completa terminada")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
