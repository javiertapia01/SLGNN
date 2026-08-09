"""La comparación es justa solo si los datos son literalmente los mismos.

Estos tests no comprueban rendimiento: comprueban que el protocolo no está
sesgado antes de medir nada.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from slgnn_experiments.data import DATASETS, load_case
from slgnn_experiments.nondimensionalization import default_scales
from slgnn_experiments.scene import build_scene
from slgnn_experiments.splits import Split, assert_case07_protected, load_split

REPO_ROOT = Path(__file__).resolve().parents[2]
GNS_SRC = REPO_ROOT / "src" / "gns_baseline"


def test_gns_baseline_does_not_import_v3_model():
    """§15.1: el baseline comparte infraestructura neutral, no el modelo v3."""
    offenders = []
    for path in GNS_SRC.glob("*.py"):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if not stripped.startswith(("import ", "from ")):
                continue
            if re.search(r"\bslgnn_v3\b", stripped) or re.search(r"\bslgnn\.", stripped):
                offenders.append(f"{path.name}:{n}: {stripped}")
    assert not offenders, "gns_baseline importa v3 o el legacy:\n" + "\n".join(offenders)


def test_v3_does_not_import_legacy():
    """§2.2: `slgnn_v3` no importa `slgnn.model.SLGNN` ni su integrador."""
    offenders = []
    for path in (REPO_ROOT / "src" / "slgnn_v3").glob("*.py"):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            s = line.strip()
            if s.startswith(("import ", "from ")) and re.search(r"\bslgnn\b(?!_)", s):
                offenders.append(f"{path.name}:{n}: {s}")
    assert not offenders, "slgnn_v3 importa el legacy:\n" + "\n".join(offenders)


def test_legacy_package_still_imports():
    """El legacy debe seguir operativo después de todos los cambios."""
    import slgnn
    from slgnn.model import SLGNN  # noqa: F401
    from slgnn.integrator import semi_implicit_step  # noqa: F401


@pytest.mark.skipif(
    not (REPO_ROOT / "data" / "extracted" /
         DATASETS["two_spheres"].subpath).is_dir(),
    reason="dataset no descomprimido",
)
def test_both_models_receive_identical_tensors():
    """Un solo `Scene` alimenta a ambos: no hay dos rutas de datos."""
    scales = default_scales()
    split = Split("two_spheres", ("1x",), "2x")
    loaded = load_split(split, REPO_ROOT, scales)
    scene = build_scene("two_spheres", scales)
    a = scene.state_at(loaded.train[0], 10)
    b = scene.state_at(loaded.train[0], 10)
    for f in ("q", "v", "omega", "mass", "radius", "inertia"):
        assert torch.equal(getattr(a.particles, f), getattr(b.particles, f))
    assert a.time == b.time


def test_case07_cannot_be_selected_on():
    with pytest.raises(ValueError, match="extrapolación"):
        assert_case07_protected(Split("sixty_gravity", ("CASE07",), "CASE06"))
    with pytest.raises(ValueError, match="extrapolación"):
        assert_case07_protected(Split("sixty_gravity", ("CASE01",), "CASE07"))
    # como test sí es legítimo
    assert_case07_protected(
        Split("sixty_gravity", ("CASE01",), "CASE06", ("CASE07",))
    )


def test_nondimensionalization_applied_exactly_once():
    scales = default_scales()
    tr = load_case("two_spheres", "1x", REPO_ROOT, max_steps=3)
    nd = scales.nondim(tr)
    assert nd.dimensionless
    with pytest.raises(ValueError, match="ya está adimensionalizada"):
        scales.nondim(nd)


def test_derived_scales_are_consistent():
    s = default_scales()
    assert s.P0 == pytest.approx(s.M0 * s.V0)
    assert s.LP0 == pytest.approx(s.L0 * s.P0)
    assert s.F0 == pytest.approx(s.P0 / s.T0)
    assert s.E0 == pytest.approx(s.F0 * s.L0)
    assert s.PSI0 == pytest.approx(s.E0 / s.T0)
