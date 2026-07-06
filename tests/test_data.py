"""Tests del loader contra las 3 variantes de esquema (ver DATA_NOTES.md)."""

from pathlib import Path

import pytest
import torch

from slgnn.data import (default_scales, finite_difference_accelerations,
                        load_case)

_HEADER_A = (
    '"Diameter","Velocity:0","Velocity:1","Velocity:2",'
    '"Angular_velocity:0","Angular_velocity:1","Angular_velocity:2",'
    '"Orientation:0","Orientation:1","Orientation:2",'
    '"Density","Particle_ID","coordinates:0","coordinates:1","coordinates:2"'
)
_HEADER_B = (
    '"Velocity:0","Velocity:1","Velocity:2",'
    '"Angular_velocity:0","Angular_velocity:1","Angular_velocity:2",'
    '"Orientation:0","Orientation:1","Orientation:2",'
    '"Density","Particle_ID","coordinates:0","coordinates:1","coordinates:2"'
)
_HEADER_C = (
    '"Diameter","Velocity:0","Velocity:1","Velocity:2",'
    '"Angular_velocity:0","Angular_velocity:1","Angular_velocity:2",'
    '"Orientation:0","Orientation:1","Orientation:2",'
    '"Density","coordinates:0","coordinates:1","coordinates:2"'
)


def _write_case(tmp_path, header, rows_per_step):
    d = tmp_path / "CASEX"
    d.mkdir()
    for k, rows in enumerate(rows_per_step):
        (d / f"data_at_timestep_{k:03d}.csv").write_text(
            header + "\n" + "\n".join(rows) + "\n"
        )
    return d


def test_variant_a_full_schema(tmp_path):
    rows = [
        ["0.005,1,0,0,0,0,0,0,0,1,4000,1,0.01,0.02,0.03",
         "0.005,-1,0,0,0,0,0,0,0,1,4000,2,0.02,0.02,0.03"],
        ["0.005,1,0,0,0,0,0,0,0,1,4000,1,0.011,0.02,0.03",
         "0.005,-1,0,0,0,0,0,0,0,1,4000,2,0.019,0.02,0.03"],
    ]
    case = _write_case(tmp_path, _HEADER_A, rows)
    tr = load_case(case, dt=1e-4, cache=False)
    assert tr.q.shape == (2, 2, 3)
    assert torch.allclose(tr.radii, torch.full((2,), 0.0025))
    assert tr.q[0, 0, 0].item() == pytest.approx(0.01)


def test_variant_a_sorts_by_particle_id(tmp_path):
    # filas desordenadas: ID 2 primero
    rows = [
        ["0.005,-1,0,0,0,0,0,0,0,1,4000,2,0.02,0.0,0.0",
         "0.005,1,0,0,0,0,0,0,0,1,4000,1,0.01,0.0,0.0"],
    ]
    case = _write_case(tmp_path, _HEADER_A, rows)
    tr = load_case(case, dt=1e-4, cache=False)
    assert tr.q[0, 0, 0].item() == pytest.approx(0.01)  # ID 1 quedó primero
    assert tr.v[0, 0, 0].item() == pytest.approx(1.0)


def test_variant_b_missing_diameter_uses_default(tmp_path):
    rows = [["1,0,0,0,0,0,0,0,1,4000,1,0.01,0.0,0.0"]]
    case = _write_case(tmp_path, _HEADER_B, rows)
    tr = load_case(case, dt=1e-4, cache=False)
    assert torch.allclose(tr.radii, torch.tensor([0.0025]))
    # masa = rho * pi/6 * d^3
    assert tr.m[0].item() == pytest.approx(4000 * 3.14159265 / 6 * 0.005**3, rel=1e-5)


def test_variant_c_missing_id_preserves_row_order(tmp_path):
    rows = [
        ["0.005,1,0,0,0,0,0,0,0,1,4000,0.05,0.0,0.0",
         "0.005,2,0,0,0,0,0,0,0,1,4000,0.10,0.0,0.0"],
    ]
    case = _write_case(tmp_path, _HEADER_C, rows)
    tr = load_case(case, dt=1e-3, cache=False)
    assert tr.q[0, 0, 0].item() == pytest.approx(0.05)
    assert tr.q[0, 1, 0].item() == pytest.approx(0.10)


def test_unknown_header_raises(tmp_path):
    header = _HEADER_A.replace('"Diameter"', '"Diametro_sorpresa"')
    rows = [["0.005,1,0,0,0,0,0,0,0,1,4000,1,0.01,0.0,0.0"]]
    case = _write_case(tmp_path, header, rows)
    with pytest.raises(ValueError, match="Cabecera no reconocida"):
        load_case(case, dt=1e-4, cache=False)


def test_finite_difference_accelerations():
    v = torch.tensor([[[0.0, 0, 0]], [[1.0, 0, 0]]])
    a = finite_difference_accelerations(v, dt=0.5)
    assert torch.allclose(a, torch.tensor([[[2.0, 0, 0]]]))


def test_scales_roundtrip_gravity():
    s = default_scales()
    assert s.gravity() == pytest.approx(9.81 * 0.01**2 / 0.005)


_REAL_CASE = (
    Path(__file__).resolve().parent.parent
    / "data" / "extracted" / "Benchmark_2Spheres_Oblique_Collision" / "1x"
)


@pytest.mark.skipif(not _REAL_CASE.exists(), reason="dataset no descargado")
def test_load_real_two_sphere_benchmark():
    tr = load_case(_REAL_CASE, dt=1e-4, cache=False)
    t, n, _ = tr.q.shape
    assert n == 2
    assert t >= 100
    # diámetro y densidad del PDF del dataset
    assert torch.allclose(tr.radii, torch.full((2,), 0.0025))
    # sanidad física: velocidades del orden de m/s, posiciones en la caja de cm
    assert tr.v.abs().max() < 50.0
    assert tr.q.abs().max() < 1.0
