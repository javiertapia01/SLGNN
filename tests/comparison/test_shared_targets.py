"""Los targets se construyen una sola vez, por la infraestructura común."""

from pathlib import Path

import pytest
import torch

from slgnn_experiments.data import load_case
from slgnn_experiments.nondimensionalization import default_scales
from slgnn_experiments.targets import accelerations, build_targets, momentum_totals

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def traj():
    return default_scales().nondim(
        load_case("two_spheres", "1x", REPO_ROOT, max_steps=40)
    )


def test_targets_are_momentum_increments(traj):
    t = build_targets(traj)
    expected = traj.mass.unsqueeze(0).unsqueeze(-1) * (traj.v[1:] - traj.v[:-1])
    assert torch.equal(t.delta_p, expected)
    expected_L = traj.inertia.unsqueeze(0).unsqueeze(-1) * (
        traj.omega[1:] - traj.omega[:-1]
    )
    assert torch.equal(t.delta_L, expected_L)


def test_targets_are_deterministic(traj):
    a, b = build_targets(traj), build_targets(traj)
    assert torch.equal(a.delta_p, b.delta_p) and torch.equal(a.delta_L, b.delta_L)


def test_acceleration_is_secondary_not_primary(traj):
    """La aceleración se conserva como métrica, y es `Delta p / (m dt)`."""
    t = build_targets(traj)
    a, _ = accelerations(traj)
    assert torch.allclose(
        a, t.delta_p / (traj.mass.unsqueeze(0).unsqueeze(-1) * traj.dt), atol=1e-12
    )


def test_two_spheres_conserve_momentum_in_the_data(traj):
    """Sistema aislado: la propia referencia DEM conserva momento.

    Si esto fallara, ninguna métrica de conservación del modelo tendría
    sentido, porque el target ya la violaría.
    """
    p, L = momentum_totals(traj)
    # La tolerancia la fija la PRECISIÓN DEL CSV, no la del solver: los
    # archivos guardan ~6 cifras significativas, así que el momento total del
    # propio DEM se conserva hasta ~1e-5 en unidades adimensionales. Apretar
    # más este test mediría el formato de texto, no la física.
    rel_p = float((p - p[0]).abs().max()) / max(float(p.abs().max()), 1e-30)
    rel_L = float((L - L[0]).abs().max()) / max(float(L.abs().max()), 1e-30)
    assert rel_p <= 1e-4, f"momento lineal del dataset deriva {rel_p:.2e}"
    assert rel_L <= 1e-3, f"momento angular del dataset deriva {rel_L:.2e}"


def test_no_spin_target_is_invented(traj):
    """`Delta L` viene de `omega` del DEM, que existe en las tres variantes."""
    assert traj.omega.shape == traj.v.shape
    assert torch.isfinite(traj.omega).all()
