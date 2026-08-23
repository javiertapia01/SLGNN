import numpy as np
import torch

from slgnn.config import SLGNNConfig
from slgnn.diagnostics import (REGIME_TO_CODE, contact_age_bucket,
                               integrator_position_residuals, ood_coverage,
                               quantile_bin, robust_summary,
                               select_stratified_starts, snapshot_geometry,
                               update_contact_ages, wasserstein_1d)
from slgnn.sdf import BoxSDF
from slgnn.state import Particles


def test_integrator_residuals_identify_semi_implicit_data():
    q0 = torch.tensor([[0.0, 0.0, 0.0]])
    v0 = torch.tensor([[1.0, 0.0, 0.0]])
    v1 = torch.tensor([[1.5, 0.0, 0.0]])
    q1 = q0 + 0.2 * v1
    r = integrator_position_residuals(q0, q1, v0, v1, 0.2)
    assert torch.allclose(r["semi_implicit"], torch.zeros_like(q0))
    assert r["explicit"].norm() > 0
    assert r["midpoint"].norm() > 0


def test_contact_age_lifecycle_and_buckets():
    ages = update_contact_ages({}, [(0, 1), (2, 3)])
    ages = update_contact_ages(ages, [(0, 1)])
    assert ages == {(0, 1): 2}
    assert [contact_age_bucket(x) for x in (1, 2, 6, 21)] == [
        "start", "short", "sustained", "long"
    ]


def test_statistics_and_coverage_have_counts():
    s = robust_summary([1.0, 2.0, 3.0])
    assert s["count"] == 3
    assert np.isclose(s["rmse"], np.sqrt(14 / 3))
    assert np.array_equal(quantile_bin([0, 1, 2, 3], 0.5, 1.5, 2.5), [0, 1, 2, 3])
    assert np.isclose(wasserstein_1d([0, 1], [1, 2]), 1.0)
    cov = ood_coverage(np.arange(100.0), np.array([-10.0, 50.0, 110.0]))
    assert cov["train_count"] == 100 and cov["other_count"] == 3
    assert np.isclose(cov["ood_fraction"], 2 / 3)


def test_snapshot_regimes_cover_free_near_pair_wall_and_mixed():
    cfg = SLGNNConfig()
    particles = Particles.uniform(5, m=1.0, radius=0.5)
    wall = BoxSDF([-2, -2, -2], [2, 2, 2])
    # 0-1 overlap (pp); 2 overlaps wall; 3 overlaps both particle 4 and wall;
    # no isolated particle is close to a wall/pair in this compact fixture.
    q = torch.tensor([
        [-0.40, 0.0, 0.0], [0.40, 0.0, 0.0],
        [1.60, -1.60, 0.0], [1.60, 1.60, 0.0], [0.80, 1.60, 0.0],
    ])
    z = torch.zeros_like(q)
    geo = snapshot_geometry(q, z, z, particles, wall, cfg)
    assert geo.regime[0].item() == REGIME_TO_CODE["pp_only"]
    assert geo.regime[2].item() == REGIME_TO_CODE["pw_only"]
    assert geo.regime[3].item() == REGIME_TO_CODE["mixed"]


def test_stratified_starts_are_unique_and_bounded():
    labels = np.tile(np.arange(5), (10, 1))
    starts = select_stratified_starts(labels, max_start=8, per_stratum=2)
    ids = [x[0] for x in starts]
    assert len(ids) == len(set(ids))
    assert all(x <= 8 for x in ids)
