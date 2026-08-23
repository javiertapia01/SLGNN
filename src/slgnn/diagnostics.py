"""Pure, read-only helpers for the Phase A checkpoint audit.

The functions in this module do not alter the model, integrator, checkpoint, or
dataset.  They centralize definitions from the experimental protocol so the
analysis script and its tests use the same geometry and aggregation rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from .graph import (contact_velocities, neighbor_pairs, pair_geometry,
                    wall_geometry)


REGIME_NAMES = ("free", "near", "pp_only", "pw_only", "mixed")
REGIME_TO_CODE = {name: i for i, name in enumerate(REGIME_NAMES)}


@dataclass
class SnapshotGeometry:
    """Per-particle physical labels and contact-intensity features."""

    regime: torch.Tensor
    pp_delta_max: torch.Tensor
    pp_delta_mean: torch.Tensor
    pp_approach: torch.Tensor
    pp_tangential: torch.Tensor
    pw_delta: torch.Tensor
    pw_approach: torch.Tensor
    pw_tangential: torch.Tensor
    contact_degree: torch.Tensor
    pp_contact_pairs: torch.Tensor
    pw_contact_ids: torch.Tensor
    pp_active_pairs: torch.Tensor
    pp_gap: torch.Tensor
    pp_u_n: torch.Tensor
    pp_u_tau: torch.Tensor
    pw_gap: torch.Tensor
    wall_phi: torch.Tensor


def _incident_sum(values, i, j, n):
    out = torch.zeros(n, dtype=values.dtype, device=values.device)
    out.index_add_(0, i, values)
    out.index_add_(0, j, values)
    return out


def _incident_max(values, i, j, n):
    out = torch.zeros(n, dtype=values.dtype, device=values.device)
    if values.numel():
        out.scatter_reduce_(0, i, values, reduce="amax", include_self=True)
        out.scatter_reduce_(0, j, values, reduce="amax", include_self=True)
    return out


@torch.no_grad()
def snapshot_geometry(q, v, omega, particles, wall, cfg, t: float = 0.0):
    """Classify one state using SLGNN's own pair and wall geometry.

    Compression is the geometric ``(-gap)+`` requested by the protocol, not
    the model's softplus feature.  The active band is ``gap <= g_on`` for both
    pair and wall contacts (equivalent to ``d <= r_on`` for unit-diameter
    equal spheres with the checkpoint configuration).
    """

    n = q.shape[0]
    radii = particles.radii.to(q)
    idx = neighbor_pairs(q, cfg.r_list)
    ps = pair_geometry(q, radii, idx, cfg)
    _, u_n, u_tau = contact_velocities(v, omega, ps)
    ws = wall_geometry(q, v, omega, radii, wall, t, cfg)

    pp_active = ps.g <= cfg.g_on
    pp_contact = ps.g <= 0
    pw_active = ws.g <= cfg.g_on
    pw_contact = ws.g <= 0

    ai, aj = ps.idx_i[pp_active], ps.idx_j[pp_active]
    ci, cj = ps.idx_i[pp_contact], ps.idx_j[pp_contact]
    ones_active = torch.ones(ai.shape[0], dtype=q.dtype, device=q.device)
    active_degree = _incident_sum(ones_active, ai, aj, n)
    contact_degree = _incident_sum(
        torch.ones(ci.shape[0], dtype=q.dtype, device=q.device), ci, cj, n
    ) + pw_contact.to(q.dtype)

    has_pp_contact = torch.zeros(n, dtype=torch.bool, device=q.device)
    if ci.numel():
        has_pp_contact[ci] = True
        has_pp_contact[cj] = True
    has_pp_active = active_degree > 0

    regime = torch.full(
        (n,), REGIME_TO_CODE["near"], dtype=torch.long, device=q.device
    )
    any_active = has_pp_active | pw_active
    regime[~any_active] = REGIME_TO_CODE["free"]
    regime[has_pp_contact & ~pw_contact] = REGIME_TO_CODE["pp_only"]
    regime[pw_contact & ~has_pp_contact] = REGIME_TO_CODE["pw_only"]
    regime[pw_contact & has_pp_contact] = REGIME_TO_CODE["mixed"]

    gap_a = ps.g[pp_active]
    delta_a = (-gap_a).clamp_min(0)
    approach_a = (-u_n[pp_active]).clamp_min(0)
    tangential_a = u_tau[pp_active].norm(dim=-1)
    pp_delta_max = _incident_max(delta_a, ai, aj, n)
    pp_delta_sum = _incident_sum(delta_a, ai, aj, n)
    pp_delta_mean = pp_delta_sum / active_degree.clamp_min(1)
    pp_approach = _incident_max(approach_a, ai, aj, n)
    pp_tangential = _incident_max(tangential_a, ai, aj, n)

    return SnapshotGeometry(
        regime=regime,
        pp_delta_max=pp_delta_max,
        pp_delta_mean=pp_delta_mean,
        pp_approach=pp_approach,
        pp_tangential=pp_tangential,
        pw_delta=(-ws.g).clamp_min(0),
        pw_approach=(-ws.w_n).clamp_min(0),
        pw_tangential=ws.w_tau.norm(dim=-1),
        contact_degree=contact_degree,
        pp_contact_pairs=torch.stack([ci, cj], dim=-1),
        pw_contact_ids=torch.nonzero(pw_contact, as_tuple=False).flatten(),
        pp_active_pairs=torch.stack([ai, aj], dim=-1),
        pp_gap=ps.g,
        pp_u_n=u_n,
        pp_u_tau=u_tau.norm(dim=-1),
        pw_gap=ws.g,
        wall_phi=ws.phi,
    )


def integrator_position_residuals(q0, q1, v0, v1, dt: float):
    """Oracle position residuals for semi-implicit, explicit, and midpoint."""

    return {
        "semi_implicit": q1 - (q0 + dt * v1),
        "explicit": q1 - (q0 + dt * v0),
        "midpoint": q1 - (q0 + 0.5 * dt * (v0 + v1)),
    }


def update_contact_ages(previous: dict, active: Iterable):
    """Return ages for the active identifiers and discard ended contacts."""

    return {key: int(previous.get(key, 0)) + 1 for key in active}


def contact_age_bucket(age: int) -> str:
    if age <= 1:
        return "start"
    if age <= 5:
        return "short"
    if age <= 20:
        return "sustained"
    return "long"


def robust_summary(values) -> dict:
    """Protocol summary for a one-dimensional non-negative metric."""

    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "count": 0, "mean": np.nan, "median": np.nan, "rmse": np.nan,
            "p90": np.nan, "p95": np.nan, "p99": np.nan,
            "max_robust": np.nan, "max": np.nan,
        }
    return {
        "count": int(x.size),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "rmse": float(np.sqrt(np.mean(x * x))),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "max_robust": float(np.quantile(x, 0.999)),
        "max": float(x.max()),
    }


def quantile_bin(values, q50: float, q90: float, q99: float):
    """Return protocol intensity bins 0-Q50, Q50-Q90, Q90-Q99, >Q99."""

    x = np.asarray(values)
    out = np.zeros(x.shape, dtype=np.int8)
    out[x > q50] = 1
    out[x > q90] = 2
    out[x > q99] = 3
    return out


def wasserstein_1d(a, b) -> float:
    """Exact empirical one-dimensional Wasserstein-1 distance."""

    a = np.sort(np.asarray(a, dtype=np.float64).ravel())
    b = np.sort(np.asarray(b, dtype=np.float64).ravel())
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if not a.size or not b.size:
        return float("nan")
    all_values = np.concatenate([a, b])
    all_values.sort()
    deltas = np.diff(all_values)
    if not deltas.size:
        return 0.0
    points = all_values[:-1]
    cdf_a = np.searchsorted(a, points, side="right") / a.size
    cdf_b = np.searchsorted(b, points, side="right") / b.size
    return float(np.sum(np.abs(cdf_a - cdf_b) * deltas))


def ood_coverage(train, other, eps: float = 1e-12) -> dict:
    """Q0.5-Q99.5 coverage and IQR-normalized Wasserstein-1 distance."""

    train = np.asarray(train, dtype=np.float64)
    other = np.asarray(other, dtype=np.float64)
    train = train[np.isfinite(train)]
    other = other[np.isfinite(other)]
    lo, hi = np.quantile(train, [0.005, 0.995])
    iqr = np.quantile(train, 0.75) - np.quantile(train, 0.25)
    return {
        "train_q005": float(lo),
        "train_q995": float(hi),
        "ood_fraction": float(np.mean((other < lo) | (other > hi))),
        "wasserstein_1": wasserstein_1d(train, other),
        "wasserstein_1_normalized": wasserstein_1d(train, other) / (iqr + eps),
        "train_count": int(train.size),
        "other_count": int(other.size),
    }


def select_stratified_starts(frame_regimes, max_start: int,
                             per_stratum: int = 4):
    """Select deterministic, spread-out starts containing each regime.

    ``frame_regimes`` is a [T,N] integer array.  A frame may qualify for more
    than one stratum, but a selected timestep is used only once.
    """

    labels = np.asarray(frame_regimes)
    selected = []
    used = set()
    for code, name in enumerate(REGIME_NAMES):
        candidates = np.flatnonzero((labels[: max_start + 1] == code).any(axis=1))
        candidates = np.array([x for x in candidates if int(x) not in used])
        if not candidates.size:
            continue
        indices = np.linspace(0, candidates.size - 1,
                              min(per_stratum, candidates.size)).round().astype(int)
        for start in candidates[indices]:
            start = int(start)
            if start not in used:
                selected.append((start, name))
                used.add(start)
    return sorted(selected)


def max_geometric_penetration(geometry: SnapshotGeometry) -> float:
    pair = float((-geometry.pp_gap).clamp_min(0).max()) if geometry.pp_gap.numel() else 0.0
    wall = float((-geometry.pw_gap).clamp_min(0).max()) if geometry.pw_gap.numel() else 0.0
    return max(pair, wall)
