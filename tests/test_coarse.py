"""T1 — contratos de `C_phi` (§7.2).

Los cuatro contratos que la especificación exige, en el estilo de los tests de
garantías por construcción que ya usa `tests/test_model.py`: invariancia a
permutación, invariancia del espectro bajo rotación rígida del sistema completo,
escalamiento correcto con el ancho de ventana, y cierre energético.
"""

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from slgnn.data import default_scales, load_case
from slgnn.sdf import BoxSDF
from twin.coarse import (CoarseConfig, calibrate_dissipation, coarse_grain,
                         coarse_grain_windows)
from twin.units import Scaling

_CASE = (Path(__file__).resolve().parent.parent / "data" / "extracted"
         / "60Spheres_Gravity_Inside_Cuboidal_Enclosure" / "CASE01")

needs_data = pytest.mark.skipif(not _CASE.exists(), reason="dataset no descargado")

_BOX_M = 0.03
_N_FRAMES = 600


def _setup(n_frames=_N_FRAMES):
    scales = default_scales()
    scaling = Scaling.from_slgnn(scales)
    tr = scales.nondim(load_case(_CASE, dt=1e-4, dtype=torch.float64))
    tr = replace(tr, q=tr.q[:n_frames], v=tr.v[:n_frames], omega=tr.omega[:n_frames])
    wall = BoxSDF([0.0] * 3, [scales.length(_BOX_M)] * 3)
    g = np.zeros(3)
    g[1] = -scales.gravity(9.81)  # eje de sedimentación verificado: -y
    return tr, wall, scaling, g


def _rotation(seed=0):
    torch.manual_seed(seed)
    a = torch.randn(3, dtype=torch.float64)
    K = torch.tensor([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]],
                     dtype=torch.float64)
    return torch.linalg.matrix_exp(K)


# ---------------------------------------------------------------------------
# Contrato 1 — invariancia a permutación de índices de partícula
# ---------------------------------------------------------------------------

@needs_data
def test_spectrum_invariant_to_particle_permutation():
    tr, wall, scaling, g = _setup(300)
    cfg = CoarseConfig()
    base = coarse_grain(tr, wall, scaling, cfg, g_vec=g)

    perm = torch.randperm(tr.q.shape[1], generator=torch.Generator().manual_seed(3))
    shuffled = replace(tr, q=tr.q[:, perm], v=tr.v[:, perm], omega=tr.omega[:, perm],
                       m=tr.m[perm], radii=tr.radii[perm])
    other = coarse_grain(shuffled, wall, scaling, cfg, g_vec=g)

    assert other.n_events == base.n_events
    assert np.allclose(other.spectrum.rates, base.spectrum.rates)
    assert other.ke == pytest.approx(base.ke, rel=1e-12)
    assert other.p_diss == pytest.approx(base.p_diss, rel=1e-12)


# ---------------------------------------------------------------------------
# Contrato 2 — invariancia bajo rotación rígida (partículas + gravedad + pared)
# ---------------------------------------------------------------------------

@needs_data
def test_spectrum_invariant_under_rigid_so3_rotation():
    """Reutiliza la pose de `BoxSDF` (§4.2), la misma maquinaria que valida la
    equivariancia SO(3) del modelo."""
    tr, wall, scaling, g = _setup(300)
    cfg = CoarseConfig()
    base = coarse_grain(tr, wall, scaling, cfg, g_vec=g)

    Q = _rotation()
    rot = replace(tr, q=tr.q @ Q.T, v=tr.v @ Q.T, omega=tr.omega @ Q.T)
    wall_rot = BoxSDF([0.0] * 3, [default_scales().length(_BOX_M)] * 3,
                      pose=(Q, torch.zeros(3, dtype=torch.float64)))
    g_rot = (torch.as_tensor(g) @ Q.T).numpy()
    other = coarse_grain(rot, wall_rot, scaling, cfg, g_vec=g_rot)

    assert other.n_events == base.n_events
    assert np.allclose(other.spectrum.rates, base.spectrum.rates)
    assert other.ke == pytest.approx(base.ke, rel=1e-9)
    assert other.pe == pytest.approx(base.pe, rel=1e-9)
    assert other.p_diss == pytest.approx(base.p_diss, rel=1e-9)


# ---------------------------------------------------------------------------
# Contrato 3 — escalamiento con el ancho de ventana
# ---------------------------------------------------------------------------

@needs_data
def test_rates_are_conserved_and_counts_are_not_when_window_doubles():
    tr, wall, scaling, g = _setup(600)
    cfg = CoarseConfig()
    full, first, second = coarse_grain_windows(
        tr, wall, scaling, cfg, [(0, 600), (0, 300), (300, 600)], g_vec=g
    )

    # los conteos son extensivos y se suman exactamente
    assert np.allclose(full.spectrum.counts,
                       first.spectrum.counts + second.spectrum.counts)
    assert full.n_events == first.n_events + second.n_events
    # las tasas son intensivas: promedio pesado por duración
    w = np.array([first.duration, second.duration])
    expected = (first.spectrum.rates * w[0] + second.spectrum.rates * w[1]) / w.sum()
    assert np.allclose(full.spectrum.rates, expected)
    assert full.duration == pytest.approx(first.duration + second.duration)


# ---------------------------------------------------------------------------
# Contrato 4 — cierre energético
# ---------------------------------------------------------------------------

@needs_data
def test_energy_closure_after_dissipation_calibration():
    """En una caja estática no entra energía: `p_in ≡ 0` por construcción.

    El estimador crudo `(1-e²)·E_impacto` **no cierra** el balance — recupera
    del orden de un tercio de la pérdida real. Este test fija dos cosas: que la
    calibración de §`calibrate_dissipation` deja el residual bajo el 10 % que
    exige la especificación, y que el factor faltante es sistemático (todas las
    ventanas piden un kappa parecido), no ruido de muestreo.
    """
    tr, wall, scaling, g = _setup(1200)
    cfg = CoarseConfig()
    windows = [(a, a + 300) for a in range(0, 1200, 300)]

    cal = calibrate_dissipation(tr, wall, scaling, cfg, windows, g_vec=g)
    assert cal["kappa"] > 1.5, "el estimador crudo no debería cerrar el balance"
    assert cal["residual"] < 0.10, f"cierre energético al {cal['residual']:.1%}"
    per = np.array(cal["per_window"])
    assert per.std() / per.mean() < 0.15, "el factor faltante debe ser sistemático"

    calibrated = replace(cfg, kappa_diss=cal["kappa"])
    feats = coarse_grain_windows(tr, wall, scaling, calibrated, windows, g_vec=g)
    for f in feats:
        assert f.p_in_wall == pytest.approx(0.0, abs=1e-15), "pared fija: v_W = 0"
        assert f.closure_gap < 0.35, "residual del balance por ventana"
    total_in = sum(f.p_in_balance * f.duration for f in feats)
    total_scale = sum(abs(f.p_mech) * f.duration for f in feats)
    assert abs(total_in) / total_scale < 0.10


# ---------------------------------------------------------------------------
# Contratos que no necesitan el dataset
# ---------------------------------------------------------------------------

class _Traj:
    def __init__(self, q, v, omega, m, radii, dt):
        self.q, self.v, self.omega = q, v, omega
        self.m, self.radii, self.dt = m, radii, dt


def _bouncing_pair(n_bounces=4, period=20, v=1.0, dt=1e-3, radius=0.5):
    """Par que rebota periódicamente: proceso de eventos de tasa conocida."""
    t = np.arange(n_bounces * period)
    saw = np.abs((t % period) - period / 2)
    gap = v * dt * saw - v * dt * 1.5
    pos = np.zeros((t.size, 2, 3))
    pos[:, 1, 0] = 2 * radius + gap
    vel = np.zeros((t.size, 2, 3))
    vel[:, 1, 0] = np.where((t % period) < period / 2, -v, v)
    return _Traj(pos, vel, np.zeros_like(vel), np.full(2, 2.0), np.full(2, radius), dt)


def test_synthetic_event_rate_matches_closed_form():
    dt, period, n = 1e-3, 20, 4
    tr = _bouncing_pair(n_bounces=n, period=period, dt=dt)
    scaling = Scaling(L=1.0, T=1.0, M=1.0)
    cfg = CoarseConfig(e_min=1e-6, e_max=1e2, n_bins=16)
    f = coarse_grain(tr, BoxSDF([-1e3] * 3, [1e3] * 3), scaling, cfg)

    assert f.n_events == n
    duration = n * period * dt
    expected_rate = n / (duration * float(tr.m.sum()))
    assert f.spectrum.total_rate() == pytest.approx(expected_rate, rel=1e-9)
    m_eff = 1.0
    assert f.spectrum.mean_energy() == pytest.approx(0.5 * m_eff * 1.0**2, rel=0.15)


def test_spectrum_reports_overflow_instead_of_dropping_events():
    tr = _bouncing_pair()
    scaling = Scaling(L=1.0, T=1.0, M=1.0)
    narrow = CoarseConfig(e_min=1e-9, e_max=1e-6, n_bins=8)
    f = coarse_grain(tr, BoxSDF([-1e3] * 3, [1e3] * 3), scaling, narrow)
    assert f.spectrum.rates.sum() == 0.0
    assert f.spectrum.overflow.sum() > 0.0, "los eventos fuera de rango deben reportarse"
