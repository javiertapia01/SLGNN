"""Contratos de la biblioteca, el acoplamiento y el lazo de control.

Incluye el test que ancla el hallazgo sobre `omega(t)` del cilindro: si alguien
"arregla" el perfil volviendo al triangular, este test falla con los datos en la
mano.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from slgnn.data import default_scales, load_case
from twin.coarse import CoarseConfig, MacroFeatures, Spectrum
from twin.confidence import ConfidenceMonitor, Envelope
from twin.control import (MPC, ConstantOmega, MacroPlant, MPCConfig,
                          closed_loop)
from twin.coupling import (AndamioWarning, LibraryPolicy, PeriodicPolicy,
                           TriggeredPolicy, make_policy, psd_feedback_factor)
from twin.cylinder import (make_cylinder_sdf, omega_pdf_literal,
                           omega_triangular, verify_omega_profile)
from twin.harness import wasserstein
from twin.library import DOWN, UP, SpectrumLibrary, hysteresis_verdict
from twin.macro import PBM, MacroState, PBMConfig
from twin.units import Scaling

_CYL = (Path(__file__).resolve().parents[2] / "data" / "extracted"
        / "Extrapolation_2073Spheres_Gravity_Inside_Rotating_Cylinder" / "CASE08")


def _spectrum(peak_bin, amplitude=1.0, n_bins=30):
    cfg = CoarseConfig(n_bins=n_bins)
    rates = np.zeros((n_bins, 2))
    rates[peak_bin, 0] = amplitude
    return Spectrum(edges=cfg.bin_edges(), rates=rates, duration=1.0, mass=1.0,
                    underflow=np.zeros(2), overflow=np.zeros(2))


def _feature(omega, peak_bin, branch, amplitude=1.0):
    return MacroFeatures(
        spectrum=_spectrum(peak_bin, amplitude), omega=omega, ke=1.0, pe=0.0,
        p_diss=1.0, p_mech=0.0, p_in_balance=1.0 + omega, p_in_wall=np.nan,
        closure_gap=np.nan, fill=0.15, com=np.zeros(3), com_r=0.01 * omega,
        com_theta=0.0, theta_toe=np.nan, theta_shoulder=np.nan,
        t_start=0.0, t_end=1.0, n_events=10, n_unresolved=0, branch=branch,
    )


# ---------------------------------------------------------------------------
# Biblioteca
# ---------------------------------------------------------------------------

def test_library_interpolates_between_nodes_and_forbids_extrapolation():
    lib = SpectrumLibrary().fit([_feature(2.0, 10, UP), _feature(8.0, 10, UP)])
    inside = lib.query(5.0)
    assert not inside.ood
    assert inside.spectrum.total_rate() > 0

    outside = lib.query(20.0)
    assert outside.ood, "fuera del rango cubierto debe marcarse OOD"
    # no extrapola: devuelve el extremo, no una prolongación
    assert outside.spectrum.total_rate() == pytest.approx(
        lib.query(8.0).spectrum.total_rate(), rel=1e-9)


def test_library_interpolation_is_geometric_in_rate():
    """La interpolación es lineal en log-tasa, así que preserva positividad."""
    lib = SpectrumLibrary().fit([_feature(0.0, 10, UP, amplitude=1.0),
                                 _feature(10.0, 10, UP, amplitude=100.0)])
    mid = lib.query(5.0).spectrum.total_rate()
    assert mid == pytest.approx(10.0, rel=1e-6)   # media geométrica, no 50.5


def test_branch_discrepancy_becomes_epistemic_uncertainty():
    """La histéresis medida ES la incertidumbre del espectro (§7.7)."""
    agree = SpectrumLibrary().fit([
        _feature(2.0, 10, UP), _feature(8.0, 10, UP),
        _feature(2.0, 10, DOWN), _feature(8.0, 10, DOWN)])
    assert agree.query(5.0).sigma_spec == pytest.approx(0.0, abs=1e-9)

    differ = SpectrumLibrary().fit([
        _feature(2.0, 6, UP), _feature(8.0, 6, UP),
        _feature(2.0, 20, DOWN), _feature(8.0, 20, DOWN)])
    assert differ.query(5.0).sigma_spec > 1.0, "ramas distintas → sigma grande"


def test_hysteresis_verdict_is_resolved_in_omega_not_by_median():
    """Una mediana baja no puede tapar un extremo fuera de umbral."""
    report = {"status": "ok", "probes": [
        {"omega": 1.0, "wasserstein_decades": 2.5, "rate_rel_error": 0.5},
        {"omega": 5.0, "wasserstein_decades": 0.01, "rate_rel_error": 0.01},
        {"omega": 9.0, "wasserstein_decades": 0.01, "rate_rel_error": 0.01},
    ]}
    v = hysteresis_verdict(report, w_threshold=0.15, rate_threshold=0.30)
    assert v["verdict"] == "depende del camino"
    assert v["n_violating"] == 1
    assert v["omega_violating_range"] == (1.0, 1.0)


def test_wasserstein_requires_shared_binning():
    a, b = _spectrum(5), _spectrum(5, n_bins=20)
    with pytest.raises(ValueError, match="binning"):
        wasserstein(a, b)


# ---------------------------------------------------------------------------
# Acoplamiento
# ---------------------------------------------------------------------------

def test_psd_feedback_is_declared_as_scaffold():
    state = MacroState(M=np.array([1.0, 0, 0, 0, 0, 0]))
    with pytest.warns(AndamioWarning, match="andamio"):
        psd_feedback_factor(state, PBMConfig())


def test_refresh_policies_behave_as_documented():
    state = MacroState(M=np.ones(6))
    u = np.array([5.0, 0.01])
    assert make_policy("static").should_refresh(0, state, u)
    assert not make_policy("static").should_refresh(1, state, u)

    periodic = PeriodicPolicy(every=3)
    assert [periodic.should_refresh(k, state, u) for k in range(6)] == \
        [True, False, False, True, False, False]

    trig = TriggeredPolicy(eps=0.1)
    assert trig.should_refresh(0, state, u)
    assert not trig.should_refresh(1, state, u)
    assert trig.should_refresh(2, state, np.array([9.0, 0.01]))

    assert all(LibraryPolicy().should_refresh(k, state, u) for k in range(4))


# ---------------------------------------------------------------------------
# Confianza y control
# ---------------------------------------------------------------------------

def _plant(mode="library"):
    lib = SpectrumLibrary().fit([_feature(w, 18, UP, amplitude=10.0 * w + 1)
                                 for w in (1.0, 4.0, 7.0, 10.0)])
    cfg = MPCConfig(horizon=4, dt=0.5, omega_min=1.0, omega_max=10.0,
                    iters=12, spectrum_mode=mode, omega_nominal=4.0)
    return MacroPlant(PBM(PBMConfig()), lib, cfg), lib, cfg


def test_ood_triggers_restriction_of_omega_bounds():
    _, lib, _ = _plant()
    monitor = ConfidenceMonitor(envelope=Envelope.from_features(lib.nodes))
    report = monitor.evaluate(lib.query(50.0))
    assert report.triggered
    bounds, d_max = monitor.restrict((0.0, 100.0), 5.0, report)
    assert bounds == pytest.approx(lib.omega_range)
    assert d_max < 5.0, "el interruptor debe frenar también la tasa de cambio"


def test_mpc_respects_bounds_and_is_deterministic():
    plant, _, cfg = _plant()
    state = MacroState(M=np.array([1.0, 0, 0, 0, 0, 0]))
    first = MPC(cfg).act(state, plant).u
    second = MPC(cfg).act(state, plant).u
    assert np.allclose(first, second), "el MPC debe ser determinista"
    assert np.all(first[:, 0] >= cfg.omega_min - 1e-9)
    assert np.all(first[:, 0] <= cfg.omega_max + 1e-9)
    assert np.all(first[:, 1] >= cfg.feed_min - 1e-9)
    assert np.all(first[:, 1] <= cfg.feed_max + 1e-9)


def test_mpc_reports_the_asymmetry_between_actions():
    """omega tiene respaldo microdinámico; F_feed no. El reporte debe decirlo."""
    plant, _, cfg = _plant()
    meta = MPC(cfg).act(MacroState(M=np.ones(6)), plant).meta
    assert meta["omega_has_micro_support"] is True
    assert meta["feed_has_micro_support"] is False


def test_closed_loop_conserves_the_mass_balance():
    plant, _, cfg = _plant()
    state = MacroState(M=np.array([1.0, 0, 0, 0, 0, 0]))
    res = closed_loop(ConstantOmega(4.0, 0.0), plant, state, n_steps=5)
    assert len(res["history"]) == 5
    assert res["product_kg"] >= 0.0
    assert np.all(np.array(res["final_state"]) >= 0.0)
    assert sum(res["final_state"]) <= state.holdup + 1e-9  # sin alimentación


def test_spectrum_modes_produce_different_plans():
    state = MacroState(M=np.array([1.0, 0, 0, 0, 0, 0]))
    plans = {}
    for mode in ("library", "static", "fixed_rates"):
        plant, _, cfg = _plant(mode)
        plans[mode] = MPC(cfg).act(state, plant).u[:, 0]
    assert not np.allclose(plans["library"], plans["fixed_rates"]), (
        "si el modo del espectro no cambia el plan, E1 daría un falso negativo"
    )


# ---------------------------------------------------------------------------
# El hallazgo sobre omega(t) del cilindro
# ---------------------------------------------------------------------------

def test_omega_profiles_differ_exactly_where_the_data_discriminates():
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert omega_pdf_literal(t) == pytest.approx(omega_triangular(t))
    # entre 1.0 y 1.5 s los perfiles se separan: uno invierte, el otro reposa
    for t in (1.1, 1.25, 1.4):
        assert omega_pdf_literal(t) < 0.0
        assert omega_triangular(t) == 0.0
    assert omega_pdf_literal(1.6) == 0.0


@pytest.mark.skipif(not _CYL.exists(), reason="dataset no descargado")
def test_cylinder_data_refutes_the_triangular_omega_profile():
    """Ancla del hallazgo: si alguien vuelve al perfil triangular, esto falla.

    Dos señales independientes sobre `Extrapolation_2073Spheres/CASE08`:
    el lecho gira en sentido negativo entre 1.0 y 1.5 s (el perfil triangular
    predice reposo), y el spin propio tiene un segundo máximo en t ≈ 1.5 s.
    """
    scales = default_scales()
    scaling = Scaling.from_slgnn(scales)
    traj = scales.nondim(load_case(_CYL, dt=1e-3, dtype=torch.float64))
    check = verify_omega_profile(traj, scaling)

    assert check["bed_reverses_between_1_0_and_1_5s"], (
        "el lecho invierte el giro: el perfil triangular no puede explicarlo")
    assert check["second_spin_peak_present"], (
        "el segundo pico de spin solo existe si omega vuelve a crecer en módulo")
    assert check["verdict"] == "pdf_literal"


def test_cylinder_sdf_is_built_in_nondimensional_units():
    scaling = Scaling.from_slgnn(default_scales())
    sdf = make_cylinder_sdf(scaling)
    assert sdf.radius == pytest.approx(0.05 / scaling.L)
    assert sdf.z_max == pytest.approx(0.1 / scaling.L)
    # la velocidad angular adimensional escala con T
    t_nd = 0.5 / scaling.T
    assert sdf.omega_fn(t_nd) == pytest.approx(omega_pdf_literal(0.5) * scaling.T)
