"""T2 — contratos del PBM (§7.4).

El primero de estos tests no es una formalidad: si `S_b` no responde a la forma
del espectro, el PBM es ciego al puente micro-macro y E1 daría un falso negativo
por construcción.
"""

import numpy as np
import pytest

from twin.coarse import CoarseConfig, Spectrum
from twin.macro import (PBM, MacroState, PBMConfig, progeny_matrix,
                        selection_rates, sensitivity_to_shape)


def _template(n_bins=30, e_min=1e-12, e_max=1e-2) -> Spectrum:
    cfg = CoarseConfig(n_bins=n_bins, e_min=e_min, e_max=e_max)
    return Spectrum(edges=cfg.bin_edges(), rates=np.zeros((n_bins, 2)),
                    duration=1.0, mass=1.0, underflow=np.zeros(2),
                    overflow=np.zeros(2))


# ---------------------------------------------------------------------------
# El chequeo previo indispensable
# ---------------------------------------------------------------------------

def test_selection_rates_respond_to_spectrum_shape_not_only_total_energy():
    cfg = PBMConfig()
    result = sensitivity_to_shape(cfg, _template())
    assert result["energy_soft"] == pytest.approx(result["energy_hard"], rel=1e-9)
    assert result["shape_sensitive"], (
        "S_b debe distinguir dos espectros de igual energía total y distinta "
        f"forma; razones observadas {result['ratio_per_class']}"
    )


def test_selection_rate_is_monotone_in_impact_energy():
    cfg = PBMConfig()
    tpl = _template()
    rates = np.zeros_like(tpl.rates)
    values = []
    for k in (8, 12, 16, 20):
        r = rates.copy()
        r[k, 0] = 1.0
        values.append(selection_rates(tpl.with_rates(r), cfg))
    for a, b in zip(values, values[1:]):
        assert np.all(b >= a - 1e-12), "más energía por impacto no puede romper menos"


def test_coarse_classes_are_selected_faster_than_fine_ones():
    cfg = PBMConfig()
    tpl = _template()
    rates = np.zeros_like(tpl.rates)
    rates[18, 0] = 1e5
    S = selection_rates(tpl.with_rates(rates), cfg)
    assert np.all(np.diff(S) < 0), f"S_b debe decrecer con el tamaño: {S}"


# ---------------------------------------------------------------------------
# Conservación de masa
# ---------------------------------------------------------------------------

def test_progeny_columns_sum_to_one_exactly():
    mat = progeny_matrix(PBMConfig())
    for j in range(PBMConfig().n_classes - 1):
        assert mat[:, j].sum() == pytest.approx(1.0, abs=1e-15)
    assert np.allclose(np.triu(mat), 0.0), "la progenie solo puede ir a clases más finas"
    assert mat[:, -1].sum() == 0.0, "la clase más fina no se rompe más"


def test_mass_is_conserved_to_machine_precision_without_feed_or_discharge():
    cfg = PBMConfig(k_discharge=0.0)
    pbm = PBM(cfg)
    state = MacroState(M=np.array([1.0, 0.5, 0.25, 0.1, 0.05, 0.0]))
    S = np.array([2.0, 1.5, 1.0, 0.7, 0.3, 0.0])
    total0 = state.holdup
    for _ in range(50):
        state = pbm.step(state, S, f_feed=0.0, dt=0.05)
    assert state.holdup == pytest.approx(total0, rel=1e-13)
    assert np.all(state.M >= 0.0)


def test_holdup_balance_with_feed_and_discharge():
    cfg = PBMConfig()
    pbm = PBM(cfg)
    state = MacroState(M=np.full(cfg.n_classes, 0.1))
    f_feed, dt = 0.02, 0.01
    before = state.holdup
    out = pbm.discharge_rate(state)
    after = pbm.step(state, np.zeros(cfg.n_classes), f_feed, dt).holdup
    assert (after - before) / dt == pytest.approx(f_feed - out, rel=2e-2)


# ---------------------------------------------------------------------------
# Comportamiento físico
# ---------------------------------------------------------------------------

def test_p80_decreases_with_supplied_energy():
    cfg = PBMConfig(k_discharge=0.0)
    pbm = PBM(cfg)
    tpl = _template()
    rates = np.zeros_like(tpl.rates)
    p80s = []
    for amplitude in (1e3, 1e4, 1e5):
        r = rates.copy()
        r[20, 0] = amplitude
        S = selection_rates(tpl.with_rates(r), cfg)
        state = MacroState(M=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        for _ in range(20):
            state = pbm.step(state, S, f_feed=0.0, dt=0.05)
        p80s.append(pbm.p80(state))
    assert p80s[0] > p80s[1] > p80s[2], f"P80 debe decrecer con la energía: {p80s}"


def test_integrator_is_stable_for_growing_macro_dt():
    cfg = PBMConfig()
    pbm = PBM(cfg)
    S = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    finals = []
    for dt in (0.01, 0.05, 0.2, 1.0):
        state = MacroState(M=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        n = int(round(2.0 / dt))
        for _ in range(n):
            state = pbm.step(state, S, f_feed=0.01, dt=dt, substeps=8)
        assert np.all(np.isfinite(state.M)), f"divergencia con dt={dt}"
        assert np.all(state.M >= 0.0)
        finals.append(state.holdup)
    spread = (max(finals) - min(finals)) / np.mean(finals)
    assert spread < 0.05, f"el holdup final no debería depender del dt macro: {finals}"


def test_product_rate_counts_only_material_under_target_size():
    cfg = PBMConfig(d_target=1.0e-3)
    pbm = PBM(cfg)
    coarse_only = MacroState(M=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    fine_only = MacroState(M=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    assert pbm.product_rate(coarse_only) == 0.0
    assert pbm.product_rate(fine_only) > 0.0
