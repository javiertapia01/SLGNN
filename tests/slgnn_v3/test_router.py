"""Router determinista y ciclo de vida: sin doble conteo, restitución solo al
nacer, y `v3-H` que falla en vez de degradarse en silencio."""

import pytest
import torch

from slgnn_v3 import RouterProfile, V3Config, V3State, box_surfaces
from slgnn_v3.config import RouterConfig
from slgnn_v3.router import (
    ContactLifecycle,
    ContactMode,
    assert_no_double_counting,
    build_router,
    impulsive_mask,
    regular_weight,
)

from .conftest import DTYPE, build_set, default_box, make_particles, small_model


def test_hybrid_profile_fails_loudly():
    """§9.3: `v3-H` debe fallar con un mensaje claro, sin fallback silencioso."""
    cfg = RouterConfig(profile=RouterProfile.HYBRID)
    with pytest.raises(NotImplementedError) as e:
        build_router(cfg)
    msg = str(e.value)
    assert "fricción impulsiva" in msg
    assert "memoria tangencial" in msg
    assert "fallback silencioso" in msg


def test_router_modes_are_disjoint():
    for profile in (RouterProfile.COMPLIANT, RouterProfile.IMPULSIVE):
        model = small_model(profile=profile, seed=1)
        pb = make_particles([[2.0, 2.0, 0.45], [2.0, 2.0, 1.35]], radius=0.5)
        res = model.step(V3State(pb, time=0.0), 0.05, default_box(),
                         torch.tensor([0.0, 0.0, -0.98], dtype=DTYPE))
        assert_no_double_counting(res.mode)
        w = regular_weight(res.mode, DTYPE)
        m = impulsive_mask(res.mode).to(DTYPE)
        assert float((w * m).max()) == 0.0      # nunca ambos a la vez


def test_compliant_profile_produces_no_impulses():
    model = small_model(profile=RouterProfile.COMPLIANT, seed=2)
    pb = make_particles([[2.0, 2.0, 0.4]], v=[[0.0, 0.0, -2.0]], radius=0.5)
    res = model.step(V3State(pb, time=0.0), 0.05, default_box())
    assert res.diagnostics.router["n_impulsive"] == 0
    assert res.impulses.numel() == 0
    assert float(res.delta_p_impulse.abs().max()) == 0.0


def test_impulsive_profile_produces_no_potential():
    model = small_model(profile=RouterProfile.IMPULSIVE, seed=3)
    pb = make_particles([[2.0, 2.0, 0.4]], v=[[0.0, 0.0, -2.0]], radius=0.5)
    res = model.step(V3State(pb, time=0.0), 0.05, default_box())
    assert res.diagnostics.router["n_compliant"] == 0
    assert res.diagnostics.energies["V_total"] == 0.0
    assert res.diagnostics.dissipation["Psi_n"] == 0.0


def test_lifecycle_birth_then_persistence():
    """§18.8: un contacto persistente no vuelve a nacer en cada frame."""
    cfg = RouterConfig(g_on=0.0, g_off=0.05, n_grace=1)
    life = ContactLifecycle(cfg)
    model_cfg = V3Config()
    box = default_box()

    def contacts_at(z):
        pb = make_particles([[2.0, 2.0, z]], radius=0.5)
        return build_set(pb, box, model_cfg)

    seen_births = []
    for z in (0.60, 0.48, 0.47, 0.46, 0.45):
        cs = contacts_at(z)
        floor = (cs.surface == 4).nonzero().flatten()
        _, birth, age = life.step(cs)
        seen_births.append(bool(birth[floor]))
    assert seen_births == [False, True, False, False, False]


def test_lifecycle_hysteresis_prevents_flicker():
    """Ruido alrededor de `g = 0` no debe crear y destruir el contacto."""
    life = ContactLifecycle(RouterConfig(g_on=0.0, g_off=0.05, n_grace=1))
    box, cfg = default_box(), V3Config()
    births = 0
    for z in (0.49, 0.502, 0.498, 0.503, 0.497):     # g oscila entre -0.01 y +0.003
        pb = make_particles([[2.0, 2.0, z]], radius=0.5)
        _, birth, _ = life.step(build_set(pb, box, cfg))
        births += int(birth.sum())
    assert births == 1


def test_lifecycle_grace_then_new_contact():
    life = ContactLifecycle(RouterConfig(g_on=0.0, g_off=0.05, n_grace=1))
    box, cfg = default_box(), V3Config()
    seq = [0.45, 2.0, 2.0, 2.0, 0.45]     # contacto, separación larga, contacto
    births = []
    for z in seq:
        pb = make_particles([[2.0, 2.0, z]], radius=0.5)
        cs = build_set(pb, box, cfg)
        _, birth, _ = life.step(cs)
        births.append(int(birth.sum()))
    assert births[0] == 1
    assert births[-1] == 1        # reaparición = contacto nuevo


def test_restitution_applied_only_at_birth():
    """El impulso de restitución solo actúa en el paso de nacimiento."""
    model = small_model(profile=RouterProfile.IMPULSIVE, seed=4)
    box = default_box()
    pb = make_particles([[2.0, 2.0, 0.55]], v=[[0.0, 0.0, -1.0]], radius=0.5)
    st = V3State(pb, time=0.0)
    applied = []
    for _ in range(4):
        res = model.step(st, 0.05, box, torch.tensor([0.0, 0.0, -0.98], dtype=DTYPE))
        applied.append(res.diagnostics.impact.get("restitution_applied", 0))
        st = res.next_state
    assert sum(applied) <= 1, f"restitución aplicada más de una vez: {applied}"


def test_free_flight_produces_no_contact_response():
    model = small_model(seed=5)
    pb = make_particles([[2.5, 2.5, 2.5]], v=[[0.1, 0.0, 0.0]], radius=0.5)
    res = model.step(V3State(pb, time=0.0), 0.05, default_box())
    assert res.diagnostics.geometry["n_candidates"] == 0
    assert float(res.forces.detach().abs().max()) == 0.0
