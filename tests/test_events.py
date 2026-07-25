"""T0 — el estimador de eventos contra los dos micro-benchmarks (§2.1).

Los dos archivos que ya validan los canales de SLGNN validan también el
estimador de espectro. Si estos tests no pasan, nada de lo que venga después
significa algo.

La verdad de referencia es **analítica e independiente del detector**: antes del
contacto no hay fuerzas en ninguno de los dos benchmarks, de modo que el
movimiento relativo es rectilíneo y el instante de contacto sale de resolver
|r0 + v_rel·t| = R_i + R_j.

Los tests corren en unidades SI. `twin.events` es agnóstico a la escala (solo
exige consistencia); la conversión formal adimensional → SI vive en
`twin.coarse`, según la regla de §6.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from slgnn.data import load_case
from slgnn.sdf import BoxSDF
from twin.events import PP, PW, detect_events, detect_pp_events, detect_pw_events

_DATA = Path(__file__).resolve().parent.parent / "data" / "extracted"
_TWO_SPHERE = _DATA / "Benchmark_2Spheres_Oblique_Collision"
_WALL = _DATA / "Benchmark_1Sphere_Multiple_Wall_Collision"

needs_two_sphere = pytest.mark.skipif(
    not _TWO_SPHERE.exists(), reason="dataset no descargado"
)
needs_wall = pytest.mark.skipif(not _WALL.exists(), reason="dataset no descargado")

# Semiespacio z >= 0: la pared plana del benchmark de 1 esfera. Una caja de
# extensión enorme reproduce exactamente phi(x) = z sin agregar clases nuevas
# a slgnn.sdf.
_HALF_SPACE = BoxSDF([-1e3, -1e3, 0.0], [1e3, 1e3, 1e3])


def _analytic_normal_speed(r0, v_rel, contact_distance):
    """Rapidez normal de aproximación al tocarse, para movimiento rectilíneo."""
    a = float(v_rel @ v_rel)
    b = 2.0 * float(r0 @ v_rel)
    c = float(r0 @ r0) - contact_distance**2
    disc = b * b - 4 * a * c
    assert disc > 0, "las trayectorias rectilíneas no llegan a tocarse"
    t_star = (-b - math.sqrt(disc)) / (2 * a)
    r_star = r0 + v_rel * t_star
    e = r_star / np.linalg.norm(r_star)
    return -float(v_rel @ e)


# ---------------------------------------------------------------------------
# Benchmark 1: partícula-partícula
# ---------------------------------------------------------------------------

@needs_two_sphere
@pytest.mark.parametrize("subcase", ["1x", "2x", "4x"])
def test_two_sphere_single_event_with_correct_energy(subcase):
    tr = load_case(_TWO_SPHERE / subcase, dt=1e-4, dtype=torch.float64)
    ev = detect_pp_events(tr.q, tr.v, tr.radii, tr.m, delta=0.0, max_lookback=32,
                          dt=tr.dt)

    assert len(ev) == 1, f"{subcase}: se esperaba exactamente un evento, hubo {len(ev)}"
    assert ev.kind[0] == PP

    q0, v0 = tr.q[0].numpy(), tr.v[0].numpy()
    m = tr.m.numpy()
    v_n_true = _analytic_normal_speed(
        q0[1] - q0[0], v0[1] - v0[0], float(tr.radii[0] + tr.radii[1])
    )
    m_eff = m[0] * m[1] / (m[0] + m[1])
    e_true = 0.5 * m_eff * v_n_true**2

    assert ev.v_n[0] == pytest.approx(v_n_true, rel=0.02)
    assert ev.m_eff[0] == pytest.approx(m_eff, rel=1e-12)
    assert ev.E_impact[0] == pytest.approx(e_true, rel=0.02)


@needs_two_sphere
def test_two_sphere_energies_separate_by_scale():
    """Los tres subcasos deben caer en bins log-espaciados distintos."""
    energies = []
    for subcase in ["1x", "2x", "4x"]:
        tr = load_case(_TWO_SPHERE / subcase, dt=1e-4, dtype=torch.float64)
        ev = detect_pp_events(tr.q, tr.v, tr.radii, tr.m, delta=0.0, dt=tr.dt)
        energies.append(float(ev.E_impact[0]))

    assert energies[0] < energies[1] < energies[2]
    # bins de razón 2 en energía: cada subcase cae en un bin distinto
    bins = [math.floor(math.log2(e)) for e in energies]
    assert len(set(bins)) == 3, f"energías no separables en bins log2: {energies}"


@needs_two_sphere
def test_two_sphere_uses_free_flight_velocity_not_k_pre_one():
    """Regresión sobre la desviación documentada respecto de la especificación.

    El contacto abarca ~11 snapshots, así que leer la velocidad un snapshot
    antes del mínimo la subestima groseramente. El onset debe caer antes de que
    empiece el solapamiento.
    """
    tr = load_case(_TWO_SPHERE / "1x", dt=1e-4, dtype=torch.float64)
    ev = detect_pp_events(tr.q, tr.v, tr.radii, tr.m, delta=0.0, dt=tr.dt)

    assert ev.step[0] - ev.onset[0] > 1, "el onset debe preceder al contacto, no al mínimo"
    d = torch.linalg.vector_norm(tr.q[:, 0] - tr.q[:, 1], dim=-1).numpy()
    contact = float(tr.radii[0] + tr.radii[1])
    assert d[ev.onset[0]] >= contact, "el onset debe ser un snapshot de vuelo libre"
    assert d[ev.onset[0] + 1] < contact, "el onset debe ser el último de vuelo libre"


# ---------------------------------------------------------------------------
# Benchmark 2: partícula-pared
# ---------------------------------------------------------------------------

@needs_wall
@pytest.mark.parametrize("angle", ["10", "30", "45", "60", "90"])
def test_wall_single_event_with_correct_normal_energy(angle):
    tr = load_case(_WALL / angle, dt=1e-4, dtype=torch.float64)
    ev = detect_pw_events(tr.q, tr.v, tr.radii, tr.m, _HALF_SPACE,
                          delta=0.0, max_lookback=32, dt=tr.dt)

    assert len(ev) == 1, f"{angle}°: se esperaba un evento, hubo {len(ev)}"
    assert ev.kind[0] == PW
    assert ev.j[0] == -1

    # sin gravedad: v_z es constante hasta el contacto, y la normal es +z
    v_n_true = -float(tr.v[0, 0, 2])
    assert ev.v_n[0] == pytest.approx(v_n_true, rel=1e-6)
    assert ev.E_impact[0] == pytest.approx(0.5 * float(tr.m[0]) * v_n_true**2, rel=1e-6)
    # la componente tangencial se separa correctamente
    assert ev.v_t[0] == pytest.approx(
        float(torch.linalg.vector_norm(tr.v[0, 0, :2])), rel=1e-6
    )


@needs_wall
def test_wall_normal_energy_is_maximal_at_head_on():
    """Discrepancia documentada con el criterio de aceptación de §2.1.

    Las instrucciones piden `E_normal` monótona creciente con la etiqueta de
    ángulo. Los datos dicen otra cosa: las etiquetas 10–60 son el ángulo
    respecto de la **normal** (v0 = 2·(sin θ, 0, −cos θ)), de modo que la
    energía normal *decrece* con la etiqueta. La etiqueta 90 es un impacto
    frontal, v0 = (0, 0, −2), y por eso sí es el máximo. Lo que se verifica
    aquí es la física real, no la monotonía supuesta.
    """
    energies, tangential = {}, {}
    for angle in ["10", "30", "45", "60", "90"]:
        tr = load_case(_WALL / angle, dt=1e-4, dtype=torch.float64)
        ev = detect_pw_events(tr.q, tr.v, tr.radii, tr.m, _HALF_SPACE, delta=0.0,
                              dt=tr.dt)
        energies[angle] = float(ev.E_impact[0])
        tangential[angle] = float(ev.v_t[0])

    assert max(energies, key=energies.get) == "90"
    # entre 10 y 60 la energía normal decrece y la tangencial crece
    oblique = ["10", "30", "45", "60"]
    assert [energies[a] for a in oblique] == sorted(
        (energies[a] for a in oblique), reverse=True
    )
    assert [tangential[a] for a in oblique] == sorted(tangential[a] for a in oblique)


@needs_wall
def test_wall_benchmark_has_no_particle_particle_events():
    tr = load_case(_WALL / "45", dt=1e-4, dtype=torch.float64)
    ev = detect_events(tr.q, tr.v, tr.radii, tr.m, _HALF_SPACE, delta=0.0, dt=tr.dt)
    assert len(ev.of_kind(PP)) == 0
    assert len(ev.of_kind(PW)) == 1


# ---------------------------------------------------------------------------
# Contratos sintéticos del detector
# ---------------------------------------------------------------------------

def _head_on_pair(n_pre=10, v=1.0, dt=1e-3, radius=0.5, penetration=2e-3):
    """Dos esferas que se aproximan de frente, penetran y rebotan.

    Perfil triangular del gap: baja a rapidez `v`, alcanza `-penetration` y
    vuelve a subir. La partícula 0 está fija; la 1 se mueve sobre +x.
    """
    t = np.arange(2 * n_pre + 1)
    gap = v * dt * np.abs(t - n_pre) - penetration
    pos = np.zeros((t.size, 2, 3))
    pos[:, 1, 0] = 2 * radius + gap
    vel = np.zeros((t.size, 2, 3))
    vel[:, 1, 0] = np.where(t < n_pre, -v, v)
    return pos, vel, np.full(2, radius), np.full(2, 2.0)


def test_head_on_energy_matches_closed_form():
    pos, vel, radii, m = _head_on_pair()
    ev = detect_pp_events(pos, vel, radii, m, delta=0.0, dt=1e-3)
    assert len(ev) == 1
    m_eff = m[0] * m[1] / (m[0] + m[1])
    assert ev.v_n[0] == pytest.approx(1.0, rel=1e-9)
    assert ev.E_impact[0] == pytest.approx(0.5 * m_eff * 1.0**2, rel=1e-9)


def test_detector_is_invariant_to_particle_permutation():
    pos, vel, radii, m = _head_on_pair()
    base = detect_pp_events(pos, vel, radii, m, delta=0.0, dt=1e-3)
    perm = [1, 0]
    flipped = detect_pp_events(pos[:, perm], vel[:, perm], radii[perm], m[perm],
                               delta=0.0, dt=1e-3)
    assert len(base) == len(flipped) == 1
    assert flipped.E_impact[0] == pytest.approx(base.E_impact[0], rel=1e-12)
    assert flipped.step[0] == base.step[0]


def test_sustained_contact_is_discarded_not_counted_as_impact():
    """Una partícula apoyada sobre otra no genera impactos: genera carga."""
    n = 60
    pos = np.zeros((n, 2, 3))
    pos[:, 1, 0] = 0.999 + 1e-6 * np.sin(np.arange(n))  # solapamiento permanente
    vel = np.zeros((n, 2, 3))
    vel[:, 1, 0] = 1e-6 * np.cos(np.arange(n))
    ev = detect_pp_events(pos, vel, np.full(2, 0.5), np.full(2, 1.0),
                          delta=0.0, max_lookback=8, dt=1e-3)
    assert len(ev) == 0
    assert ev.n_unresolved > 0, "el contacto sostenido debe reportarse, no ignorarse"


def test_no_events_when_particles_never_touch():
    pos = np.zeros((20, 2, 3))
    pos[:, 1, 0] = 5.0
    vel = np.zeros((20, 2, 3))
    ev = detect_pp_events(pos, vel, np.full(2, 0.5), np.full(2, 1.0), delta=0.0,
                          dt=1e-3)
    assert len(ev) == 0
