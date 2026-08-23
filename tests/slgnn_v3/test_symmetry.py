"""Simetrías y conservaciones del paso completo (§11 de la formulación).

La equivarianza que se exige es **condicionada**: se transforma el sistema
físico entero —partículas, gravedad y pared—, no solo las partículas. Rotar
las partículas dejando la gravedad fija describe otro experimento y no tiene
por qué dar la respuesta rotada (§11.2).
"""

import pytest
import torch

from slgnn_v3 import ParticleBatch, RouterProfile, V3State, box_surfaces, half_space
from slgnn_v3.config import DissipationConfig
from slgnn_v3.contact_operator import JT_times_contact_vector
from slgnn_v3.potential import conservative_force

from .conftest import (
    DTYPE,
    build_set,
    default_box,
    make_particles,
    random_particles,
    random_rotation,
    rel_error,
    small_model,
)


def _rotated_box(Q, a):
    """Caja rotada: seis planos con normal `Q n` y offset `n.(Q^T a) + d`."""
    from slgnn_v3.surfaces import Plane, SurfaceSet

    base = default_box().surfaces
    out = []
    for s in base:
        n = torch.tensor(s.inward_normal, dtype=DTYPE)
        n_rot = Q @ n
        out.append(Plane(tuple(n_rot.tolist()), s.offset + float(n_rot @ a), name=s.name))
    return SurfaceSet(out)


@pytest.mark.parametrize("profile", [RouterProfile.COMPLIANT, RouterProfile.IMPULSIVE])
def test_se3_equivariance_of_the_full_step(profile):
    """`epsilon_equiv = ||y(Qx+a) - Q y(x)|| / ||y(x)|| <= 1e-8`."""
    model = small_model(profile=profile, seed=3)
    Q, a = random_rotation(17), torch.tensor([0.7, -0.4, 0.25], dtype=DTYPE)
    pb = make_particles(
        [[2.0, 2.0, 0.45], [2.0, 2.0, 1.4], [2.7, 2.2, 1.9]],
        v=[[0.1, 0, -0.4], [0, 0.2, -0.5], [-0.3, 0, 0.1]],
        omega=[[0.2, 0, 0], [0, 0.1, 0], [0, 0, 0.3]], radius=0.5,
    )
    g = torch.tensor([0.0, 0.0, -0.98], dtype=DTYPE)

    model.reset_lifecycle()
    base = model.step(V3State(pb, time=0.2), 0.05, default_box(), g)

    pb_rot = ParticleBatch.from_arrays(
        pb.q @ Q.T + a, pb.v @ Q.T, pb.omega @ Q.T, pb.mass, pb.radius,
        inertia=pb.inertia,
    )
    model.reset_lifecycle()
    rot = model.step(V3State(pb_rot, time=0.2), 0.05, _rotated_box(Q, a), g @ Q.T)

    assert rel_error(rot.delta_p, base.delta_p @ Q.T) <= 1e-8
    # `Delta L` es exactamente cero en el MVP normal (ver el test dedicado más
    # abajo), así que se mide contra la escala de `Delta p`: un cociente puro
    # dividiría por cero.
    scale = float(base.delta_p.detach().norm())
    assert rel_error(rot.delta_L, base.delta_L @ Q.T, floor=scale) <= 1e-8
    q_expected = base.next_state.particles.q @ Q.T + a
    assert rel_error(rot.next_state.particles.q, q_expected) <= 1e-8


@pytest.mark.parametrize("profile", [RouterProfile.COMPLIANT, RouterProfile.IMPULSIVE])
def test_permutation_equivariance_of_the_full_step(profile):
    model = small_model(profile=profile, seed=4)
    pb = make_particles(
        [[2.0, 2.0, 0.45], [2.0, 2.0, 1.4], [2.7, 2.2, 1.9], [1.2, 2.0, 2.5]],
        v=[[0.1, 0, -0.4], [0, 0.2, -0.5], [-0.3, 0, 0.1], [0.2, 0.1, 0]],
        omega=[[0.2, 0, 0], [0, 0.1, 0], [0, 0, 0.3], [0.1, 0.1, 0]], radius=0.5,
    )
    g = torch.tensor([0.0, 0.0, -0.98], dtype=DTYPE)
    perm = torch.tensor([2, 0, 3, 1])

    model.reset_lifecycle()
    base = model.step(V3State(pb, time=0.0), 0.05, default_box(), g)
    pb_perm = ParticleBatch.from_arrays(
        pb.q[perm], pb.v[perm], pb.omega[perm], pb.mass[perm], pb.radius[perm],
        inertia=pb.inertia[perm],
    )
    model.reset_lifecycle()
    other = model.step(V3State(pb_perm, time=0.0), 0.05, default_box(), g)
    assert rel_error(other.delta_p, base.delta_p[perm]) <= 1e-10
    assert rel_error(other.delta_L, base.delta_L[perm]) <= 1e-10


def test_potential_conserves_internal_momentum_structurally():
    """`V` no aplica `J^T`, así que su conservación se verifica aparte.

    `V_pp` depende de las posiciones solo a través de `d_ij`, luego
    `-grad V_pp` es automáticamente igual y opuesto a lo largo de `n`, y su
    torque respecto de cualquier origen se cancela.
    """
    model = small_model(seed=5)
    pb = random_particles(6, seed=21, box=(1.0, 2.6)).requires_grad_q()
    cs = build_set(pb, cfg=model.cfg)
    pp = cs.subset(~cs.is_wall)
    h_node, h_edge, _ = model.encoder(pp, pb)
    V, _ = model.head_V.total_potential(pp, model.proc_V(pp, h_node, h_edge),
                                        torch.ones_like(pp.gap))
    F = conservative_force(V, pb.q, create_graph=False)
    assert float(F.sum(dim=0).detach().abs().max()) <= 1e-12
    L = torch.linalg.cross(pb.q.detach(), F, dim=-1).sum(dim=0)
    scale = torch.linalg.cross(pb.q.detach(), F, dim=-1).norm(dim=-1).sum().clamp_min(1e-30)
    assert float((L.norm() / scale).detach()) <= 1e-10


@pytest.mark.parametrize("profile", [RouterProfile.COMPLIANT, RouterProfile.IMPULSIVE])
def test_internal_momentum_conserved_without_walls(profile):
    """Sistema aislado sin paredes: momento lineal y angular exactos."""
    from slgnn_v3.surfaces import SurfaceSet

    model = small_model(profile=profile, seed=6)
    pb = make_particles(
        [[0.0, 0.0, 0.0], [0.94, 0.1, 0.0], [1.8, 0.6, 0.2]],
        v=[[1.0, 0, 0], [-0.4, 0.1, 0], [0.0, -0.3, 0.1]],
        omega=[[0.2, 0, 0], [0, 0.3, 0], [0, 0, -0.1]], radius=0.5,
    )
    res = model.step(V3State(pb, time=0.0), 0.05, SurfaceSet([]))
    p = res.next_state.particles
    dp = pb.mass.unsqueeze(-1) * (p.v - pb.v)
    dL = pb.inertia.unsqueeze(-1) * (p.omega - pb.omega)
    assert float(dp.detach().sum(dim=0).abs().max()) <= 1e-12
    total = (torch.linalg.cross(pb.q, dp, dim=-1) + dL).sum(dim=0)
    assert float(total.detach().abs().max()) <= 1e-8


def test_mvp_produces_exactly_zero_angular_momentum():
    """Límite declarado del MVP normal: sin canal tangencial no hay torque.

    `Delta L = 0` **exactamente**, no aproximadamente. Se afirma como test para
    que ningún informe pueda presentar un `Delta L` predicho como si el modelo
    tuviera física rotacional: la fricción y el spin llegan en la fase 9.
    """
    for profile in (RouterProfile.COMPLIANT, RouterProfile.IMPULSIVE):
        model = small_model(profile=profile, seed=8)
        pb = make_particles(
            [[2.0, 2.0, 0.45], [2.0, 2.0, 1.4]],
            v=[[0.4, 0.1, -0.4], [0, 0.2, -0.5]],
            omega=[[0.9, 0.3, 0], [0.2, 0.1, 0.4]], radius=0.5,
        )
        res = model.step(V3State(pb, time=0.0), 0.05, default_box(),
                         torch.tensor([0.0, 0.0, -0.98], dtype=DTYPE))
        assert float(res.delta_L.detach().abs().max()) <= 1e-18


def test_tangential_compliant_step_is_se3_equivariant():
    """El nuevo canal de spin rota como vector y conserva la forma del paso."""
    from slgnn_v3.surfaces import SurfaceSet

    model = small_model(
        profile=RouterProfile.COMPLIANT,
        seed=18,
        dissipation=DissipationConfig(
            tangential=True, state_independent_coefficients=True
        ),
    )
    pb = make_particles(
        [[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [1.75, 0.25, 0.1]],
        v=[[0.0, 0.8, 0.1], [0.2, -0.1, 0.0], [-0.1, 0.3, -0.2]],
        omega=[[0.1, 0.2, 0.3], [-0.2, 0.1, 0.0], [0.0, -0.1, 0.2]],
        radius=0.5,
    )
    Q = random_rotation(29)
    a = torch.tensor([0.4, -0.2, 0.7], dtype=DTYPE)
    pb_rot = ParticleBatch.from_arrays(
        pb.q @ Q.T + a,
        pb.v @ Q.T,
        pb.omega @ Q.T,
        pb.mass,
        pb.radius,
        inertia=pb.inertia,
    )
    empty = SurfaceSet([])
    model.reset_lifecycle()
    base = model.step(V3State(pb), 0.02, empty)
    model.reset_lifecycle()
    rot = model.step(V3State(pb_rot), 0.02, empty)
    assert rel_error(rot.delta_p, base.delta_p @ Q.T) <= 1e-8
    assert rel_error(rot.delta_L, base.delta_L @ Q.T) <= 1e-8
    assert float(base.delta_L.detach().norm()) > 1e-8


def test_moving_wall_changes_the_answer():
    """La velocidad de pared no es decoración: cambia el resultado."""
    from slgnn_v3.surfaces import Plane, SurfaceSet, WallMotion

    model = small_model(profile=RouterProfile.IMPULSIVE, seed=7)
    pb = make_particles([[0.0, 0.0, 0.45]], v=[[0.0, 0.0, -1.0]],
                        omega=[[0.0, 0.0, 0.0]], radius=0.5)
    still = SurfaceSet([Plane((0.0, 0.0, 1.0), 0.0, name="floor")])
    moving = SurfaceSet([Plane(
        (0.0, 0.0, 1.0), 0.0, name="floor",
        motion=WallMotion(center=(0.0, 0.0, 0.0), velocity_fn=lambda t: (0.0, 0.0, 1.0)),
    )])
    model.reset_lifecycle()
    a = model.step(V3State(pb, time=0.0), 0.05, still)
    model.reset_lifecycle()
    b = model.step(V3State(pb, time=0.0), 0.05, moving)
    assert float((a.delta_p - b.delta_p).detach().abs().max()) > 1e-6
