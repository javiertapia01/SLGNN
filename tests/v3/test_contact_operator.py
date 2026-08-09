"""Operadores `J`, `J^T` y Delassus: las 5 pruebas obligatorias de §5.3."""

import pytest
import torch

from slgnn_v3 import ParticleBatch, V3Config
from slgnn_v3.contact_operator import (
    JT_times_contact_vector,
    J_normal,
    J_times_velocity,
    assemble_normal_delassus,
    connected_components,
    pack_by_component,
    unpack_by_component,
)

from .conftest import (
    DTYPE,
    build_set,
    default_box,
    make_particles,
    random_particles,
    random_rotation,
    rel_error,
)


def test_J_matches_direct_construction():
    """1. `J nu` coincide con la velocidad construida directamente."""
    pb = random_particles(8, seed=5, box=(0.6, 3.0))
    cs = build_set(pb, default_box())
    u = J_times_velocity(cs, pb.v, pb.omega)
    assert float((u - cs.u).abs().max()) == 0.0


def test_adjoint_identity():
    """2. `<J nu, lambda> = <nu, J^T lambda>` con error relativo <= 1e-10."""
    pb = random_particles(9, seed=6, box=(0.6, 3.0))
    cs = build_set(pb, default_box())
    g = torch.Generator().manual_seed(21)
    lam = torch.randn(cs.n_contacts, 3, generator=g, dtype=DTYPE)
    lhs = (J_times_velocity(cs, pb.v, pb.omega, subtract_wall=False) * lam).sum()
    F, T = JT_times_contact_vector(cs, lam, pb.n)
    rhs = (pb.v * F).sum() + (pb.omega * T).sum()
    assert abs(float(lhs - rhs)) / (abs(float(lhs)) + 1e-30) <= 1e-10


def test_internal_contact_is_equal_and_opposite():
    """3. Un contacto interno aplica fuerzas/impulsos iguales y opuestos."""
    pb = random_particles(8, seed=7, box=(0.6, 2.5))
    cs = build_set(pb)
    sub = cs.subset(~cs.is_wall)
    g = torch.Generator().manual_seed(31)
    lam = torch.randn(sub.n_contacts, 3, generator=g, dtype=DTYPE)
    F, _ = JT_times_contact_vector(sub, lam, pb.n)
    assert float(F.sum(dim=0).abs().max()) <= 1e-14


def test_angular_momentum_orbital_plus_spin():
    """4. El punto común y los torques conservan momento angular total."""
    pb = random_particles(8, seed=8, box=(0.6, 2.5))
    sub = build_set(pb).subset(~build_set(pb).is_wall)
    g = torch.Generator().manual_seed(41)
    lam = torch.randn(sub.n_contacts, 3, generator=g, dtype=DTYPE)
    F, T = JT_times_contact_vector(sub, lam, pb.n)
    L = (torch.linalg.cross(pb.q, F, dim=-1) + T).sum(dim=0)
    scale = (torch.linalg.cross(pb.q, F, dim=-1).norm(dim=-1).sum() + T.norm(dim=-1).sum())
    assert float(L.norm() / scale) <= 1e-10


def test_permutation_invariance():
    """5a. El resultado es invariante a permutación de IDs de partícula."""
    pb = random_particles(7, seed=9, box=(0.6, 2.5))
    perm = torch.randperm(pb.n, generator=torch.Generator().manual_seed(3))
    inv = torch.argsort(perm)
    pb2 = ParticleBatch.from_arrays(
        pb.q[perm], pb.v[perm], pb.omega[perm], pb.mass[perm], pb.radius[perm],
        inertia=pb.inertia[perm],
    )
    box = default_box()
    cs, cs2 = build_set(pb, box), build_set(pb2, box)
    assert cs.n_contacts == cs2.n_contacts
    # los gaps son un multiconjunto invariante
    a = torch.sort(cs.gap).values
    b = torch.sort(cs2.gap).values
    assert float((a - b).abs().max()) <= 1e-12


def test_se3_equivariance_of_operators():
    """5b. Rotar y trasladar todo el sistema rota fuerzas y torques."""
    Q = random_rotation(13)
    pb = random_particles(7, seed=10, box=(1.2, 3.0))
    a = torch.tensor([0.4, -0.3, 0.2], dtype=DTYPE)
    pb_rot = ParticleBatch.from_arrays(
        pb.q @ Q.T + a, pb.v @ Q.T, pb.omega @ Q.T, pb.mass, pb.radius,
        inertia=pb.inertia,
    )
    cs, cs_rot = build_set(pb), build_set(pb_rot)
    g = torch.Generator().manual_seed(51)
    lam = torch.randn(cs.n_contacts, 3, generator=g, dtype=DTYPE)
    F, T = JT_times_contact_vector(cs, lam, pb.n)
    F2, T2 = JT_times_contact_vector(cs_rot, lam @ Q.T, pb.n)
    assert rel_error(F2, F @ Q.T) <= 1e-8
    assert rel_error(T2, T @ Q.T) <= 1e-8


def test_delassus_symmetric_and_psd():
    pb = random_particles(9, seed=11, box=(0.6, 2.5))
    cs = build_set(pb, default_box())
    lay = connected_components(cs, pb.n)
    A = assemble_normal_delassus(cs, pb.mass, pb.inertia, lay)
    assert float((A - A.transpose(1, 2)).abs().max()) == 0.0
    ev = torch.linalg.eigvalsh(A)
    assert float(ev.min()) >= -1e-12


def test_delassus_matches_operator_application():
    """`A x` coincide con `J M^-1 J^T` aplicado y proyectado sobre normales."""
    pb = random_particles(9, seed=12, box=(0.6, 2.5))
    cs = build_set(pb, default_box())
    lay = connected_components(cs, pb.n)
    A = assemble_normal_delassus(cs, pb.mass, pb.inertia, lay)
    x = torch.randn(cs.n_contacts, generator=torch.Generator().manual_seed(61), dtype=DTYPE)
    F, T = JT_times_contact_vector(cs, x.unsqueeze(-1) * cs.n, pb.n)
    direct = J_normal(cs.subset(torch.ones(cs.n_contacts, dtype=torch.bool)),
                      F / pb.mass.unsqueeze(-1), T / pb.inertia.unsqueeze(-1))
    # J_normal resta v_W; aquí queremos la parte homogénea
    direct = direct + (cs.wall_velocity * cs.n).sum(-1)
    via_A = unpack_by_component(
        torch.einsum("kab,kb->ka", A, pack_by_component(x, lay)), lay
    )
    assert float((direct - via_A).abs().max()) <= 1e-12


def test_components_couple_shared_particles():
    """Un contacto de pared y uno pp sobre la misma esfera caen en la misma
    componente: es lo que un solve por pares pierde."""
    q = torch.tensor([[2.0, 2.0, 0.5], [2.0, 2.0, 1.45]], dtype=DTYPE)
    pb = make_particles(q, radius=0.5)
    cs = build_set(pb, default_box())
    lay = connected_components(cs, pb.n)
    touching = cs.gap <= 1e-9
    comps = lay.component[touching].unique()
    assert comps.numel() == 1
    assert cs.n_pp >= 1 and cs.n_pw >= 1


def test_empty_contact_set_is_handled():
    pb = make_particles([[2.0, 2.0, 2.0]])
    cs = build_set(pb, default_box())
    assert cs.n_contacts == 0
    lay = connected_components(cs, pb.n)
    assert lay.n_components == 0
    A = assemble_normal_delassus(cs, pb.mass, pb.inertia, lay)
    assert A.numel() == 0
