"""Superficies: convención de signo, multi-superficie, tiempo y pared móvil.

Cubre los problemas conocidos del legacy §18.2 (dos/tres caras activas en
arista y esquina) y §18.5 (el tiempo de pared cambia realmente entre pasos).
"""

import math

import pytest
import torch

from slgnn_v3.surfaces import (
    CylinderLateral,
    Plane,
    SurfaceSet,
    WallMotion,
    box_surfaces,
    dynamical_cylinder_omega_literal,
    rotating_cylinder_surfaces,
)

from .conftest import DTYPE, make_particles, random_rotation


def test_sign_convention_inside_positive():
    box = box_surfaces([0, 0, 0], [1, 1, 1])
    x = torch.tensor([[0.5, 0.5, 0.5], [-0.1, 0.5, 0.5]], dtype=DTYPE)
    phi = box.global_phi(x, 0.0)
    assert float(phi[0]) > 0          # interior
    assert float(phi[1]) < 0          # fuera


def test_gradient_points_inward():
    box = box_surfaces([0, 0, 0], [1, 1, 1])
    x = torch.tensor([[0.05, 0.5, 0.5]], dtype=DTYPE)
    n = box.surfaces[0].normal(x, 0.0)     # cara -x
    assert torch.allclose(n, torch.tensor([[1.0, 0.0, 0.0]], dtype=DTYPE))


def test_analytic_normal_matches_autograd():
    cyl = CylinderLateral(center_xy=(0.1, -0.2), radius=1.0)
    x = torch.tensor([[0.4, 0.3, 0.7], [-0.5, 0.1, 0.0]], dtype=DTYPE)
    analytic = cyl.normal(x, 0.0)
    numeric = super(CylinderLateral, cyl).normal(x, 0.0)  # ruta de autograd
    assert torch.allclose(analytic, numeric, atol=1e-10)


def test_phi_finite_differences():
    cyl = CylinderLateral(center_xy=(0.0, 0.0), radius=1.0)
    x = torch.tensor([[0.3, 0.4, 0.0]], dtype=DTYPE, requires_grad=True)
    phi = cyl.phi(x, 0.0).sum()
    (g,) = torch.autograd.grad(phi, x)
    h = 1e-6
    fd = torch.zeros(3, dtype=DTYPE)
    for k in range(3):
        d = torch.zeros(1, 3, dtype=DTYPE)
        d[0, k] = h
        fd[k] = (cyl.phi(x.detach() + d, 0.0) - cyl.phi(x.detach() - d, 0.0)) / (2 * h)
    assert torch.allclose(g[0], fd, atol=1e-6)


def test_gap_sign_separation_contact_penetration():
    box = box_surfaces([0, 0, 0], [4, 4, 4])
    R = 0.5
    for z, expected in ((2.0, "sep"), (0.5, "contact"), (0.3, "pen")):
        pb = make_particles([[2.0, 2.0, z]], radius=R)
        wq = box.query(pb.q, pb.radius, pb.batch_id, 0.0, band=2.0)
        m = wq.surface == 4     # cara -z
        gap = float(wq.gap[m])
        if expected == "sep":
            assert gap > 0
        elif expected == "contact":
            assert gap == pytest.approx(0.0, abs=1e-12)
        else:
            assert gap < 0


def test_edge_gives_two_surfaces():
    """Una partícula en una arista toca dos caras: `min` sobre caras las pierde."""
    box = box_surfaces([0, 0, 0], [4, 4, 4])
    pb = make_particles([[0.45, 0.45, 2.0]], radius=0.5)
    wq = box.query(pb.q, pb.radius, pb.batch_id, 0.0, band=0.0)
    assert wq.particle.numel() == 2
    assert set(wq.surface.tolist()) == {0, 2}     # -x, -y


def test_corner_gives_three_surfaces():
    box = box_surfaces([0, 0, 0], [4, 4, 4])
    pb = make_particles([[0.45, 0.45, 0.45]], radius=0.5)
    wq = box.query(pb.q, pb.radius, pb.batch_id, 0.0, band=0.0)
    assert wq.particle.numel() == 3
    assert set(wq.surface.tolist()) == {0, 2, 4}


def test_surface_ids_are_stable():
    box = box_surfaces([0, 0, 0], [1, 1, 1])
    assert box.names == ["-x", "+x", "-y", "+y", "-z", "+z"]
    assert [s.surface_id for s in box.surfaces] == [0, 1, 2, 3, 4, 5]


def test_wall_velocity_changes_with_time_at_fixed_geometry():
    """SDF axisimétrica constante en el tiempo, velocidad tangencial no.

    Es el punto ciego que la velocidad de pared explícita resuelve: la forma
    del cilindro no cambia al rotar, así que ninguna consulta de `phi` puede
    revelar el movimiento.
    """
    surf = rotating_cylinder_surfaces(
        (0.0, 0.0), 1.0, 0.0, 2.0, omega_fn=dynamical_cylinder_omega_literal
    )
    x = torch.tensor([[0.9, 0.0, 1.0]], dtype=DTYPE)
    phi0, phi1 = surf.global_phi(x, 0.0), surf.global_phi(x, 0.7)
    assert torch.allclose(phi0, phi1)                       # geometría inmóvil
    v0 = surf.surfaces[0].wall_velocity(x, 0.0)
    v1 = surf.surfaces[0].wall_velocity(x, 0.25)
    assert float(v0.norm()) == pytest.approx(0.0, abs=1e-14)
    assert float(v1.norm()) > 1e-3                          # cinemática sí cambia


def test_cylinder_omega_literal_reverses_direction():
    """El tambor invierte el giro entre t = 1.0 y t = 1.5 (DATA_NOTES.md §5)."""
    assert dynamical_cylinder_omega_literal(0.25)[2] > 0
    assert dynamical_cylinder_omega_literal(0.5)[2] == pytest.approx(2 * math.pi * 2.0)
    assert dynamical_cylinder_omega_literal(1.25)[2] < 0
    assert dynamical_cylinder_omega_literal(2.0)[2] == 0.0


def test_wall_velocity_formula():
    """`v_W = V_W + Omega_W x (x - c_W)`, evaluada en el punto dado."""
    motion = WallMotion(center=(1.0, 0.0, 0.0),
                        omega_fn=lambda t: (0.0, 0.0, 2.0),
                        velocity_fn=lambda t: (0.5, 0.0, 0.0))
    x = torch.tensor([[1.0, 1.0, 0.0]], dtype=DTYPE)
    v = motion.local_velocity(x, 0.0)
    # Omega x (x - c) = (0,0,2) x (0,1,0) = (-2, 0, 0); mas V_W = (0.5,0,0)
    assert torch.allclose(v, torch.tensor([[-1.5, 0.0, 0.0]], dtype=DTYPE))


def test_rigid_transform_of_a_plane():
    """Rotar y trasladar la pared transforma `phi` como `phi(Q^T(x-a))`."""
    Q, a = random_rotation(3), torch.tensor([0.3, -0.2, 0.5], dtype=DTYPE)
    n = torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE)
    p0 = Plane(tuple(n.tolist()), 0.0)
    n_rot = Q @ n
    p1 = Plane(tuple(n_rot.tolist()), float(n_rot @ a))
    x = torch.randn(6, 3, dtype=DTYPE)
    x_rot = x @ Q.T + a
    assert torch.allclose(p0.phi(x, 0.0), p1.phi(x_rot, 0.0), atol=1e-12)
