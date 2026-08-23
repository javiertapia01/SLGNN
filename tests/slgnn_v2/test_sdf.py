import math

import torch

from slgnn.sdf import BoxSDF, RotatingCylinderSDF, dynamical_cylinder_omega


def test_box_phi_values():
    box = BoxSDF([0.0, 0.0, 0.0], [0.03, 0.03, 0.03])
    x = torch.tensor(
        [[0.015, 0.015, 0.015], [0.001, 0.015, 0.015], [0.015, 0.015, 0.029]],
        dtype=torch.float64,
    )
    phi = box.phi(x, 0.0)
    expected = torch.tensor([0.015, 0.001, 0.001], dtype=torch.float64)
    assert torch.allclose(phi, expected)


def test_box_gradient_unit_inward():
    box = BoxSDF([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    x = torch.tensor([[0.1, 0.5, 0.5], [0.5, 0.5, 0.93]], dtype=torch.float64)
    g = box.grad_phi(x, 0.0)
    assert torch.allclose(g.norm(dim=-1), torch.ones(2, dtype=torch.float64))
    # cara más cercana x=0: normal entrante +x ; cara z=1: normal entrante -z
    assert torch.allclose(g[0], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
    assert torch.allclose(g[1], torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64))


def test_box_pose_invariance():
    # phi_posed(Q x + c) == phi_0(x)  (§4.2)
    torch.manual_seed(0)
    aa = torch.randn(3, dtype=torch.float64)
    Q = torch.linalg.matrix_exp(
        torch.tensor(
            [[0, -aa[2], aa[1]], [aa[2], 0, -aa[0]], [-aa[1], aa[0], 0]],
            dtype=torch.float64,
        )
    )
    c = torch.randn(3, dtype=torch.float64)
    box0 = BoxSDF([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
    box1 = BoxSDF([0.0, 0.0, 0.0], [1.0, 2.0, 3.0], pose=(Q, c))
    x = torch.rand(20, 3, dtype=torch.float64) * torch.tensor([1.0, 2.0, 3.0])
    assert torch.allclose(box1.phi(x @ Q.T + c, 0.0), box0.phi(x, 0.0), atol=1e-12)


def test_cylinder_phi_and_features():
    cyl = RotatingCylinderSDF((0.0, 0.002), 0.05, 0.0, 0.1)
    x = torch.tensor(
        [
            [0.0, 0.002, 0.05],   # centro: lateral a 0.05, tapas a 0.05
            [0.04, 0.002, 0.05],  # cerca de pared lateral: 0.01
            [0.0, 0.002, 0.001],  # cerca de tapa inferior: 0.001
        ],
        dtype=torch.float64,
    )
    phi = cyl.phi(x, 0.0)
    assert torch.allclose(
        phi, torch.tensor([0.05, 0.01, 0.001], dtype=torch.float64), atol=1e-12
    )


def test_cylinder_wall_velocity_lateral_vs_caps():
    cyl = RotatingCylinderSDF((0.0, 0.002), 0.05, 0.0, 0.1)
    t = 0.25  # omega = 2*pi*1 rad/s
    omega = dynamical_cylinder_omega(t)
    assert abs(omega - 2 * math.pi) < 1e-12
    x = torch.tensor(
        [
            [0.05, 0.002, 0.05],  # sobre pared lateral
            [0.01, 0.002, 0.0],   # sobre tapa inferior (fija)
        ],
        dtype=torch.float64,
    )
    v = cyl.wall_velocity(x, t)
    # lateral: v = omega ẑ x r = omega*(-dy, dx, 0) con r=(0.05, 0, 0)
    assert torch.allclose(
        v[0], torch.tensor([0.0, omega * 0.05, 0.0], dtype=torch.float64), atol=1e-12
    )
    assert torch.allclose(v[1], torch.zeros(3, dtype=torch.float64))


def test_omega_profile_triangle():
    assert dynamical_cylinder_omega(0.0) == 0.0
    assert abs(dynamical_cylinder_omega(0.5) - 4 * math.pi) < 1e-12   # pico 2 (norm.)
    assert abs(dynamical_cylinder_omega(0.75) - 2 * math.pi) < 1e-12  # bajando
    assert dynamical_cylinder_omega(1.0) == 0.0
    assert dynamical_cylinder_omega(1.5) == 0.0  # nunca negativa
    assert dynamical_cylinder_omega(2.0) == 0.0
