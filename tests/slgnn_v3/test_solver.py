"""Solver normal: caso analítico, acoplamiento multicontacto y complementariedad."""

import math

import pytest
import torch

from slgnn_v3 import RouterProfile, V3State
from slgnn_v3.config import SolverConfig
from slgnn_v3.contact_operator import (
    assemble_normal_delassus,
    connected_components,
    pack_by_component,
    unpack_by_component,
)
from slgnn_v3.solver import analytic_two_body_impulse, solve_normal

from .conftest import DTYPE, build_set, default_box, make_particles, small_model


def _packed_problem(A_dense, b_vec, kappa_vec):
    """Empaqueta un problema denso de una sola componente."""
    from slgnn_v3.contact_operator import ComponentLayout

    c = A_dense.shape[0]
    lay = ComponentLayout(
        component=torch.zeros(c, dtype=torch.long),
        slot=torch.arange(c),
        n_components=1, max_size=c,
        valid=torch.ones(1, c, dtype=torch.bool),
        contact_of=torch.arange(c).unsqueeze(0),
        sizes=torch.tensor([c]),
    )
    return A_dense.unsqueeze(0), b_vec.unsqueeze(0), kappa_vec.unsqueeze(0), lay


def test_single_contact_closed_form():
    A = torch.tensor([[2.0]], dtype=DTYPE)
    b = torch.tensor([-3.0], dtype=DTYPE)
    k = torch.tensor([0.0], dtype=DTYPE)
    lam, diag = solve_normal(*_packed_problem(A, b, k), SolverConfig(kappa_floor=0.0))
    assert float(lam[0, 0]) == pytest.approx(1.5, rel=1e-12)
    assert float(diag.primal_residual.max()) <= 1e-12


def test_no_impulse_when_separating():
    """Si no hace falta impulso, `lambda = 0` y la velocidad admisible es positiva."""
    A = torch.tensor([[2.0]], dtype=DTYPE)
    b = torch.tensor([+3.0], dtype=DTYPE)
    lam, diag = solve_normal(*_packed_problem(A, b, torch.zeros(1, dtype=DTYPE)),
                             SolverConfig(kappa_floor=0.0))
    assert float(lam[0, 0]) == 0.0
    assert float(diag.min_normal_residual) > 0


def test_two_body_head_on_analytic():
    """§10.6: `Lambda_n = -(1+e) u_n^- / (1/m_i + 1/m_j)`, error rel <= 1e-8.

    Se resuelve con el solver completo del modelo, no con la fórmula: es el
    solver quien tiene que recuperar el resultado analítico.
    """
    m_i, m_j, e, u_n = 1.0, 2.5, 0.7, -3.0
    q = torch.tensor([[0.0, 0.0, 0.0], [0.98, 0.0, 0.0]], dtype=DTYPE)
    pb = make_particles(
        q, v=[[1.5, 0.0, 0.0], [-1.5, 0.0, 0.0]],
        radius=0.5, mass=torch.tensor([m_i, m_j], dtype=DTYPE),
    )
    cs = build_set(pb)
    assert float(cs.u_n) == pytest.approx(u_n, rel=1e-12)

    lay = connected_components(cs, pb.n)
    A = assemble_normal_delassus(cs, pb.mass, pb.inertia, lay)
    b = torch.tensor([(1.0 + e) * u_n], dtype=DTYPE)
    lam, diag = solve_normal(
        A, pack_by_component(b, lay), pack_by_component(torch.zeros(1, dtype=DTYPE), lay),
        lay, SolverConfig(kappa_floor=0.0),
    )
    expected = analytic_two_body_impulse(u_n, e, m_i, m_j)
    got = float(unpack_by_component(lam, lay)[0])
    assert abs(got - expected) / expected <= 1e-8


def test_two_body_post_impact_velocity_and_restitution():
    """Velocidad postimpacto, momento total y restitución efectiva."""
    m_i, m_j, e = 1.0, 2.5, 0.7
    v_i, v_j = 1.5, -1.5
    u_n = v_j - v_i
    lam = analytic_two_body_impulse(u_n, e, m_i, m_j)
    vi2, vj2 = v_i - lam / m_i, v_j + lam / m_j
    assert (m_i * vi2 + m_j * vj2) == pytest.approx(m_i * v_i + m_j * v_j, rel=1e-12)
    assert (vj2 - vi2) == pytest.approx(-e * u_n, rel=1e-12)


def test_coupled_solve_differs_from_independent():
    """§18.7: en multicontacto, el solve acoplado no coincide con resolver
    cada contacto por separado. Si coincidiera, `A` sería diagonal y el
    acoplamiento que exige la especificación no existiría."""
    # cadena de tres esferas en contacto simultáneo sobre el eje x
    q = torch.tensor([[0.0, 0.0, 0.0], [0.98, 0.0, 0.0], [1.96, 0.0, 0.0]], dtype=DTYPE)
    pb = make_particles(q, v=[[2.0, 0, 0], [0.0, 0, 0], [0.0, 0, 0]], radius=0.5)
    cs = build_set(pb)
    pp = cs.subset(~cs.is_wall)
    assert pp.n_contacts == 2
    lay = connected_components(pp, pb.n)
    assert lay.n_components == 1 and lay.max_size == 2

    A = assemble_normal_delassus(pp, pb.mass, pb.inertia, lay)
    assert float(A[0, 0, 1].abs()) > 1e-6      # acoplamiento real
    b = pack_by_component(pp.u_n, lay)
    k = pack_by_component(torch.zeros(2, dtype=DTYPE), lay)
    lam_coupled = unpack_by_component(
        solve_normal(A, b, k, lay, SolverConfig(kappa_floor=0.0, max_iters=400))[0], lay
    )
    diag_only = torch.diag_embed(torch.diagonal(A, dim1=1, dim2=2))
    lam_indep = unpack_by_component(
        solve_normal(diag_only, b, k, lay, SolverConfig(kappa_floor=0.0))[0], lay
    )
    assert float((lam_coupled - lam_indep).abs().max()) > 1e-3


def test_coupled_solve_satisfies_complementarity():
    q = torch.tensor([[0.0, 0.0, 0.0], [0.98, 0.0, 0.0], [1.96, 0.0, 0.0]], dtype=DTYPE)
    pb = make_particles(q, v=[[2.0, 0, 0], [0.0, 0, 0], [-1.0, 0, 0]], radius=0.5)
    cs = build_set(pb)
    pp = cs.subset(~cs.is_wall)
    lay = connected_components(pp, pb.n)
    A = assemble_normal_delassus(pp, pb.mass, pb.inertia, lay)
    lam, diag = solve_normal(
        A, pack_by_component(pp.u_n, lay),
        pack_by_component(torch.zeros(pp.n_contacts, dtype=DTYPE), lay),
        lay, SolverConfig(kappa_floor=0.0, eval_max_iters=4000, tol=1e-12),
        eval_mode=True,
    )
    assert float(diag.min_lambda) >= -1e-14
    assert float(diag.min_normal_residual) >= -1e-9
    assert float(diag.complementarity.max()) <= 1e-7
    assert float(diag.primal_residual.max()) <= 1e-7


def test_kappa_regularizes():
    """`kappa > 0` reduce el impulso: es compliance, no un truco numérico."""
    A = torch.tensor([[1.0]], dtype=DTYPE)
    b = torch.tensor([-1.0], dtype=DTYPE)
    lam0 = solve_normal(*_packed_problem(A, b, torch.tensor([0.0], dtype=DTYPE)),
                        SolverConfig(kappa_floor=0.0))[0]
    lam1 = solve_normal(*_packed_problem(A, b, torch.tensor([1.0], dtype=DTYPE)),
                        SolverConfig(kappa_floor=0.0))[0]
    assert float(lam1) < float(lam0)
    assert float(lam1) == pytest.approx(0.5, rel=1e-12)


def test_diagnostics_are_reported():
    A = torch.tensor([[2.0, 0.3], [0.3, 1.5]], dtype=DTYPE)
    b = torch.tensor([-1.0, -0.5], dtype=DTYPE)
    _, diag = solve_normal(*_packed_problem(A, b, torch.zeros(2, dtype=DTYPE)),
                           SolverConfig(kappa_floor=0.0))
    s = diag.summary()
    for key in ("solver_primal_residual_max", "solver_min_lambda",
                "solver_complementarity_max", "solver_iterations",
                "solver_conditioning_max", "solver_component_size_max"):
        assert key in s
    assert s["solver_conditioning_max"] > 1.0


def test_solution_is_differentiable():
    """El gradiente atraviesa el solver desenrollado."""
    A = torch.tensor([[2.0, 0.4], [0.4, 1.2]], dtype=DTYPE)
    kappa = torch.tensor([0.1, 0.2], dtype=DTYPE, requires_grad=True)
    b = torch.tensor([-1.0, -0.8], dtype=DTYPE, requires_grad=True)
    Ap, bp, kp, lay = _packed_problem(A, b, kappa)
    lam, _ = solve_normal(Ap, bp, kp, lay, SolverConfig(kappa_floor=0.0, max_iters=200))
    lam.sum().backward()
    assert torch.isfinite(kappa.grad).all() and float(kappa.grad.abs().sum()) > 0
    assert torch.isfinite(b.grad).all() and float(b.grad.abs().sum()) > 0


def test_lipschitz_is_an_upper_bound_on_lambda_max():
    """Regresión: con `H = [[2,-1],[-1,2]]` el vector de unos es autovector del
    autovalor MENOR (1), no del mayor (3). Una iteración de potencia iniciada
    ahí devuelve L = 1, el paso 1/L queda 3x por encima del límite estable y
    FISTA diverge a NaN. La cota de Gershgorin no tiene ese modo de fallo."""
    from slgnn_v3.solver import _lipschitz_bound

    A = torch.tensor([[2.0, -1.0], [-1.0, 2.0]], dtype=DTYPE)
    Ap, bp, kp, lay = _packed_problem(A, torch.tensor([-1.0, -1.0], dtype=DTYPE),
                                      torch.zeros(2, dtype=DTYPE))
    L = _lipschitz_bound(Ap, lay.valid)
    assert float(L) >= float(torch.linalg.eigvalsh(A).max()) - 1e-12
    lam, diag = solve_normal(Ap, bp, kp, lay, SolverConfig(kappa_floor=0.0,
                                                          max_iters=500))
    assert torch.isfinite(lam).all()
    assert float(diag.primal_residual.max()) <= 1e-9


def test_solver_runs_in_float64_by_default_in_tests():
    A = torch.tensor([[1.0]], dtype=DTYPE)
    lam, _ = solve_normal(*_packed_problem(A, torch.tensor([-1.0], dtype=DTYPE),
                                           torch.zeros(1, dtype=DTYPE)),
                          SolverConfig())
    assert lam.dtype == torch.float64
