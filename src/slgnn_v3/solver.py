"""Solver normal acoplado con complementariedad (§10).

Resuelve, por **componente conexa** del grafo de contactos,

    lambda = argmin_{lambda >= 0}  1/2 lambda^T H lambda + b^T lambda,
    H = A_n + diag(kappa),   A_n = J_n M^-1 J_n^T,
    b = u_n^* + iota e min(u_n^-, 0) + beta min(g, 0)/dt,

cuyas condiciones KKT son exactamente

    0 <= lambda  _|_  u_n^+ + b_n + kappa lambda >= 0.

Dos backends:

1. **Cerrado exacto** para componentes de un solo contacto:
   `lambda = max(0, -b / H)`.
2. **FISTA proyectado desenrollado** para componentes acopladas, con `H`
   materializada densa solo dentro de la componente. Resolver contacto por
   contacto cuando comparten partículas está prohibido (§22.6): ignorar el
   acoplamiento viola simultáneamente no penetración, momento y fricción.

El tamaño de paso `1/L` se estima por iteración de potencia bajo `no_grad`:
es un parámetro del optimizador, no del modelo, y dejarlo fuera del grafo es
intencional y está documentado aquí.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .config import SolverConfig
from .contact_operator import ComponentLayout


@dataclass
class SolveDiagnostics:
    """Diagnósticos por componente. Sin esto un número fijo de iteraciones no
    constituye una solución verificada (§10.4)."""

    primal_residual: torch.Tensor      # [K] residuo natural ||x - P(x - r)||_inf
    min_lambda: torch.Tensor           # [K]
    min_normal_residual: torch.Tensor  # [K]
    complementarity: torch.Tensor      # [K] max |lambda * r|
    iterations: int
    converged: torch.Tensor            # [K] bool
    conditioning: torch.Tensor         # [K]
    component_size: torch.Tensor       # [K]
    lipschitz: torch.Tensor            # [K]

    def summary(self) -> dict[str, float]:
        f = lambda t: float(t.max()) if t.numel() else 0.0
        g = lambda t: float(t.min()) if t.numel() else 0.0
        return {
            "solver_primal_residual_max": f(self.primal_residual),
            "solver_min_lambda": g(self.min_lambda),
            "solver_min_normal_residual": g(self.min_normal_residual),
            "solver_complementarity_max": f(self.complementarity),
            "solver_iterations": float(self.iterations),
            "solver_converged_fraction": (
                float(self.converged.to(torch.float64).mean()) if self.converged.numel() else 1.0
            ),
            "solver_conditioning_max": f(self.conditioning),
            "solver_component_size_max": f(self.component_size.to(torch.float64)),
        }


def _lipschitz_bound(H: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Cota de Gershgorin `max_a sum_b |H_ab|`, `[K]`.

    Para `H` simétrica esta cota es **siempre** `>= lambda_max`, así que
    `1/L` es un paso de FISTA demostrablemente estable. Se prefiere a una
    estimación iterativa porque una subestimación de `lambda_max` no degrada
    la convergencia: la destruye. Con `H = [[2,-1],[-1,2]]` —una cadena de
    tres esferas— la cota vale 3, exactamente `lambda_max`.
    """
    with torch.no_grad():
        return (H.abs().sum(dim=-1) * valid).amax(dim=-1)


def _power_iteration(H: torch.Tensor, valid: torch.Tensor, iters: int,
                     seed: int = 12345) -> torch.Tensor:
    """Mayor autovalor por componente, `[K]`. Solo para diagnóstico.

    El vector inicial es pseudoaleatorio con semilla fija, no el vector de
    unos: en una matriz simétrica de cadena, `(1,...,1)` es exactamente un
    autovector del autovalor **menor**, y la iteración se queda ahí.
    """
    with torch.no_grad():
        g = torch.Generator(device="cpu").manual_seed(seed)
        y = torch.randn(H.shape[0], H.shape[1], generator=g, dtype=torch.float64)
        y = y.to(dtype=H.dtype, device=H.device) * valid
        y = y / y.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        lam = torch.zeros(H.shape[0], dtype=H.dtype, device=H.device)
        for _ in range(iters):
            z = torch.einsum("kab,kb->ka", H, y) * valid
            lam = z.norm(dim=-1)
            y = z / lam.clamp_min(1e-30).unsqueeze(-1)
        return lam


def solve_normal(
    A: torch.Tensor,
    b: torch.Tensor,
    kappa: torch.Tensor,
    layout: ComponentLayout,
    cfg: SolverConfig,
    eval_mode: bool = False,
) -> tuple[torch.Tensor, SolveDiagnostics]:
    """Resuelve el LCP normal. `A [K,c,c]`, `b`, `kappa` `[K,c]` empaquetados.

    Devuelve `lambda [K,c]` (ceros en el padding) y los diagnósticos.
    """
    K, cmax = A.shape[0], A.shape[1]
    dev, dt_ = A.device, A.dtype
    valid = layout.valid
    if K == 0 or cmax == 0:
        z0 = torch.zeros(0, dtype=dt_, device=dev)
        zb = torch.zeros(0, dtype=torch.bool, device=dev)
        return torch.zeros(K, cmax, dtype=dt_, device=dev), SolveDiagnostics(
            z0, z0, z0, z0, 0, zb, z0, torch.zeros(0, dtype=torch.long, device=dev), z0
        )

    vf = valid.to(dt_)
    # Padding con filas y columnas exactamente nulas: sus autovalores son 0 y
    # no contaminan la iteración de potencia mientras `y` se enmascare.
    diag = (kappa + cfg.kappa_floor) * vf
    H = A * (vf.unsqueeze(1) * vf.unsqueeze(2)) + torch.diag_embed(diag)
    b = b * vf

    L = _lipschitz_bound(H, valid).clamp_min(1e-30)
    step = (1.0 / L).unsqueeze(-1)

    max_iters = cfg.eval_max_iters if eval_mode else cfg.max_iters
    x = torch.zeros(K, cmax, dtype=dt_, device=dev)
    y = x
    t = torch.ones(K, 1, dtype=dt_, device=dev)
    used = 0
    for it in range(max_iters):
        grad = torch.einsum("kab,kb->ka", H, y) + b
        x_new = (y - step * grad).clamp_min(0.0) * vf
        t_new = 0.5 * (1.0 + torch.sqrt(1.0 + 4.0 * t * t))
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, t = x_new, t_new
        used = it + 1
        if eval_mode and (used % 10 == 0):
            with torch.no_grad():
                r = torch.einsum("kab,kb->ka", H, x) + b
                nat = ((x - (x - r).clamp_min(0.0)).abs() * vf).amax(dim=-1)
                if float(nat.max()) <= cfg.tol:
                    break

    # Componentes de un solo contacto: solución cerrada exacta, que además
    # sirve de test permanente contra el camino iterativo.
    single = layout.sizes == 1
    if bool(single.any()):
        h00 = H[:, 0, 0].clamp_min(1e-30)
        exact = (-b[:, 0] / h00).clamp_min(0.0)
        x = torch.where(
            (single.unsqueeze(-1)) & (torch.arange(cmax, device=dev).unsqueeze(0) == 0),
            exact.unsqueeze(-1), x,
        )
        if cmax > 1:
            x = torch.where(single.unsqueeze(-1) &
                            (torch.arange(cmax, device=dev).unsqueeze(0) > 0),
                            torch.zeros_like(x), x)

    with torch.no_grad():
        r = torch.einsum("kab,kb->ka", H, x) + b
        big = torch.finfo(dt_).max
        nat = ((x - (x - r).clamp_min(0.0)).abs() * vf).amax(dim=-1)
        min_lam = torch.where(valid, x, torch.full_like(x, big)).amin(dim=-1)
        min_r = torch.where(valid, r, torch.full_like(r, big)).amin(dim=-1)
        comp = ((x * r).abs() * vf).amax(dim=-1)
        # Diagnóstico de condicionamiento: lambda_max por iteración de
        # potencia y lambda_min por la iteración desplazada `L I - H`.
        l_max = _power_iteration(H, valid, cfg.power_iters)
        shifted = torch.diag_embed(L.unsqueeze(-1).expand(-1, cmax) * vf) - H
        l_min = (L - _power_iteration(shifted, valid, cfg.power_iters)).clamp_min(1e-30)
        cond = l_max.clamp_min(1e-30) / l_min
        conv = nat <= cfg.tol

    return x, SolveDiagnostics(
        primal_residual=nat, min_lambda=min_lam, min_normal_residual=min_r,
        complementarity=comp, iterations=used, converged=conv,
        conditioning=cond, component_size=layout.sizes, lipschitz=L,
    )


def analytic_two_body_impulse(
    u_n: float, e: float, m_i: float, m_j: float
) -> float:
    """`Lambda_n = -(1+e) u_n^- / (1/m_i + 1/m_j)` para choque frontal.

    Referencia analítica de §10.6 / eq. (15.6). Vive en el código, no solo en
    el test, para que cualquiera pueda contrastar el solver desde una consola.
    """
    if u_n >= 0:
        return 0.0
    return -(1.0 + e) * u_n / (1.0 / m_i + 1.0 / m_j)
