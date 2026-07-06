"""Grafo dinámico y geometría de contactos de SLGNN-v2 (§6-§9).

La lista de vecinos es discreta (sin gradiente); toda la geometría de pares
(r_ij, d_ij, e_ij, gaps, velocidades de contacto) se recalcula de forma
diferenciable respecto de posiciones y velocidades.
"""

from dataclasses import dataclass

import torch

from .config import SLGNNConfig
from .cutoff import quintic_window, softplus_compression


def neighbor_pairs(q: torch.Tensor, r_list: float) -> torch.Tensor:
    """Pares candidatos {i, j} con i < j y d_ij < r_list (§8). [E, 2] long."""
    with torch.no_grad():
        d = torch.cdist(q, q)
        mask = (d < r_list).triu(diagonal=1)
        return mask.nonzero()


@dataclass
class PairSet:
    """Geometría diferenciable de las aristas no dirigidas (§8-§9)."""

    idx_i: torch.Tensor  # [E]
    idx_j: torch.Tensor  # [E]
    d: torch.Tensor      # [E] distancia entre centros
    e: torch.Tensor      # [E,3] unitario de i hacia j
    g: torch.Tensor      # [E] separación superficial d - (R_i + R_j)
    delta: torch.Tensor  # [E] compresión suave sp_beta(-g)
    chi: torch.Tensor    # [E] ventana C² chi_pp(d)
    l_i: torch.Tensor    # [E] brazo de i al punto común de contacto
    l_j: torch.Tensor    # [E] brazo de j


def pair_geometry(q: torch.Tensor, radii: torch.Tensor, idx: torch.Tensor,
                  cfg: SLGNNConfig) -> PairSet:
    i, j = idx[:, 0], idx[:, 1]
    r = q[j] - q[i]
    d = r.norm(dim=-1)
    e = r / (d.unsqueeze(-1) + cfg.eps)
    g = d - (radii[i] + radii[j])
    delta = softplus_compression(g, cfg.beta)
    chi = quintic_window(d, cfg.r_on, cfg.r_off)
    # punto común de contacto (§9): l_i + l_j = d, reparto proporcional a radios
    frac_i = radii[i] / (radii[i] + radii[j])
    l_i = d * frac_i
    l_j = d - l_i
    return PairSet(i, j, d, e, g, delta, chi, l_i, l_j)


def contact_velocities(v: torch.Tensor, omega: torch.Tensor, ps: PairSet):
    """Velocidad relativa en el punto común de contacto (§9).

    u_ij^c = (v_j + w_j x r_j^c) - (v_i + w_i x r_i^c), con brazos
    r_i^c = l_i e_ij, r_j^c = -l_j e_ij. Devuelve (u_c, u_n, u_tau).
    """
    i, j = ps.idx_i, ps.idx_j
    arm_i = ps.l_i.unsqueeze(-1) * ps.e
    arm_j = -ps.l_j.unsqueeze(-1) * ps.e
    v_ci = v[i] + torch.linalg.cross(omega[i], arm_i, dim=-1)
    v_cj = v[j] + torch.linalg.cross(omega[j], arm_j, dim=-1)
    u_c = v_cj - v_ci
    u_n = (u_c * ps.e).sum(dim=-1)
    u_tau = u_c - u_n.unsqueeze(-1) * ps.e
    return u_c, u_n, u_tau


@dataclass
class WallSet:
    """Geometría y cinemática partícula-pared por partícula (§4-§7)."""

    phi: torch.Tensor     # [N] distancia con signo (en grafo respecto de q)
    nu: torch.Tensor      # [N,3] normal entrante
    x_w: torch.Tensor     # [N,3] punto de pared más próximo
    v_w: torch.Tensor     # [N,3] velocidad local de pared
    g: torch.Tensor       # [N] gap superficial phi - R
    delta: torch.Tensor   # [N] compresión suave
    chi: torch.Tensor     # [N] ventana chi_pW(g)
    w_rel: torch.Tensor   # [N,3] velocidad relativa de contacto w_iW (§7)
    w_n: torch.Tensor     # [N]
    w_tau: torch.Tensor   # [N,3]


def wall_geometry(q, v, omega, radii, wall, t: float, cfg: SLGNNConfig) -> WallSet:
    phi, nu, x_w, v_w = wall.query(q, t, cfg.eps)
    g = phi - radii
    delta = softplus_compression(g, cfg.beta)
    chi = quintic_window(g, cfg.g_on, cfg.g_off)
    # w_iW = v_i + w_i x (-R_i nu_i) - v_W (§7)
    arm = -radii.unsqueeze(-1) * nu
    w_rel = v + torch.linalg.cross(omega, arm, dim=-1) - v_w
    w_n = (nu * w_rel).sum(dim=-1)
    w_tau = w_rel - w_n.unsqueeze(-1) * nu
    return WallSet(phi, nu, x_w, v_w, g, delta, chi, w_rel, w_n, w_tau)
