"""SLGNN-v2: modelo completo con rotación (variante v2-R de la formulación).

Forward pass según §32: estado -> SDF y grafo -> processors escalares (V, R, H)
-> energías -> Euler-Lagrange vía autograd -> Rayleigh vía autograd -> rama
histórica explícita -> aceleraciones.

Garantías por construcción (§37): invariancia de permutación, equivariancia
SO(3) (las MLPs solo producen escalares invariantes; los vectores emergen de
gradientes o de bases geométricas), acción-reacción, conservación de momento
lineal y angular interno, disipación de Rayleigh con el signo correcto, y
suavidad C² de entrada/salida del grafo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SLGNNConfig
from .cutoff import neg_part
from .graph import (WallSet, contact_velocities, neighbor_pairs,
                    pair_geometry, wall_geometry)
from .nets import (MaterialEncoder, ScalarProcessor, bias_init_last, mlp,
                   symmetric_pair, zero_init_last)
from .state import Particles, SLGNNOutput


class SLGNN(nn.Module):
    def __init__(self, cfg: SLGNNConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or SLGNNConfig()

        node_in = cfg.h_mat + 4          # [eta, phi, g_W, delta_W, chi_W]
        edge_in = 3 * cfg.h_mat + 4      # [sym(eta), d, g, delta, chi]
        node_in_h = node_in + 2          # + [w_n, |w_tau|]
        edge_in_h = edge_in + 2          # + [u_n, |u_tau|]
        h = cfg.hidden

        self.material_encoder = MaterialEncoder(cfg.n_types, cfg.n_props, cfg.h_mat)

        # tres canales neuronales (§14)
        self.proc_V = ScalarProcessor(node_in, edge_in, h, cfg.layers)
        self.proc_R = ScalarProcessor(node_in, edge_in, h, cfg.layers)
        if cfg.use_history:
            self.proc_H = ScalarProcessor(node_in_h, edge_in_h, h, cfg.layers)

        # cabezas conservativas (§18-§19): escalar por arista / por partícula
        self.head_pp_V = mlp([h + 4, h, 1])
        self.head_pw_V = mlp([h + 4, h, 1])
        # cabezas de Rayleigh (§22): coeficientes (c_n, c_tau) via softplus
        self.head_pp_R = bias_init_last(mlp([h + 4, h, 2]), cfg.rayleigh_bias_init)
        self.head_pw_R = bias_init_last(mlp([h + 4, h, 2]), cfg.rayleigh_bias_init)
        # cabezas históricas (§26): amplitudes escalares (a_n, a_tau), inicio en 0
        if cfg.use_history:
            self.head_pp_H = zero_init_last(mlp([h + 6, h, 2]))
            self.head_pw_H = zero_init_last(mlp([h + 6, h, 2]))

    def forward(self, q, v, omega, particles: Particles, wall=None,
                t: float = 0.0, g_vec=None) -> SLGNNOutput:
        cfg = self.cfg
        n = q.shape[0]
        dtype, device = q.dtype, q.device
        if g_vec is None:
            g_vec = torch.zeros(3, dtype=dtype, device=device)
        else:
            g_vec = torch.as_tensor(g_vec, dtype=dtype, device=device)

        outer_grad = torch.is_grad_enabled()
        with torch.enable_grad():
            if not q.requires_grad:
                q = q.detach().requires_grad_(True)
            if not v.requires_grad:
                v = v.detach().requires_grad_(True)
            if not omega.requires_grad:
                omega = omega.detach().requires_grad_(True)

            m = particles.m.to(dtype=dtype, device=device)
            radii = particles.radii.to(dtype=dtype, device=device)
            inertia = particles.inertia.to(dtype=dtype, device=device)

            # --- pasos 3-7 (§32): geometría de pared, grafo y contactos ---
            idx = neighbor_pairs(q, cfg.r_list)
            ps = pair_geometry(q, radii, idx, cfg)
            u_c, u_n, u_tau = contact_velocities(v, omega, ps)
            u_tau_norm = u_tau.norm(dim=-1)

            if wall is not None:
                ws = wall_geometry(q, v, omega, radii, wall, t, cfg)
            else:
                # sin pared: sentinela "lejos" con chi_W = 0 que anula el canal
                zeros_n = torch.zeros(n, dtype=dtype, device=device)
                zeros_n3 = torch.zeros(n, 3, dtype=dtype, device=device)
                nu_far = torch.zeros(n, 3, dtype=dtype, device=device)
                nu_far[:, 2] = 1.0
                ws = WallSet(
                    phi=radii + cfg.g_off, nu=nu_far, x_w=zeros_n3, v_w=zeros_n3,
                    g=torch.full_like(zeros_n, cfg.g_off), delta=zeros_n,
                    chi=zeros_n, w_rel=zeros_n3, w_n=zeros_n, w_tau=zeros_n3,
                )
            has_wall = wall is not None

            # --- features invariantes (§11, §13) ---
            eta = self.material_encoder(particles, dtype)
            node_geo = torch.stack([ws.phi, ws.g, ws.delta, ws.chi], dim=-1)
            edge_geo = torch.stack([ps.d, ps.g, ps.delta, ps.chi], dim=-1)
            s_node = torch.cat([eta, node_geo], dim=-1)
            s_edge = torch.cat(
                [symmetric_pair(eta[ps.idx_i], eta[ps.idx_j]), edge_geo], dim=-1
            )

            # --- paso 8 (§32): processors por canal ---
            zV_n, zV_e = self.proc_V(s_node, s_edge, ps.idx_i, ps.idx_j, ps.chi)
            zR_n, zR_e = self.proc_R(s_node, s_edge, ps.idx_i, ps.idx_j, ps.chi)

            # --- paso 9: energías (§17-§20) ---
            V_g = -(m * (q @ g_vec)).sum()
            U_pp = self.head_pp_V(torch.cat([zV_e, edge_geo], dim=-1)).squeeze(-1)
            V_pp = (ps.chi * U_pp).sum()
            if has_wall:
                U_pw = self.head_pw_V(torch.cat([zV_n, node_geo], dim=-1)).squeeze(-1)
                V_pw = (ws.chi * U_pw).sum()
            else:
                V_pw = torch.zeros((), dtype=dtype, device=device)
            # el término nulo garantiza conexión al grafo aunque E=0 y g=0
            V_total = V_g + V_pp + V_pw + q.sum() * 0.0

            # --- paso 11a: fuerza conservativa por Euler-Lagrange (§28) ---
            (grad_q,) = torch.autograd.grad(V_total, q, create_graph=outer_grad)
            f_cons = -grad_q

            # --- paso 10: función de Rayleigh (§21-§25) ---
            c_pp = F.softplus(self.head_pp_R(torch.cat([zR_e, edge_geo], dim=-1)))
            r_pp = 0.5 * (
                ps.chi
                * (
                    c_pp[:, 0] * neg_part(u_n, cfg.beta) ** 2
                    + c_pp[:, 1] * u_tau.pow(2).sum(dim=-1)
                )
            ).sum()
            if has_wall:
                c_pw = F.softplus(self.head_pw_R(torch.cat([zR_n, node_geo], dim=-1)))
                r_pw = 0.5 * (
                    ws.chi
                    * (
                        c_pw[:, 0] * neg_part(ws.w_n, cfg.beta) ** 2
                        + c_pw[:, 1] * ws.w_tau.pow(2).sum(dim=-1)
                    )
                ).sum()
            else:
                r_pw = torch.zeros((), dtype=dtype, device=device)
            r_total = r_pp + r_pw + (v.sum() + omega.sum()) * 0.0

            # --- paso 11b: fuerzas y torques disipativos, signo obligatorio (§25) ---
            grad_v, grad_w = torch.autograd.grad(
                r_total, (v, omega), create_graph=outer_grad
            )
            f_ray = -grad_v
            tau_ray = -grad_w

            # --- paso 12: rama histórica estructurada (§26) ---
            f_hist = torch.zeros(n, 3, dtype=dtype, device=device)
            tau_hist = torch.zeros(n, 3, dtype=dtype, device=device)
            p_hist_pp = torch.zeros(ps.d.shape[0], dtype=dtype, device=device)
            p_hist_pw = torch.zeros(n, dtype=dtype, device=device)
            if cfg.use_history:
                s_node_h = torch.cat(
                    [s_node, torch.stack([ws.w_n, ws.w_tau.norm(dim=-1)], dim=-1)],
                    dim=-1,
                )
                s_edge_h = torch.cat(
                    [s_edge, torch.stack([u_n, u_tau_norm], dim=-1)], dim=-1
                )
                zH_n, zH_e = self.proc_H(s_node_h, s_edge_h, ps.idx_i, ps.idx_j, ps.chi)

                a_pp = self.head_pp_H(
                    torch.cat(
                        [zH_e, edge_geo, u_n.unsqueeze(-1), u_tau_norm.unsqueeze(-1)],
                        dim=-1,
                    )
                )
                u_tau_hat = u_tau / (u_tau_norm.unsqueeze(-1) + cfg.eps)
                # fuerza sobre i desde j; antisimetría exacta por construcción (§26.3)
                f_edge = ps.chi.unsqueeze(-1) * (
                    a_pp[:, 0:1] * ps.e + a_pp[:, 1:2] * u_tau_hat
                )
                f_hist = f_hist.index_add(0, ps.idx_i, f_edge)
                f_hist = f_hist.index_add(0, ps.idx_j, -f_edge)
                arm_i = ps.l_i.unsqueeze(-1) * ps.e
                arm_j = -ps.l_j.unsqueeze(-1) * ps.e
                tau_hist = tau_hist.index_add(
                    0, ps.idx_i, torch.linalg.cross(arm_i, f_edge, dim=-1)
                )
                tau_hist = tau_hist.index_add(
                    0, ps.idx_j, torch.linalg.cross(arm_j, -f_edge, dim=-1)
                )
                # potencia neta del residual del par: P = -u_c . f (§34.6)
                p_hist_pp = -(u_c * f_edge).sum(dim=-1)

                if has_wall:
                    a_pw = self.head_pw_H(
                        torch.cat(
                            [
                                zH_n, node_geo,
                                ws.w_n.unsqueeze(-1),
                                ws.w_tau.norm(dim=-1).unsqueeze(-1),
                            ],
                            dim=-1,
                        )
                    )
                    w_tau_hat = ws.w_tau / (
                        ws.w_tau.norm(dim=-1, keepdim=True) + cfg.eps
                    )
                    f_w = ws.chi.unsqueeze(-1) * (
                        a_pw[:, 0:1] * ws.nu + a_pw[:, 1:2] * w_tau_hat
                    )
                    f_hist = f_hist + f_w
                    tau_hist = tau_hist + torch.linalg.cross(
                        -radii.unsqueeze(-1) * ws.nu, f_w, dim=-1
                    )
                    p_hist_pw = (ws.w_rel * f_w).sum(dim=-1)

            # --- paso 13: aceleraciones (§28) ---
            a = (f_cons + f_ray + f_hist) / m.unsqueeze(-1)
            alpha = (tau_ray + tau_hist) / inertia.unsqueeze(-1)

            diagnostics = {
                "V_g": V_g, "V_pp": V_pp, "V_pW": V_pw,
                "R_pp": r_pp, "R_pW": r_pw,
                "f_cons": f_cons, "f_R": f_ray, "tau_R": tau_ray,
                "f_H": f_hist, "tau_H": tau_hist,
                "P_hist_pp": p_hist_pp, "P_hist_pW": p_hist_pw,
                "phi": ws.phi, "chi_W": ws.chi,
                "n_edges": ps.d.shape[0],
            }

        if not outer_grad:
            a = a.detach()
            alpha = alpha.detach()
            diagnostics = {
                k: (val.detach() if torch.is_tensor(val) else val)
                for k, val in diagnostics.items()
            }
        return SLGNNOutput(a=a, alpha=alpha, diagnostics=diagnostics)
