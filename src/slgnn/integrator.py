"""Integración temporal de SLGNN-v2 (§31): Euler semiimplícito.

v_{k+1} = v_k + dt a_k ; q_{k+1} = q_k + dt v_{k+1} ; w_{k+1} = w_k + dt alpha_k.
Tras cada paso se reconstruyen geometría, grafo y features dentro del forward
del modelo (§32), de modo que el rollout solo encadena pasos.
"""

import torch

from .state import Particles


def semi_implicit_step(model, q, v, omega, particles: Particles, wall,
                       t: float, dt: float, g_vec=None):
    out = model(q, v, omega, particles, wall=wall, t=t, g_vec=g_vec)
    v_next = v + dt * out.a
    omega_next = omega + dt * out.alpha
    q_next = q + dt * v_next
    return q_next, v_next, omega_next, out


def rollout(model, q0, v0, omega0, particles: Particles, wall, dt: float,
            n_steps: int, t0: float = 0.0, g_vec=None, keep_graph: bool = False):
    """Rollout recursivo (§11 de la presentación formal).

    Con keep_graph=True se mantiene el grafo de autograd (para pérdida de
    rollout multi-step, Etapa III); con False se integra sin memoria de grafo
    (inferencia larga).
    """
    qs, vs, ws = [q0], [v0], [omega0]
    q, v, w = q0, v0, omega0
    ctx = torch.enable_grad() if keep_graph else torch.no_grad()
    with ctx:
        for k in range(n_steps):
            q, v, w, _ = semi_implicit_step(
                model, q, v, w, particles, wall, t0 + k * dt, dt, g_vec
            )
            qs.append(q)
            vs.append(v)
            ws.append(w)
    return torch.stack(qs), torch.stack(vs), torch.stack(ws)
