"""Encoder del baseline GNS: aristas dirigidas y features vectoriales.

Diferencia deliberada con v3: aquí **sí** entran componentes vectoriales
(desplazamiento relativo, velocidades) en el marco cartesiano, que es lo que
hace un GNS estándar. El modelo no es equivariante por construcción; si lo
necesita, tiene que aprenderlo de los datos. Esa es exactamente la variable
que la comparación quiere aislar.

El grafo físico no ordenado que comparte con v3 se expande aquí a **aristas
dirigidas** (cada par `{i,j}` produce `i->j` y `j->i`) para el paso de
mensajes; la transformación se documenta porque cambia el conteo de aristas
pero no el vecindario.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_ACT = {"silu": nn.SiLU, "gelu": nn.GELU, "tanh": nn.Tanh}


def mlp(sizes: list[int], activation: str = "silu", layer_norm: bool = True):
    act = _ACT[activation]
    layers: list[nn.Module] = []
    for k, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(a, b))
        if k < len(sizes) - 2:
            layers.append(act())
    if layer_norm:
        layers.append(nn.LayerNorm(sizes[-1]))
    return nn.Sequential(*layers)


def to_directed(edges: torch.Tensor) -> torch.Tensor:
    """`[E,2]` no ordenadas -> `[2, 2E]` dirigidas, ambos sentidos."""
    if edges.numel() == 0:
        return torch.zeros(2, 0, dtype=torch.long, device=edges.device)
    fwd = edges.t()
    return torch.cat([fwd, fwd.flip(0)], dim=1)


# nodo: v (3), omega (3), m, R, I  ->  9, mas material y pared
N_NODE_FEATURES = 9
# arista: dq (3), |dq|, gap, R_i + R_j  ->  6
N_EDGE_FEATURES = 6
# pared por nodo: normal * gap (3), gap, v_wall (3)  ->  7 por superficie agregada
N_WALL_FEATURES = 7


class GNSEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden
        self.material = nn.Embedding(cfg.n_material_types, cfg.material_dim)
        self.node_mlp = mlp(
            [N_NODE_FEATURES + cfg.material_dim + N_WALL_FEATURES, h, h],
            cfg.activation,
        )
        self.edge_mlp = mlp([N_EDGE_FEATURES, h, h], cfg.activation)

    def node_features(self, particles, wall_feat: torch.Tensor) -> torch.Tensor:
        z = self.material(particles.type_id)
        scal = torch.stack([particles.mass, particles.radius, particles.inertia], dim=-1)
        raw = torch.cat([particles.v, particles.omega, scal, z, wall_feat], dim=-1)
        return self.node_mlp(raw)

    def edge_features(self, particles, directed: torch.Tensor) -> torch.Tensor:
        if directed.shape[1] == 0:
            return torch.zeros(0, self.cfg.hidden, dtype=particles.q.dtype,
                               device=particles.q.device)
        src, dst = directed[0], directed[1]
        dq = particles.q[dst] - particles.q[src]
        d = dq.norm(dim=-1, keepdim=True)
        rsum = (particles.radius[src] + particles.radius[dst]).unsqueeze(-1)
        raw = torch.cat([dq, d, d - rsum, rsum], dim=-1)
        return self.edge_mlp(raw)


def wall_features(particles, wall_query, n_nodes: int) -> torch.Tensor:
    """Agrega las consultas de pared por nodo, `[N, N_WALL_FEATURES]`.

    Son las **mismas** consultas que recibe v3: misma SDF, mismas superficies,
    misma velocidad local. Aquí se suman por nodo porque el decoder de GNS no
    tiene un canal de contacto donde colgarlas.
    """
    dtype, dev = particles.q.dtype, particles.q.device
    out = torch.zeros(n_nodes, 7, dtype=dtype, device=dev)
    if wall_query is None or wall_query.particle.numel() == 0:
        return out
    idx = wall_query.particle
    feat = torch.cat([
        wall_query.normal * wall_query.gap.unsqueeze(-1),
        wall_query.gap.unsqueeze(-1),
        wall_query.wall_velocity,
    ], dim=-1)
    return out.index_add_(0, idx, feat)
