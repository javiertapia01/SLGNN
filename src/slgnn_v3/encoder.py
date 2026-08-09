"""Encoder geométrico compartido: embeddings materiales y escalares invariantes.

**El encoder no ve velocidades.** Esa es la condición que hace posible que la
cabeza `V` comparta esta etapa sin violar la prohibición de fuga entre canales
(§5.5 de la formulación): un potencial conservativo no puede depender de
`nu`, o deja de ser conservativo. Las features cinemáticas se inyectan más
abajo, solo en los processors de `Psi` e `I`.

Las aristas físicas son no ordenadas, así que las features de par se
construyen simétricamente: `(z_i + z_j, |z_i - z_j|)`. Los vectores no entran
nunca: se reconstruyen aguas abajo con normales, brazos y `J^T`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import EncoderConfig
from .contact_kinematics import ContactSet
from .state import ParticleBatch

_ACT = {"silu": nn.SiLU, "gelu": nn.GELU, "tanh": nn.Tanh, "softplus": nn.Softplus}


def mlp(sizes: list[int], activation: str = "silu", final_activation: bool = False):
    act = _ACT[activation]
    layers: list[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), act()]
    if not final_activation:
        layers.pop()
    return nn.Sequential(*layers)


# Número de escalares geométricos por contacto producidos por `edge_features`.
N_EDGE_GEO = 7
# Número de escalares cinemáticos por contacto producidos por `kinematic_features`.
N_EDGE_KIN = 5


def edge_features(contacts: ContactSet, particles: ParticleBatch) -> torch.Tensor:
    """Escalares geométricos por contacto, `[C, N_EDGE_GEO]`.

    Diferenciables respecto de `q`: `gap` y `delta` transmiten gradiente al
    camino energético. Ningún `detach` en esta ruta.
    """
    R = particles.radius
    Ri = R[contacts.i]
    Rj = torch.where(contacts.is_wall, torch.zeros_like(Ri), R[contacts.j.clamp(min=0)])
    return torch.stack([
        contacts.gap,
        contacts.delta,
        contacts.window,
        contacts.activation,
        Ri + Rj,
        (Ri - Rj).abs(),
        contacts.is_wall.to(contacts.gap.dtype),
    ], dim=-1)


def kinematic_features(contacts: ContactSet, particles: ParticleBatch,
                       age: torch.Tensor | None = None) -> torch.Tensor:
    """Escalares cinemáticos por contacto, `[C, N_EDGE_KIN]`.

    Solo llegan a `Psi` y a `I`. `Psi` los consume a través de una
    parametrización que preserva convexidad (ver `dissipation.py`).

    Los spines entran como `(|w_i| + |w_j|, ||w_i| - |w_j||)`, no como dos
    entradas separadas: una arista física es **no ordenada**, y `(|w_i|, |w_j|)`
    cambia al intercambiar `i` y `j`, lo que rompe la equivarianza a
    permutación del paso completo (§6.1). `u_n` y `|u_tau|` ya son invariantes
    al intercambio porque `u` y `n` se invierten a la vez.
    """
    wi = particles.omega[contacts.i].norm(dim=-1)
    wj = torch.where(
        contacts.is_wall, torch.zeros_like(wi),
        particles.omega[contacts.j.clamp(min=0)].norm(dim=-1),
    )
    if age is None:
        age = torch.zeros_like(contacts.u_n)
    return torch.stack([
        contacts.u_n,
        contacts.u_tau.norm(dim=-1),
        wi + wj,
        (wi - wj).abs(),
        age.to(contacts.u_n.dtype),
    ], dim=-1)


class GeometricEncoder(nn.Module):
    """Latentes de nodo y de contacto a partir de geometría y material."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden
        self.material = nn.Embedding(cfg.n_material_types, cfg.material_dim)
        # Embedding material de la pared: la pared es un "tipo" más, pero no un
        # nodo del grafo de partículas.
        self.wall_material = nn.Parameter(torch.zeros(cfg.material_dim))
        self.node_mlp = mlp([3 + cfg.material_dim, h, h], cfg.activation)
        self.edge_mlp = mlp([N_EDGE_GEO + 2 * cfg.material_dim, h, h], cfg.activation)

    def node_latents(self, particles: ParticleBatch) -> torch.Tensor:
        z = self.material(particles.type_id)
        feats = torch.stack(
            [particles.mass, particles.radius, particles.inertia], dim=-1
        )
        return self.node_mlp(torch.cat([feats, z], dim=-1))

    def pair_material(self, contacts: ContactSet, particles: ParticleBatch) -> torch.Tensor:
        """Descriptor material simétrico `(z_i + z_o, |z_i - z_o|)`."""
        z = self.material(particles.type_id)
        zi = z[contacts.i]
        zw = self.wall_material.to(zi.dtype).expand_as(zi)
        zj = torch.where(
            contacts.is_wall.unsqueeze(-1), zw, z[contacts.j.clamp(min=0)]
        )
        return torch.cat([zi + zj, (zi - zj).abs()], dim=-1)

    def forward(self, contacts: ContactSet, particles: ParticleBatch):
        h_node = self.node_latents(particles)
        geo = edge_features(contacts, particles)
        h_edge = self.edge_mlp(
            torch.cat([geo, self.pair_material(contacts, particles)], dim=-1)
        )
        return h_node, h_edge, geo
