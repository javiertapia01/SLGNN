"""Redes neuronales de SLGNN-v2: todas producen escalares invariantes (§11).

Los vectores físicos nunca salen de una MLP; se construyen después sobre
bases geométricas equivariantes (e_ij, nu_i, u_tau, w_tau). Activaciones
softplus (C²) para que las energías sean al menos dos veces diferenciables
(§15).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .state import Particles


def mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for k in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[k], sizes[k + 1]))
        if k < len(sizes) - 2:
            layers.append(nn.Softplus())
    return nn.Sequential(*layers)


def zero_init_last(net: nn.Sequential) -> nn.Sequential:
    """Anula la última capa: el canal histórico parte exactamente en cero,
    forzando a que energía y Rayleigh expliquen primero la dinámica (§34.5)."""
    last = net[-1]
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    return net


def bias_init_last(net: nn.Sequential, value: float) -> nn.Sequential:
    nn.init.constant_(net[-1].bias, value)
    return net


class MaterialEncoder(nn.Module):
    """Embedding material eta_i = MLP(onehot(tau), m, R, mu) (§12)."""

    def __init__(self, n_types: int, n_props: int, h_mat: int):
        super().__init__()
        self.n_types = n_types
        self.net = mlp([n_types + 2 + n_props, h_mat, h_mat])

    def forward(self, particles: Particles, dtype: torch.dtype) -> torch.Tensor:
        onehot = F.one_hot(particles.type_ids, self.n_types).to(dtype)
        feats = torch.cat(
            [
                onehot,
                particles.m.to(dtype).unsqueeze(-1),
                particles.radii.to(dtype).unsqueeze(-1),
                particles.props.to(dtype),
            ],
            dim=-1,
        )
        return self.net(feats)


def symmetric_pair(eta_i: torch.Tensor, eta_j: torch.Tensor) -> torch.Tensor:
    """Descriptor simétrico de arista (§13): [eta+; |eta-|; eta_i*eta_j]."""
    return torch.cat(
        [eta_i + eta_j, (eta_i - eta_j).abs(), eta_i * eta_j], dim=-1
    )


class ScalarProcessor(nn.Module):
    """Message passing escalar simétrico (§15).

    Mensajes no dirigidos m_ij = m_ji por construcción (los nodos entran como
    h_i + h_j y |h_i - h_j|), gateados por la ventana chi para que la
    contribución de una arista se anule suavemente antes de salir del grafo
    (§10). Actualizaciones residuales de nodo y arista.
    """

    def __init__(self, node_in: int, edge_in: int, hidden: int, layers: int):
        super().__init__()
        self.enc_node = mlp([node_in, hidden, hidden])
        self.enc_edge = mlp([edge_in, hidden, hidden])
        self.psi_msg = nn.ModuleList(
            mlp([3 * hidden, hidden, hidden]) for _ in range(layers)
        )
        self.psi_node = nn.ModuleList(
            mlp([2 * hidden + node_in, hidden, hidden]) for _ in range(layers)
        )
        self.psi_edge = nn.ModuleList(
            mlp([3 * hidden, hidden, hidden]) for _ in range(layers)
        )

    def forward(self, s_node, s_edge, idx_i, idx_j, chi):
        h = self.enc_node(s_node)
        e = self.enc_edge(s_edge)
        cw = chi.unsqueeze(-1)
        for psi_m, psi_n, psi_e in zip(self.psi_msg, self.psi_node, self.psi_edge):
            msg = cw * psi_m(
                torch.cat([e, h[idx_i] + h[idx_j], (h[idx_i] - h[idx_j]).abs()], dim=-1)
            )
            agg = torch.zeros_like(h).index_add(0, idx_i, msg).index_add(0, idx_j, msg)
            h = h + psi_n(torch.cat([h, agg, s_node], dim=-1))
            e = e + cw * psi_e(
                torch.cat([e, h[idx_i] + h[idx_j], (h[idx_i] - h[idx_j]).abs()], dim=-1)
            )
        return h, e
