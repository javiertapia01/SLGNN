"""Processor de paso de mensajes del baseline GNS.

Bloques residuales encoder–processor–decoder al estilo de Sanchez-Gonzalez
et al. (2020): mensajes sobre aristas dirigidas, agregación por suma en el
nodo destino, y actualización residual de nodo y arista.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import mlp


class GNSProcessor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg.hidden
        self.steps = cfg.n_message_steps
        self.edge_up = nn.ModuleList(
            mlp([3 * h, h, h], cfg.activation) for _ in range(self.steps)
        )
        self.node_up = nn.ModuleList(
            mlp([2 * h, h, h], cfg.activation) for _ in range(self.steps)
        )

    def forward(self, h_node: torch.Tensor, h_edge: torch.Tensor,
                directed: torch.Tensor) -> torch.Tensor:
        if directed.shape[1] == 0:
            return h_node
        src, dst = directed[0], directed[1]
        for k in range(self.steps):
            m = self.edge_up[k](torch.cat([h_node[src], h_node[dst], h_edge], dim=-1))
            h_edge = h_edge + m
            agg = torch.zeros_like(h_node).index_add_(0, dst, m)
            h_node = h_node + self.node_up[k](torch.cat([h_node, agg], dim=-1))
        return h_node
