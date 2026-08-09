"""Processors escalares y simétricos, uno por cabeza física.

El encoder geométrico se comparte, pero `V`, `Psi` e `I` tienen processor
propio (§5.5). Sin esa separación no hay diagnóstico posible: tres MLP
colgando de un mismo latente opaco pueden intercambiarse funciones y el
informe de energía deja de significar nada.

El paso de mensajes opera sobre el grafo de contactos usando la lista de
incidencia del `ContactSet`. Las combinaciones son simétricas
(`h_i + h_j`, `|h_i - h_j|`), así que un par no ordenado produce un único
conjunto de parámetros compartido por `i` y `j`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import EncoderConfig
from .contact_kinematics import ContactSet
from .encoder import mlp


class ContactProcessor(nn.Module):
    """Message passing escalar sobre el grafo de contactos.

    `extra_dim` son features adicionales por contacto (cinemáticas) que solo
    reciben los canales autorizados a verlas.
    """

    def __init__(self, cfg: EncoderConfig, extra_dim: int = 0, n_steps: int | None = None):
        super().__init__()
        h = cfg.hidden
        self.extra_dim = extra_dim
        self.n_steps = cfg.n_message_steps if n_steps is None else n_steps
        self.edge_in = mlp([h + extra_dim, h, h], cfg.activation)
        self.msg = nn.ModuleList(
            mlp([3 * h + extra_dim, h, h], cfg.activation) for _ in range(self.n_steps)
        )
        self.node_up = nn.ModuleList(
            mlp([2 * h, h, h], cfg.activation) for _ in range(self.n_steps)
        )
        self.edge_up = nn.ModuleList(
            mlp([2 * h, h, h], cfg.activation) for _ in range(self.n_steps)
        )
        # Latente del "nodo pared": la pared no es un grado de libertad, pero
        # el mensaje de un contacto de pared necesita un segundo extremo.
        self.wall_node = nn.Parameter(torch.zeros(h))

    def forward(
        self,
        contacts: ContactSet,
        h_node: torch.Tensor,
        h_edge: torch.Tensor,
        extra: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Devuelve el latente por contacto, `[C, hidden]`."""
        C = contacts.n_contacts
        if C == 0:
            return h_edge
        if self.extra_dim:
            assert extra is not None and extra.shape[-1] == self.extra_dim
            h_e = self.edge_in(torch.cat([h_edge, extra], dim=-1))
        else:
            h_e = self.edge_in(h_edge)
            extra = None

        wall_h = self.wall_node.to(h_node.dtype).expand(C, -1)
        for step in range(self.n_steps):
            ha = h_node[contacts.i]
            hb = torch.where(
                contacts.is_wall.unsqueeze(-1), wall_h, h_node[contacts.j.clamp(min=0)]
            )
            pieces = [ha + hb, (ha - hb).abs(), h_e]
            if extra is not None:
                pieces.append(extra)
            m = self.msg[step](torch.cat(pieces, dim=-1))
            m = contacts.window.unsqueeze(-1) * m   # la ventana C2 apaga la arista

            agg = torch.zeros_like(h_node).index_add_(
                0, contacts.inc_node, m[contacts.inc_contact]
            )
            h_node = h_node + self.node_up[step](torch.cat([h_node, agg], dim=-1))
            h_e = h_e + self.edge_up[step](torch.cat([h_e, m], dim=-1))
        return h_e
