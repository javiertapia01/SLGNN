"""`GNSClassicReduced`: comparación **secundaria**, cercana al GNS clásico.

Sigue la receta de Sanchez-Gonzalez et al. (2020) en lo esencial:

- secuencia de seis posiciones como entrada;
- cinco velocidades discretas derivadas de esa secuencia;
- aristas dirigidas por radio de conexión;
- desplazamiento relativo y distancia como features de arista;
- distancias a las fronteras, recortadas al radio de conexión;
- decoder de incremento de velocidad (aceleración normalizada);
- modelo reducido para CPU.

**No es la comparación principal** y no debe presentarse como tal: cambia a la
vez la historia disponible (6 frames en lugar de 1), el target (aceleración
normalizada en lugar de `Delta p`) y la representación de la pared (distancias
recortadas en lugar de consultas SDF con velocidad local). Con tres cosas
cambiadas al mismo tiempo, una diferencia de rendimiento no es atribuible.

Licencia: implementación propia contra la descripción publicada. No se ha
copiado ni adaptado código de `geoelements/gns`; si en el futuro se adapta,
debe conservarse su licencia MIT y añadirse `THIRD_PARTY_NOTICES.md`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from slgnn_experiments.scene import shared_graph

from .config import GNSConfig
from .encoder import mlp, to_directed
from .processor import GNSProcessor


class GNSClassicReduced(nn.Module):
    """Requiere una ventana de historia; el runner debe suministrarla."""

    def __init__(self, cfg: GNSConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden
        n_vel = cfg.history_length - 1
        # nodo: n_vel velocidades (3 c/u) + 6 distancias a frontera recortadas
        self.node_mlp = mlp([3 * n_vel + 6 + cfg.material_dim, h, h], cfg.activation)
        self.edge_mlp = mlp([4, h, h], cfg.activation)
        self.material = nn.Embedding(cfg.n_material_types, cfg.material_dim)
        self.processor = GNSProcessor(cfg)
        self.decoder = mlp([h, h, 3], cfg.activation, layer_norm=False)
        nn.init.normal_(self.decoder[-1].weight, std=1e-2)
        nn.init.zeros_(self.decoder[-1].bias)

    def reset_lifecycle(self) -> None:
        """Sin estado de contacto."""

    def n_parameters(self) -> tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        return total, sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def profile(self) -> str:
        return "gns-classic-reduced"

    def forward(self, history: torch.Tensor, particles, box_min, box_max,
                connectivity_radius: float) -> torch.Tensor:
        """`history [H, N, 3]` -> incremento de velocidad `[N, 3]`."""
        vel = history[1:] - history[:-1]                 # [H-1, N, 3]
        q = history[-1]
        lo = torch.as_tensor(box_min, dtype=q.dtype, device=q.device)
        hi = torch.as_tensor(box_max, dtype=q.dtype, device=q.device)
        bounds = torch.cat([q - lo, hi - q], dim=-1).clamp(-connectivity_radius,
                                                           connectivity_radius)
        bounds = bounds / connectivity_radius
        z = self.material(particles.type_id)
        h_node = self.node_mlp(torch.cat(
            [vel.permute(1, 0, 2).reshape(q.shape[0], -1), bounds, z], dim=-1
        ))

        edges = shared_graph(particles.replace(q=q), self.cfg.pp_gap_off, self.cfg.skin)
        directed = to_directed(edges)
        if directed.shape[1]:
            src, dst = directed[0], directed[1]
            dq = (q[dst] - q[src]) / connectivity_radius
            h_edge = self.edge_mlp(
                torch.cat([dq, dq.norm(dim=-1, keepdim=True)], dim=-1)
            )
        else:
            h_edge = torch.zeros(0, self.cfg.hidden, dtype=q.dtype, device=q.device)
        return self.decoder(self.processor(h_node, h_edge, directed))
