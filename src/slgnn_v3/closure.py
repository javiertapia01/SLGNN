"""Cabeza `C`: cierre residual opcional. **Congelada y desactivada.**

Se reserva para fuerzas no resueltas por el estado observado —fluido no
modelado, rugosidad efectiva, error sistemático del modelo reducido— y
**solo** puede activarse después de identificar `V`, `Psi`, `M` e `I`.

No se activa nunca para compensar (§10.1 de la formulación):

- un signo incorrecto;
- una SDF desconectada del grafo de autograd;
- detección tardía de contacto;
- ausencia de spin o de memoria;
- un solver que no converge;
- doble contabilización entre rama regular e impulsiva.

Activarla antes de validar los canales físicos mejora las métricas y destruye
la identificabilidad, que es justo lo que v3 existe para preservar.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ClosureHead(nn.Module):
    """Estructura admisible, inicializada en cero y congelada.

    Produce magnitudes escalares por arista sobre bases geométricas,
    `lambda^C = a_n n + a_tau u_tau/||u_tau|| + a_xi xi/||xi||`, y se aplica
    con `J^T` para conservar acción–reacción. En el MVP, `forward` devuelve
    cero exacto y no tiene parámetros entrenables activos.
    """

    def __init__(self, enabled: bool = False):
        super().__init__()
        if enabled:
            raise NotImplementedError(
                "El cierre residual C no puede activarse en el MVP. Requiere un "
                "análisis de residuales que identifique estructura reproducible "
                "no explicada por V, Psi, M e I. Ver docs/slgnn_v3/DECISIONS.md."
            )
        self.enabled = False

    def forward(self, contacts) -> torch.Tensor:
        return torch.zeros_like(contacts.u)

    @property
    def status(self) -> str:
        return "frozen/disabled"
