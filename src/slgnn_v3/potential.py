"""Cabeza conservativa `V`: potencial elástico repulsivo por construcción.

No se produce un potencial escalar crudo. Se parametriza la **magnitud de la
fuerza normal**, que es no negativa por construcción, y el potencial se
obtiene integrándola (§7 de las instrucciones, §6.2 de la formulación):

    f_n(delta, h) = f0 * a(delta) * softplus(k_theta(delta, h)),   a(0) = 0
    U(delta, h)   = int_0^delta f_n(s, h) ds

De ahí salen gratis las dos condiciones que importan:

    U(0) = 0        (el cero energético está en separación, no en un offset
                     arbitrario que haría "U >= 0" una restricción vacía)
    dU/d(delta) >= 0  (la fuerza elástica es repulsiva)

La integral se evalúa con Gauss–Legendre de orden fijo, con nodos y pesos
registrados como buffers: la cuadratura es una constante del modelo.

La fuerza se obtiene por autograd de `V` respecto de `q`. La gravedad **no**
está aquí: se suma como fuerza externa analítica y se contabiliza aparte
(D-006), de modo que no puede contarse dos veces.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EncoderConfig, PotentialConfig
from .contact_kinematics import ContactSet
from .encoder import mlp
from .smoothing import gauss_legendre_01


class PotentialHead(nn.Module):
    """Magnitud elástica normal y su potencial integrado."""

    def __init__(self, cfg: PotentialConfig, enc: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        # entrada: (s, latente del contacto); s es la compresión de cuadratura
        self.k_net = mlp([1 + enc.hidden, cfg.hidden, cfg.hidden, 1], enc.activation)
        x, w = gauss_legendre_01(cfg.quad_nodes)
        self.register_buffer("quad_x", x)
        self.register_buffer("quad_w", w)

    def _a(self, s: torch.Tensor) -> torch.Tensor:
        """`a(delta)`: `delta` (ley lineal aprendida) o `delta^1.5` (prior Hertz).

        Debe cumplir `a(0) = 0` y `a > 0` para `delta > 0`.
        """
        if self.cfg.exponent == 1.0:
            return s
        return s.clamp_min(0.0) ** self.cfg.exponent

    def normal_force(self, delta: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """`f_n(delta, h) >= 0`, `[C]`."""
        k = self.k_net(torch.cat([delta.unsqueeze(-1), h], dim=-1)).squeeze(-1)
        return self.cfg.f0 * self._a(delta) * F.softplus(k)

    def energy(self, delta: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """`U(delta, h) = int_0^delta f_n(s,h) ds >= 0`, `[C]`.

        Cuadratura sobre `[0, delta]`: `int = delta * sum_g w_g f_n(x_g delta)`.
        Con `delta = 0` el resultado es exactamente `0` sin caso especial.
        """
        if delta.numel() == 0:
            return delta
        x = self.quad_x.to(delta.dtype)          # [G]
        w = self.quad_w.to(delta.dtype)
        s = delta.unsqueeze(-1) * x              # [C, G]
        h_rep = h.unsqueeze(1).expand(-1, x.numel(), -1)
        k = self.k_net(torch.cat([s.unsqueeze(-1), h_rep], dim=-1)).squeeze(-1)
        f = self.cfg.f0 * self._a(s) * F.softplus(k)   # [C, G]
        return delta * (f * w).sum(dim=-1)

    def total_potential(
        self, contacts: ContactSet, h: torch.Tensor, weight: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """`V = sum_alpha weight_alpha * window_alpha * U_alpha`.

        `weight` es `(1 - gamma)`: solo los contactos compliant contribuyen.
        Devuelve el escalar y las energías desglosadas pp / pW.
        """
        if contacts.n_contacts == 0:
            zero = contacts.gap.sum()
            return zero, {"V_pp": zero.detach(), "V_pW": zero.detach()}
        u = self.energy(contacts.delta, h) * contacts.window * weight
        v_pp = u[~contacts.is_wall].sum()
        v_pw = u[contacts.is_wall].sum()
        return v_pp + v_pw, {"V_pp": v_pp.detach(), "V_pW": v_pw.detach()}


def conservative_force(
    V: torch.Tensor, q: torch.Tensor, create_graph: bool
) -> torch.Tensor:
    """`F_V = -grad_q V` por autograd, `[N, 3]`.

    `create_graph=True` habilita el doble backward que necesita una pérdida
    definida sobre las fuerzas.
    """
    if not q.requires_grad:
        raise RuntimeError(
            "conservative_force: q no es una hoja diferenciable. Usa "
            "ParticleBatch.requires_grad_q() antes de evaluar el potencial."
        )
    (grad,) = torch.autograd.grad(
        V, q, create_graph=create_graph, retain_graph=True, allow_unused=True
    )
    return torch.zeros_like(q) if grad is None else -grad
