"""Cabeza disipativa convexa `Psi`: pasividad por construcción.

Para cada contacto compliant se define `s_n = (-u_n)_+` y

    psi(s) = c1 s^2 / 2 + c2 s^3 / 3,     c1, c2 >= 0
    psi'(s) = c1 s + c2 s^2 >= 0
    psi''(s) = c1 + 2 c2 s >= 0

`c1` y `c2` salen de `softplus`, así que son no negativos, y dependen de
material, gap y compresión — **no de `s`**. Esa restricción es el punto
central de §6.4.1 de la formulación: si el coeficiente depende arbitrariamente
de la velocidad,

    Psi(v) = c(v) v^2 / 2   =>   -dPsi/dv = -c(v) v - c'(v) v^2 / 2,

y `c(v) >= 0` ya no basta para garantizar disipación. La familia convexa
explícita conserva no linealidad sin perder pasividad.

La fuerza de contacto asociada es `lambda^Psi = d_n(s_n) n`, con
`d_n = psi'`. Su potencia relativa es

    lambda^Psi . u = d_n u_n = -d_n s_n <= 0

exactamente, no aproximadamente.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DissipationConfig, EncoderConfig
from .contact_kinematics import ContactSet
from .encoder import mlp
from .smoothing import positive_part_c2


class DissipationHead(nn.Module):
    """Canal normal convexo. Tangencial y rotacional declarados, no activos."""

    def __init__(self, cfg: DissipationConfig, enc: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.tangential or cfg.rotational:
            raise NotImplementedError(
                "Los canales tangencial y rotacional de Psi no están implementados "
                "en el MVP normal. Ver docs/v3/IMPLEMENTATION_STATUS.md, fase 9."
            )
        # Dos coeficientes convexos por contacto. La entrada es el latente de
        # Psi, que sí incluye cinemática; la garantía de convexidad no depende
        # de restringir la entrada, sino de que c1, c2 no dependan de `s`.
        self.coef_net = mlp([enc.hidden, cfg.hidden, cfg.hidden, 2], enc.activation)

    def coefficients(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`(c1, c2)` no negativos, `[C]` cada uno."""
        raw = self.coef_net(h)
        c = self.cfg.d0 * F.softplus(raw)
        return c[..., 0], c[..., 1]

    @staticmethod
    def psi(s: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
        """`psi(s) >= 0`, convexa y creciente."""
        return 0.5 * c1 * s**2 + (c2 / 3.0) * s**3

    @staticmethod
    def d(s: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
        """`psi'(s) >= 0`: magnitud disipativa."""
        return c1 * s + c2 * s**2

    def forward(
        self,
        contacts: ContactSet,
        h: torch.Tensor,
        weight: torch.Tensor,
        eps_vel: float = 1e-3,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Devuelve `(lambda_Psi [C,3], Psi escalar, diagnósticos)`.

        `weight` es `(1 - gamma) * activation`: solo contactos compliant y en
        contacto real. La activación unilateral C² evita amortiguar a
        distancia sin introducir un salto en `gap = 0`.
        """
        if contacts.n_contacts == 0:
            z = contacts.gap
            return contacts.u.new_zeros(0, 3), z.sum(), {
                "Psi_n": z.sum().detach(), "D_regular": z.sum().detach(),
                "relative_power": z.sum().detach(),
            }
        c1, c2 = self.coefficients(h)
        gate = weight * contacts.window
        s_n = positive_part_c2(-contacts.u_n, eps_vel)
        d_n = self.d(s_n, c1, c2) * gate
        lam = d_n.unsqueeze(-1) * contacts.n
        psi_total = (self.psi(s_n, c1, c2) * gate).sum()
        power = (lam * contacts.u).sum(dim=-1)          # <= 0 por construcción
        return lam, psi_total, {
            "Psi_n": psi_total.detach(),
            "D_regular": (-power).sum().detach(),
            "relative_power": power.sum().detach(),
            "c1_mean": c1.mean().detach(),
            "c2_mean": c2.mean().detach(),
        }
