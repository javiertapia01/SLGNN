"""Cabeza disipativa convexa `Psi`: pasividad por construcción.

Para cada contacto compliant se define `s_n = (-u_n)_+` y

    psi(s) = c1 s^2 / 2 + c2 s^3 / 3,     c1, c2 >= 0
    psi'(s) = c1 s + c2 s^2 >= 0
    psi''(s) = c1 + 2 c2 s >= 0

`c1` y `c2` salen de `softplus`, así que son no negativos, y dependen de
material, gap y compresión — **no de `s`**. El modelo impone esta condición
mediante `encoder.dissipation_context_features`. Esa restricción es el punto
central de §6.4.1 de la formulación: si el coeficiente depende arbitrariamente
de la velocidad,

    Psi(v) = c(v) v^2 / 2   =>   -dPsi/dv = -c(v) v - c'(v) v^2 / 2,

y `c(v) >= 0` ya no basta para garantizar disipación. La familia convexa
explícita conserva no linealidad sin perder pasividad.

La fuerza de contacto asociada es

    lambda^Psi = d_n(s_n) n - d_tau(s_tau) u_tau / ||u_tau||,

con `d = psi'`. Su potencia relativa es

    lambda^Psi . u = -d_n s_n - d_tau s_tau <= 0

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
    """Canales normal y tangencial convexos; rodadura/torsión reservadas."""

    def __init__(self, cfg: DissipationConfig, enc: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.rotational:
            raise NotImplementedError(
                "El canal rotacional directo de Psi no está implementado. "
                "El spin por fuerza tangencial sí está disponible con "
                "dissipation.tangential=True."
            )
        # Dos coeficientes convexos por contacto. La entrada es el latente de
        # Psi, que sí incluye cinemática; la garantía de convexidad no depende
        # de restringir la entrada, sino de que c1, c2 no dependan de `s`.
        self.coef_net = mlp([enc.hidden, cfg.hidden, cfg.hidden, 2], enc.activation)
        self.tangential_coef_net = (
            mlp([enc.hidden, cfg.hidden, cfg.hidden, 2], enc.activation)
            if cfg.tangential else None
        )

    def coefficients(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`(c1, c2)` no negativos, `[C]` cada uno."""
        raw = self.coef_net(h)
        c = self.cfg.d0 * F.softplus(raw)
        return c[..., 0], c[..., 1]

    def tangential_coefficients(
        self, h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Coeficientes no negativos de `psi_tau`; canal explícitamente activo."""
        if self.tangential_coef_net is None:
            raise RuntimeError("Psi_tau está desactivada en la configuración")
        c = self.cfg.d0 * F.softplus(self.tangential_coef_net(h))
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
            diag = {
                "Psi_n": z.sum().detach(), "D_regular": z.sum().detach(),
                "relative_power": z.sum().detach(),
            }
            if self.cfg.tangential:
                diag.update({
                    "Psi_tau": z.sum().detach(),
                    "D_regular_tau": z.sum().detach(),
                    "relative_power_tau": z.sum().detach(),
                })
            return contacts.u.new_zeros(0, 3), z.sum(), diag
        c1, c2 = self.coefficients(h)
        gate = weight * contacts.window
        s_n = positive_part_c2(-contacts.u_n, eps_vel)
        d_n = self.d(s_n, c1, c2) * gate
        lam_n = d_n.unsqueeze(-1) * contacts.n
        psi_n = (self.psi(s_n, c1, c2) * gate).sum()
        power_n = (lam_n * contacts.u).sum(dim=-1)      # <= 0

        lam_tau = torch.zeros_like(lam_n)
        psi_tau = psi_n * 0.0
        power_tau = power_n * 0.0
        diag = {
            "Psi_n": psi_n.detach(),
            "D_regular_n": (-power_n).sum().detach(),
            "relative_power_n": power_n.sum().detach(),
            "c1_mean": c1.mean().detach(),
            "c2_mean": c2.mean().detach(),
        }
        if self.cfg.tangential:
            c1_tau, c2_tau = self.tangential_coefficients(h)
            s_tau = contacts.u_tau.norm(dim=-1)
            d_tau = self.d(s_tau, c1_tau, c2_tau) * gate
            direction = contacts.u_tau / s_tau.clamp_min(
                self.cfg.eps_tangential
            ).unsqueeze(-1)
            lam_tau = -d_tau.unsqueeze(-1) * direction
            psi_tau = (self.psi(s_tau, c1_tau, c2_tau) * gate).sum()
            power_tau = (lam_tau * contacts.u).sum(dim=-1)  # <= 0
            diag.update({
                "Psi_tau": psi_tau.detach(),
                "D_regular_tau": (-power_tau).sum().detach(),
                "relative_power_tau": power_tau.sum().detach(),
                "c1_tau_mean": c1_tau.mean().detach(),
                "c2_tau_mean": c2_tau.mean().detach(),
            })

        lam = lam_n + lam_tau
        psi_total = psi_n + psi_tau
        power = power_n + power_tau
        diag.update({
            "D_regular": (-power).sum().detach(),
            "relative_power": power.sum().detach(),
        })
        return lam, psi_total, diag
