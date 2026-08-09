"""Cabeza impulsiva `I`: parámetros de un solver, no un impulso cartesiano.

Produce por candidato impulsivo

    e     = e_max * sigmoid(~e)        restitución efectiva, en [0, e_max]
    kappa = kappa0 * softplus(~kappa)  compliance/regularización, >= 0
    mu    = None                       fricción: no implementada en el MVP

La masa efectiva **no se aprende**: se calcula de las masas conocidas y se
entrega como entrada. En el MVP normal `mu` se devuelve explícitamente como
`None` en lugar de un tensor de ceros, para que no pueda confundirse con
"fricción aprendida igual a cero" (D-012).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EncoderConfig, ImpactConfig
from .contact_kinematics import ContactSet
from .encoder import mlp
from .state import ParticleBatch


@dataclass
class ImpactParams:
    e: torch.Tensor              # [C]
    kappa: torch.Tensor          # [C]
    mu: torch.Tensor | None      # None mientras la fricción no exista
    m_eff: torch.Tensor          # [C] conocida, no aprendida
    R_eff: torch.Tensor          # [C]


def effective_mass(contacts: ContactSet, particles: ParticleBatch) -> torch.Tensor:
    """`m_eff`: `(1/m_i + 1/m_j)^-1` para un par, `m_i` para pared."""
    mi = particles.mass[contacts.i]
    inv = 1.0 / mi
    pp = ~contacts.is_wall
    if bool(pp.any()):
        mj = particles.mass[contacts.j.clamp(min=0)]
        inv = torch.where(pp, 1.0 / mi + 1.0 / mj, inv)
    return 1.0 / inv


def effective_radius(contacts: ContactSet, particles: ParticleBatch) -> torch.Tensor:
    ri = particles.radius[contacts.i]
    inv = 1.0 / ri
    pp = ~contacts.is_wall
    if bool(pp.any()):
        rj = particles.radius[contacts.j.clamp(min=0)]
        inv = torch.where(pp, 1.0 / ri + 1.0 / rj, inv)
    return 1.0 / inv


class ImpactHead(nn.Module):
    """Predice `(e, kappa)` desde descriptores invariantes preimpacto."""

    # gap, u_n^-, ||u_tau^-||, m_eff, R_eff, iota, edad
    N_PREIMPACT = 7

    def __init__(self, cfg: ImpactConfig, enc: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.friction:
            raise NotImplementedError(
                "La fricción impulsiva no está implementada en el MVP normal. "
                "Ver docs/v3/IMPLEMENTATION_STATUS.md, fase 9."
            )
        self.net = mlp(
            [self.N_PREIMPACT + enc.hidden, cfg.hidden, cfg.hidden, 2], enc.activation
        )
        # Inicialización de la última capa: pesos pequeños (no exactamente
        # cero) y sesgo nulo. Arranca en e ~ e_max/2 y kappa pequeña, que es
        # numéricamente estable, pero **sin** anular el gradiente hacia el
        # processor y el encoder: con W = 0 exacto, dL/dentrada = W^T g = 0 y
        # las capas de arriba no reciben señal en el primer paso.
        nn.init.normal_(self.net[-1].weight, std=1e-2)
        nn.init.zeros_(self.net[-1].bias)

    def preimpact_features(
        self,
        contacts: ContactSet,
        particles: ParticleBatch,
        u_n_free: torch.Tensor,
        u_tau_free: torch.Tensor,
        birth: torch.Tensor,
        age: torch.Tensor,
        m_eff: torch.Tensor,
        R_eff: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack([
            contacts.gap,
            u_n_free,
            u_tau_free.norm(dim=-1),
            m_eff,
            R_eff,
            birth.to(contacts.gap.dtype),
            age.to(contacts.gap.dtype),
        ], dim=-1)

    def forward(
        self,
        contacts: ContactSet,
        particles: ParticleBatch,
        h: torch.Tensor,
        u_n_free: torch.Tensor,
        u_tau_free: torch.Tensor,
        birth: torch.Tensor,
        age: torch.Tensor,
    ) -> ImpactParams:
        m_eff = effective_mass(contacts, particles)
        R_eff = effective_radius(contacts, particles)
        if contacts.n_contacts == 0:
            z = contacts.gap
            return ImpactParams(e=z, kappa=z, mu=None, m_eff=z, R_eff=z)
        feats = self.preimpact_features(
            contacts, particles, u_n_free, u_tau_free, birth, age, m_eff, R_eff
        )
        raw = self.net(torch.cat([feats, h], dim=-1))
        e = self.cfg.e_max * torch.sigmoid(raw[..., 0])
        kappa = self.cfg.kappa0 * F.softplus(raw[..., 1])
        return ImpactParams(e=e, kappa=kappa, mu=None, m_eff=m_eff, R_eff=R_eff)
