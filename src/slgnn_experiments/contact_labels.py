"""Clasificación de régimen por transición y por nodo, puramente geométrica.

Se construye **una sola vez** por la infraestructura común y la consumen por
igual el sampler estratificado, las métricas por régimen y el informe. Si v3 y
GNS clasificaran por su cuenta, la comparación dejaría de ser justa aunque los
datos fueran los mismos.

Categorías (§13.3):

    free              ninguna partícula en contacto
    pp_contact        solo contactos partícula-partícula
    pw_contact        solo contactos partícula-pared
    mixed_contact     ambos tipos presentes
    contact_birth     alguna clave nueva respecto del snapshot anterior
    high_compression  penetración por encima de un umbral
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import Trajectory

FREE, PP, PW, MIXED = 0, 1, 2, 3
CATEGORY_NAMES = ("free", "pp", "pw", "mixed")


@dataclass
class RegimeLabels:
    """Etiquetas por transición `k` (de `k` a `k+1`) y por nodo."""

    category: torch.Tensor        # [T-1] long, en {FREE, PP, PW, MIXED}
    node_regime: torch.Tensor     # [T-1, N] long, mismo alfabeto
    birth: torch.Tensor           # [T-1] bool, alguna clave nace
    max_penetration: torch.Tensor  # [T-1]
    n_pp: torch.Tensor            # [T-1] long
    n_pw: torch.Tensor            # [T-1] long

    def counts(self) -> dict[str, int]:
        return {
            name: int((self.category == k).sum())
            for k, name in enumerate(CATEGORY_NAMES)
        }


def _wall_gaps(q: torch.Tensor, radius: torch.Tensor, box_min, box_max) -> torch.Tensor:
    lo = torch.as_tensor(box_min, dtype=q.dtype, device=q.device)
    hi = torch.as_tensor(box_max, dtype=q.dtype, device=q.device)
    per_lo, per_hi = q - lo, hi - q
    phi = torch.stack([per_lo[:, 0], per_hi[:, 0], per_lo[:, 1], per_hi[:, 1],
                       per_lo[:, 2], per_hi[:, 2]], dim=-1)
    return phi - radius.unsqueeze(-1)


def label_transitions(
    tr: Trajectory,
    box_min=None,
    box_max=None,
    contact_gap: float = 0.0,
    high_compression: float = 0.05,
) -> RegimeLabels:
    """Etiqueta las `T-1` transiciones de una trayectoria."""
    T, N = tr.n_steps, tr.n_particles
    K = T - 1
    cat = torch.zeros(K, dtype=torch.long)
    node = torch.zeros(K, N, dtype=torch.long)
    birth = torch.zeros(K, dtype=torch.bool)
    pen = torch.zeros(K, dtype=tr.q.dtype)
    n_pp = torch.zeros(K, dtype=torch.long)
    n_pw = torch.zeros(K, dtype=torch.long)

    prev_keys: set[tuple] = set()
    for k in range(T):
        q = tr.q[k]
        rsum = tr.radius.unsqueeze(0) + tr.radius.unsqueeze(1)
        g_pp = torch.cdist(q, q) - rsum
        touch_pp = torch.triu(g_pp <= contact_gap, diagonal=1)
        idx_pp = touch_pp.nonzero()

        has_pp = torch.zeros(N, dtype=torch.bool)
        if idx_pp.numel():
            has_pp[idx_pp[:, 0]] = True
            has_pp[idx_pp[:, 1]] = True

        has_pw = torch.zeros(N, dtype=torch.bool)
        idx_pw = torch.zeros(0, 2, dtype=torch.long)
        deepest = torch.zeros((), dtype=q.dtype)
        if box_min is not None:
            g_pw = _wall_gaps(q, tr.radius, box_min, box_max)
            idx_pw = (g_pw <= contact_gap).nonzero()
            if idx_pw.numel():
                has_pw[idx_pw[:, 0]] = True
                deepest = torch.maximum(deepest, (-g_pw[g_pw <= contact_gap]).max())
        if idx_pp.numel():
            deepest = torch.maximum(deepest, (-g_pp[touch_pp]).max())

        keys = {("pp", int(a), int(b)) for a, b in idx_pp.tolist()}
        keys |= {("pw", int(a), int(b)) for a, b in idx_pw.tolist()}

        if k < K:
            n_pp[k] = idx_pp.shape[0]
            n_pw[k] = idx_pw.shape[0]
            pen[k] = deepest
            birth[k] = len(keys - prev_keys) > 0
            reg = torch.full((N,), FREE, dtype=torch.long)
            reg[has_pp] = PP
            reg[has_pw] = PW
            reg[has_pp & has_pw] = MIXED
            node[k] = reg
            if has_pp.any() and has_pw.any():
                cat[k] = MIXED
            elif has_pp.any():
                cat[k] = PP
            elif has_pw.any():
                cat[k] = PW
        prev_keys = keys

    return RegimeLabels(
        category=cat, node_regime=node, birth=birth, max_penetration=pen,
        n_pp=n_pp, n_pw=n_pw,
    )


def stratum_of(labels: RegimeLabels, k: int, high_compression: float = 0.05) -> str:
    """Estrato del sampler para la transición `k`. `contact_birth` domina."""
    if bool(labels.birth[k]):
        return "contact_birth"
    if float(labels.max_penetration[k]) > high_compression:
        return "high_compression"
    return CATEGORY_NAMES[int(labels.category[k])]


STRATA = ("free", "pp", "pw", "mixed", "contact_birth", "high_compression")
