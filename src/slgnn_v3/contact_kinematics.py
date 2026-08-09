"""Cinemática de contacto unificada: gaps, normales, punto común y brazos.

Un solo `ContactSet` contiene contactos partícula–partícula y partícula–pared.
Unificarlos no es cosmético: el solver impulsivo debe acoplar *todos* los
contactos que comparten una partícula, y una esfera apoyada en el suelo y
golpeada por otra participa de ambos tipos en la misma componente conexa.

Convención única (D-002, §5.1 de las instrucciones):

- par `{i, j}` con `i < j`: `n` va de `i` a `j`, y
  `u = (v_j + w_j x r_j) - (v_i + w_i x r_i)`;
- pared: `u = (v_i + w_i x r_iW) - v_W`, con `n` la normal entrante;
- en ambos casos `u_n < 0` significa aproximación;
- un vector de contacto `lambda` actúa `+lambda` sobre `j` y `-lambda` sobre
  `i` (par), o `+lambda` sobre la partícula (pared).

La unificación se apoya en una **lista de incidencia** `(contacto, nodo, signo,
brazo)`: con ella `J`, `J^T` y la matriz de Delassus se escriben una sola vez
sin ramificar por tipo de contacto.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import GraphConfig, V3Config
from .smoothing import compression, quintic_window
from .state import ParticleBatch
from .surfaces import SurfaceSet, WallQuery

WALL = -1  # marcador de "sin segunda partícula"


@dataclass
class ContactSet:
    """Contactos candidatos con toda su geometría y cinemática diferenciable."""

    # --- identidad (sin gradiente) ---------------------------------------
    i: torch.Tensor           # [C] long
    j: torch.Tensor           # [C] long, WALL para pared
    surface: torch.Tensor     # [C] long, WALL para partícula-partícula
    batch: torch.Tensor       # [C] long
    is_wall: torch.Tensor     # [C] bool

    # --- geometría (diferenciable respecto de q) --------------------------
    n: torch.Tensor           # [C, 3] normal unitaria
    gap: torch.Tensor         # [C]
    delta: torch.Tensor       # [C] compresión unilateral C2
    window: torch.Tensor      # [C] ventana C2 del grafo
    activation: torch.Tensor  # [C] activación unilateral de contacto C2
    x_c: torch.Tensor         # [C, 3] punto común de contacto
    r_i: torch.Tensor         # [C, 3] brazo desde q_i
    r_j: torch.Tensor         # [C, 3] brazo desde q_j (cero si pared)

    # --- cinemática (diferenciable respecto de v, omega) ------------------
    u: torch.Tensor           # [C, 3] velocidad relativa en el punto común
    u_n: torch.Tensor         # [C]
    u_tau: torch.Tensor       # [C, 3]
    wall_velocity: torch.Tensor  # [C, 3], cero para pares

    # --- incidencia (contacto -> nodos) -----------------------------------
    inc_contact: torch.Tensor  # [K] long
    inc_node: torch.Tensor     # [K] long
    inc_sign: torch.Tensor     # [K] dtype de q, +1 / -1
    inc_arm: torch.Tensor      # [K, 3]

    @property
    def n_contacts(self) -> int:
        return int(self.i.shape[0])

    @property
    def n_pp(self) -> int:
        return int((~self.is_wall).sum())

    @property
    def n_pw(self) -> int:
        return int(self.is_wall.sum())

    def keys(self) -> torch.Tensor:
        """Claves estables `[C, 4]`: `(batch, a, b, kind)`.

        Para un par, `(min(i,j), max(i,j), 0)`; para pared, `(i, surface_id, 1)`.
        Son las claves que indexan el lifecycle y, más adelante, la memoria.
        """
        a = torch.where(self.is_wall, self.i, torch.minimum(self.i, self.j))
        b = torch.where(self.is_wall, self.surface, torch.maximum(self.i, self.j))
        kind = self.is_wall.long()
        return torch.stack([self.batch, a, b, kind], dim=-1)

    def subset(self, mask: torch.Tensor) -> "ContactSet":
        """Sub-conjunto de contactos, reindexando la lista de incidencia."""
        idx = mask.nonzero().flatten()
        remap = torch.full((self.n_contacts,), -1, dtype=torch.long, device=idx.device)
        remap[idx] = torch.arange(idx.numel(), device=idx.device)
        keep = remap[self.inc_contact] >= 0
        return ContactSet(
            i=self.i[idx], j=self.j[idx], surface=self.surface[idx],
            batch=self.batch[idx], is_wall=self.is_wall[idx],
            n=self.n[idx], gap=self.gap[idx], delta=self.delta[idx],
            window=self.window[idx], activation=self.activation[idx], x_c=self.x_c[idx],
            r_i=self.r_i[idx], r_j=self.r_j[idx],
            u=self.u[idx], u_n=self.u_n[idx], u_tau=self.u_tau[idx],
            wall_velocity=self.wall_velocity[idx],
            inc_contact=remap[self.inc_contact[keep]],
            inc_node=self.inc_node[keep], inc_sign=self.inc_sign[keep],
            inc_arm=self.inc_arm[keep],
        )


def empty_contacts(particles: ParticleBatch) -> ContactSet:
    dev, dt_ = particles.device, particles.dtype
    li = lambda: torch.zeros(0, dtype=torch.long, device=dev)
    lf = lambda: torch.zeros(0, dtype=dt_, device=dev)
    lv = lambda: torch.zeros(0, 3, dtype=dt_, device=dev)
    return ContactSet(
        i=li(), j=li(), surface=li(), batch=li(),
        is_wall=torch.zeros(0, dtype=torch.bool, device=dev),
        n=lv(), gap=lf(), delta=lf(), window=lf(), activation=lf(),
        x_c=lv(), r_i=lv(), r_j=lv(),
        u=lv(), u_n=lf(), u_tau=lv(), wall_velocity=lv(),
        inc_contact=li(), inc_node=li(), inc_sign=lf(), inc_arm=lv(),
    )


def build_contacts(
    particles: ParticleBatch,
    edges: torch.Tensor,
    wall: WallQuery | None,
    cfg: V3Config,
) -> ContactSet:
    """Construye el `ContactSet` completo desde aristas candidatas y consulta
    de pared. Toda la geometría se recalcula aquí de forma diferenciable."""
    q, v, w = particles.q, particles.v, particles.omega
    R = particles.radius
    g = cfg.graph
    eps_u = cfg.potential.eps_unilateral

    parts: list[dict] = []

    # ---- partícula-partícula --------------------------------------------
    if edges.numel():
        i, j = edges[:, 0], edges[:, 1]
        rij = q[j] - q[i]
        d = rij.norm(dim=-1)
        # Piso, no epsilon aditivo: `d + eps` sesga la normal en todas las
        # distancias (con d = 1.2 y eps = 1e-12 introduce un error relativo de
        # 8e-13 que se propaga al punto común). El clamp solo actúa en el caso
        # degenerado de dos centros coincidentes.
        n = rij / d.clamp_min(cfg.eps).unsqueeze(-1)
        gap = d - (R[i] + R[j])
        # Punto común: punto medio entre las dos superficies no deformadas
        # sobre la línea de centros (D-003),
        #   x_c = ((q_i + R_i n) + (q_j - R_j n)) / 2,
        #   r_i = (q_j - q_i)/2 + (R_i - R_j) n / 2,   r_j = r_i - (q_j - q_i).
        # Escrito así, `r_i - r_j = q_j - q_i` es exacto en punto flotante
        # (no pasa por `d n`, que arrastra el epsilon de la normalización), y
        # con ello la conservación del momento angular sale exacta y no solo
        # correcta hasta 1e-12.
        r_i = 0.5 * rij + (0.5 * (R[i] - R[j])).unsqueeze(-1) * n
        r_j = r_i - rij
        x_c = q[i] + r_i
        u = (v[j] + torch.linalg.cross(w[j], r_j, dim=-1)) - (
            v[i] + torch.linalg.cross(w[i], r_i, dim=-1)
        )
        u_n = (u * n).sum(dim=-1)
        parts.append(dict(
            i=i, j=j, surface=torch.full_like(i, WALL), batch=particles.batch_id[i],
            is_wall=torch.zeros_like(i, dtype=torch.bool),
            n=n, gap=gap, delta=compression(gap, eps_u),
            window=quintic_window(gap, g.pp_gap_on, g.pp_gap_off),
            activation=quintic_window(gap, 0.0, eps_u), x_c=x_c, r_i=r_i, r_j=r_j,
            u=u, u_n=u_n, u_tau=u - u_n.unsqueeze(-1) * n,
            wall_velocity=torch.zeros_like(u),
        ))

    # ---- partícula-pared -------------------------------------------------
    if wall is not None and wall.particle.numel():
        i = wall.particle
        n = wall.normal
        gap = wall.gap
        r_i = -R[i].unsqueeze(-1) * n          # del centro hacia la pared
        x_c = q[i] + r_i
        u = v[i] + torch.linalg.cross(w[i], r_i, dim=-1) - wall.wall_velocity
        u_n = (u * n).sum(dim=-1)
        parts.append(dict(
            i=i, j=torch.full_like(i, WALL), surface=wall.surface, batch=wall.batch,
            is_wall=torch.ones_like(i, dtype=torch.bool),
            n=n, gap=gap, delta=compression(gap, eps_u),
            window=quintic_window(gap, g.pw_gap_on, g.pw_gap_off),
            activation=quintic_window(gap, 0.0, eps_u), x_c=x_c, r_i=r_i, r_j=torch.zeros_like(r_i),
            u=u, u_n=u_n, u_tau=u - u_n.unsqueeze(-1) * n,
            wall_velocity=wall.wall_velocity,
        ))

    if not parts:
        return empty_contacts(particles)

    cat = lambda k: torch.cat([p[k] for p in parts], dim=0)
    cs = dict((k, cat(k)) for k in parts[0])
    inc = _incidence(cs["i"], cs["j"], cs["is_wall"], cs["r_i"], cs["r_j"], q.dtype)
    return ContactSet(**cs, **inc)


def _incidence(i, j, is_wall, r_i, r_j, dtype) -> dict:
    """Lista `(contacto, nodo, signo, brazo)`.

    El signo codifica la convención de una sola vez: `lambda` actúa `-1` sobre
    `i` y `+1` sobre `j` en un par, y `+1` sobre la partícula en pared.
    """
    C = i.shape[0]
    cid = torch.arange(C, device=i.device)
    sign_i = torch.where(is_wall, torch.ones_like(i, dtype=torch.uint8),
                         torch.zeros_like(i, dtype=torch.uint8)).to(dtype) * 2.0 - 1.0
    # is_wall -> +1, par -> -1
    ent_c = [cid]
    ent_n = [i]
    ent_s = [sign_i]
    ent_a = [r_i]
    pp = (~is_wall).nonzero().flatten()
    if pp.numel():
        ent_c.append(cid[pp])
        ent_n.append(j[pp])
        ent_s.append(torch.ones(pp.numel(), dtype=dtype, device=i.device))
        ent_a.append(r_j[pp])
    return dict(
        inc_contact=torch.cat(ent_c), inc_node=torch.cat(ent_n),
        inc_sign=torch.cat(ent_s), inc_arm=torch.cat(ent_a),
    )
