"""Superficies múltiples con SDF diferenciable y cinemática de pared explícita.

Convención (§3.1 de la formulación oficial, D-001):

- `phi > 0` en el interior admisible, `phi = 0` en la pared, `phi < 0` fuera;
- `grad(phi)` apunta hacia el interior;
- el gap partícula–pared es `g = phi(q_i) - R_i`;
- una partícula que se acerca a una pared fija tiene `u_n < 0`.

**Multi-superficie por diseño.** Una caja es un conjunto de seis planos con
`surface_id` estable, no un `min` sobre caras: una partícula en una arista
toca dos caras a la vez y debe producir dos contactos. La SDF global (`min`)
se conserva solo como consulta diagnóstica (`SurfaceSet.global_phi`).

**Geometría y cinemática son entradas separadas** (§3.3): un cilindro
axisimétrico puede rotar sin cambiar su SDF, así que la velocidad local
`v_W(x,t) = V_W(t) + Omega_W(t) x (x - c_W(t))` se entrega aparte y se evalúa
en el punto de pared, nunca en el centro de la partícula.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch


# --------------------------------------------------------------------------
# Cinemática rígida de la pared
# --------------------------------------------------------------------------

@dataclass
class WallMotion:
    """Movimiento rígido prescrito de una superficie.

    `omega_fn(t)` devuelve la velocidad angular [3] y `velocity_fn(t)` la
    velocidad de traslación del centro de referencia [3]. Ambas son funciones
    del tiempo real: propagar `t` correctamente es obligatorio (§4.3).
    """

    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    omega_fn: Callable[[float], Sequence[float]] | None = None
    velocity_fn: Callable[[float], Sequence[float]] | None = None

    def local_velocity(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """`v_W(x,t) = V_W(t) + Omega_W(t) x (x - c_W)`, evaluada en `x`."""
        out = torch.zeros_like(x)
        if self.velocity_fn is not None:
            V = torch.as_tensor(self.velocity_fn(t), dtype=x.dtype, device=x.device)
            out = out + V
        if self.omega_fn is not None:
            W = torch.as_tensor(self.omega_fn(t), dtype=x.dtype, device=x.device)
            c = torch.as_tensor(self.center, dtype=x.dtype, device=x.device)
            out = out + torch.linalg.cross(W.expand_as(x), x - c, dim=-1)
        return out

    @property
    def is_static(self) -> bool:
        return self.omega_fn is None and self.velocity_fn is None


STATIC = WallMotion()


# --------------------------------------------------------------------------
# Superficies
# --------------------------------------------------------------------------

class Surface:
    """Una superficie con identidad estable. `surface_id` indexa la memoria."""

    surface_id: int
    name: str
    motion: WallMotion

    def phi(self, x: torch.Tensor, t: float) -> torch.Tensor:
        raise NotImplementedError

    def normal(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Normal unitaria entrante. Analítica cuando existe; si no, autograd."""
        with torch.enable_grad():
            x_ = x.detach().clone().requires_grad_(True)
            p = self.phi(x_, t)
            (g,) = torch.autograd.grad(p.sum(), x_)
        return g / (g.norm(dim=-1, keepdim=True) + 1e-30)

    def surface_point(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Punto de pared más próximo, `x_W = x - phi(x) n` (§3.3)."""
        n = self.normal(x, t)
        return x - self.phi(x, t).detach().unsqueeze(-1) * n

    def wall_velocity(self, x: torch.Tensor, t: float) -> torch.Tensor:
        return self.motion.local_velocity(x, t)


@dataclass
class Plane(Surface):
    """Semiespacio admisible `{x : n.x >= offset}`. `n` apunta al interior."""

    inward_normal: tuple[float, float, float]
    offset: float
    surface_id: int = 0
    name: str = "plane"
    motion: WallMotion = field(default_factory=lambda: STATIC)

    def _n(self, x: torch.Tensor) -> torch.Tensor:
        n = torch.as_tensor(self.inward_normal, dtype=x.dtype, device=x.device)
        return n / n.norm()

    def phi(self, x: torch.Tensor, t: float) -> torch.Tensor:
        return (x * self._n(x)).sum(dim=-1) - self.offset

    def normal(self, x: torch.Tensor, t: float) -> torch.Tensor:
        return self._n(x).expand_as(x)


@dataclass
class CylinderLateral(Surface):
    """Pared lateral de un cilindro de eje `z`: `phi = R - r`."""

    center_xy: tuple[float, float]
    radius: float
    surface_id: int = 0
    name: str = "cyl_lateral"
    motion: WallMotion = field(default_factory=lambda: STATIC)

    def phi(self, x: torch.Tensor, t: float) -> torch.Tensor:
        dx = x[..., 0] - self.center_xy[0]
        dy = x[..., 1] - self.center_xy[1]
        return self.radius - torch.sqrt(dx * dx + dy * dy + 1e-30)

    def normal(self, x: torch.Tensor, t: float) -> torch.Tensor:
        dx = x[..., 0] - self.center_xy[0]
        dy = x[..., 1] - self.center_xy[1]
        r = torch.sqrt(dx * dx + dy * dy + 1e-30)
        n = torch.stack([-dx / r, -dy / r, torch.zeros_like(r)], dim=-1)
        return n


# --------------------------------------------------------------------------
# Conjunto de superficies y consulta de contactos
# --------------------------------------------------------------------------

@dataclass
class WallQuery:
    """Contactos partícula–superficie dentro de la banda configurada."""

    particle: torch.Tensor      # [C] long
    surface: torch.Tensor       # [C] long, surface_id estable
    batch: torch.Tensor         # [C] long
    phi: torch.Tensor           # [C] en el grafo respecto de q
    gap: torch.Tensor           # [C] phi - R
    normal: torch.Tensor        # [C, 3] entrante
    surface_point: torch.Tensor # [C, 3]
    wall_velocity: torch.Tensor # [C, 3] evaluada en surface_point


class SurfaceSet:
    """Colección de superficies con IDs estables.

    `query` emite **todos** los contactos partícula–superficie dentro de la
    banda, de modo que una esquina de caja produce tres contactos simultáneos.
    """

    def __init__(self, surfaces: Sequence[Surface]):
        self.surfaces = list(surfaces)
        for k, s in enumerate(self.surfaces):
            s.surface_id = k
        ids = [s.surface_id for s in self.surfaces]
        if len(set(ids)) != len(ids):
            raise ValueError(f"surface_id duplicado: {ids}")

    def __len__(self) -> int:
        return len(self.surfaces)

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.surfaces]

    def global_phi(self, q: torch.Tensor, t: float) -> torch.Tensor:
        """SDF global `min_s phi_s`. **Solo diagnóstico**: no es la interfaz de
        contacto (§4.2 de las instrucciones)."""
        if not self.surfaces:
            return torch.full(q.shape[:-1], float("inf"), dtype=q.dtype, device=q.device)
        return torch.stack([s.phi(q, t) for s in self.surfaces], dim=-1).min(dim=-1).values

    def query(
        self,
        q: torch.Tensor,
        radius: torch.Tensor,
        batch_id: torch.Tensor,
        t: float,
        band: float,
    ) -> WallQuery:
        """Contactos con `gap <= band`, uno por cada par (partícula, superficie)."""
        dev, dt_ = q.device, q.dtype
        if not self.surfaces:
            empty_i = torch.zeros(0, dtype=torch.long, device=dev)
            empty_f = torch.zeros(0, dtype=dt_, device=dev)
            empty_v = torch.zeros(0, 3, dtype=dt_, device=dev)
            return WallQuery(empty_i, empty_i, empty_i, empty_f, empty_f,
                             empty_v, empty_v, empty_v)

        phis = torch.stack([s.phi(q, t) for s in self.surfaces], dim=-1)  # [N, S]
        gaps = phis - radius.unsqueeze(-1)
        with torch.no_grad():
            sel = (gaps <= band).nonzero()          # [C, 2] -> (particle, surface)
        pi, si = sel[:, 0], sel[:, 1]

        # phi/gap se re-indexan desde el tensor que sigue en el grafo: la ruta
        # de gradiente hacia q no se corta en ningún punto.
        phi_c = phis[pi, si]
        gap_c = gaps[pi, si]

        normals = torch.zeros(len(pi), 3, dtype=dt_, device=dev)
        xw = torch.zeros(len(pi), 3, dtype=dt_, device=dev)
        vw = torch.zeros(len(pi), 3, dtype=dt_, device=dev)
        for k, s in enumerate(self.surfaces):
            m = si == k
            if not bool(m.any()):
                continue
            xk = q[pi[m]]
            nk = s.normal(xk, t)
            normals[m] = nk
            pk = xk.detach() - phi_c[m].detach().unsqueeze(-1) * nk.detach()
            xw[m] = pk
            vw[m] = s.wall_velocity(pk, t)
        return WallQuery(pi, si, batch_id[pi], phi_c, gap_c, normals, xw, vw)


# --------------------------------------------------------------------------
# Fábricas
# --------------------------------------------------------------------------

def box_surfaces(
    box_min: Sequence[float], box_max: Sequence[float], motion: WallMotion | None = None
) -> SurfaceSet:
    """Seis planos con IDs `0..5` = `-x, +x, -y, +y, -z, +z`."""
    motion = motion or STATIC
    faces: list[Surface] = []
    axis_names = "xyz"
    for ax in range(3):
        faces.append(Plane(
            inward_normal=tuple(1.0 if k == ax else 0.0 for k in range(3)),
            offset=float(box_min[ax]), name=f"-{axis_names[ax]}", motion=motion,
        ))
        faces.append(Plane(
            inward_normal=tuple(-1.0 if k == ax else 0.0 for k in range(3)),
            offset=float(-box_max[ax]), name=f"+{axis_names[ax]}", motion=motion,
        ))
    return SurfaceSet(faces)


def half_space(normal: Sequence[float], offset: float,
               motion: WallMotion | None = None) -> SurfaceSet:
    """Una sola pared plana (benchmark de una esfera contra pared)."""
    return SurfaceSet([Plane(tuple(float(x) for x in normal), float(offset),
                             name="wall", motion=motion or STATIC)])


def rotating_cylinder_surfaces(
    center_xy: Sequence[float], radius: float, z_min: float, z_max: float,
    omega_fn: Callable[[float], Sequence[float]] | None = None,
) -> SurfaceSet:
    """Cilindro de eje `z`: pared lateral (rota) más dos tapas (fijas).

    En el dataset de extrapolación las tapas no rotan, por lo que solo la
    superficie lateral recibe `WallMotion`.
    """
    lateral_motion = WallMotion(
        center=(float(center_xy[0]), float(center_xy[1]), 0.0), omega_fn=omega_fn
    )
    return SurfaceSet([
        CylinderLateral(tuple(float(c) for c in center_xy), float(radius),
                        name="lateral", motion=lateral_motion),
        Plane((0.0, 0.0, 1.0), float(z_min), name="-z"),
        Plane((0.0, 0.0, -1.0), float(-z_max), name="+z"),
    ])


def dynamical_cylinder_omega_literal(t: float) -> tuple[float, float, float]:
    """`omega(t)` del cilindro Dynami-CAL, **fórmula literal** del PDF fuente.

    `omega(t) = 2 pi * {4t si t<0.5; 4-4t si 0.5<=t<=1.5; 0 si t>1.5}`, en
    rad/s alrededor de `+z`. El tambor **invierte el sentido de giro** entre
    `t = 1.0` y `t = 1.5`: verificado contra los datos en `data/DATA_NOTES.md`
    §5 (dos señales independientes). La lectura triangular que implementa
    `slgnn.sdf.dynamical_cylinder_omega` en el legacy está refutada; v3 usa la
    literal y no modifica el legacy.

    El tiempo entra en **segundos físicos**: si el modelo corre adimensional,
    el llamador debe reescalarlo.
    """
    if t < 0.5:
        w = 4.0 * t
    elif t <= 1.5:
        w = 4.0 - 4.0 * t
    else:
        w = 0.0
    return (0.0, 0.0, 2.0 * math.pi * w)
