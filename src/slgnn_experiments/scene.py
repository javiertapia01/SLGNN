"""Construcción neutral de la escena: superficies, gravedad y grafo candidato.

Punto único donde se decide **qué ve** un modelo de un dataset. v3 y GNS
reciben de aquí exactamente el mismo `ParticleBatch`, la misma `SurfaceSet`,
el mismo vector de gravedad y las mismas aristas candidatas; a partir de ahí
difieren solo en cómo transforman esa información en `(Delta p, Delta L)`.

Nota estructural (D-016): este módulo importa `slgnn_v3.state`,
`slgnn_v3.surfaces` y `slgnn_v3.graph`. Esos tres módulos no contienen física
aprendida ni parámetros: son contenedores tipados y geometría. La prohibición
de §15.1 es que `gns_baseline` no importe **el modelo** v3, y se cumple: el
baseline importa solo `slgnn_experiments`, y un test lo verifica sobre el
código fuente.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from slgnn_v3.graph import build_candidate_graph
from slgnn_v3.state import ParticleBatch, V3State
from slgnn_v3.surfaces import (
    SurfaceSet,
    box_surfaces,
    dynamical_cylinder_omega_literal,
    half_space,
    rotating_cylinder_surfaces,
)

from .data import DATASETS, Trajectory
from .nondimensionalization import Scales

_AXIS = {"x": 0, "y": 1, "z": 2}


@dataclass
class Scene:
    """Todo lo que un modelo necesita saber del entorno, ya adimensionalizado."""

    surfaces: SurfaceSet
    gravity: torch.Tensor | None
    dt: float
    scales: Scales
    dataset_key: str

    def state_at(self, tr: Trajectory, k: int, type_id=None) -> V3State:
        """`V3State` del snapshot `k`, con el tiempo real correspondiente."""
        pb = ParticleBatch.from_arrays(
            q=tr.q[k], v=tr.v[k], omega=tr.omega[k],
            mass=tr.mass, radius=tr.radius, inertia=tr.inertia, type_id=type_id,
        )
        return V3State(pb, time=tr.t0 + k * tr.dt)

    def batch_at(self, items: list[tuple[Trajectory, int]], type_id=None) -> V3State:
        """Batch concatenado de varias transiciones. El tiempo del batch es el
        del primer elemento; cada sistema conserva su propio `batch_id`."""
        parts = [
            ParticleBatch.from_arrays(
                q=tr.q[k], v=tr.v[k], omega=tr.omega[k],
                mass=tr.mass, radius=tr.radius, inertia=tr.inertia, type_id=type_id,
            )
            for tr, k in items
        ]
        pb = ParticleBatch.concat(parts)
        tr0, k0 = items[0]
        return V3State(pb, time=tr0.t0 + k0 * tr0.dt)


def build_scene(dataset_key: str, scales: Scales) -> Scene:
    """Escena adimensional de un dataset, desde su especificación registrada."""
    spec = DATASETS[dataset_key]
    dt = spec.dt / scales.T0

    if spec.geometry == "box":
        lo = [scales.length(x) for x in spec.box_min]
        hi = [scales.length(x) for x in spec.box_max]
        surfaces = box_surfaces(lo, hi)
    elif spec.geometry == "cylinder":
        # Cilindro de extrapolación: r = 0.05 m, eje z, centro (0, 0.002),
        # z en [0, 0.1] (data/DATA_NOTES.md §5). `omega(t)` es la fórmula
        # literal del PDF —el tambor invierte el giro—, con el tiempo
        # reconvertido a segundos físicos porque el perfil está definido ahí.
        def omega_dimensionless(t_prime: float):
            w = dynamical_cylinder_omega_literal(t_prime * scales.T0)
            return tuple(x * scales.T0 for x in w)

        surfaces = rotating_cylinder_surfaces(
            (scales.length(0.0), scales.length(0.002)),
            scales.length(0.05), scales.length(0.0), scales.length(0.1),
            omega_fn=omega_dimensionless,
        )
    else:
        surfaces = SurfaceSet([])

    gravity = None
    if spec.gravity and spec.gravity_axis:
        g = torch.zeros(3, dtype=torch.float64)
        g[_AXIS[spec.gravity_axis]] = -scales.gravity(spec.gravity)
        gravity = g

    return Scene(surfaces=surfaces, gravity=gravity, dt=dt, scales=scales,
                 dataset_key=dataset_key)


def shared_graph(particles: ParticleBatch, gap_off: float, skin: float) -> torch.Tensor:
    """Aristas candidatas. Idéntica llamada para v3 y para GNS."""
    return build_candidate_graph(particles, gap_off, skin)


def active_contact_keys(particles, surfaces, t: float, gap_off: float = 0.35,
                        skin: float = 0.15):
    """Claves de los contactos activos `(a, b, kind)` y sus gaps.

    Neutral y determinista: se aplica igual al estado predicho por cualquier
    modelo y al estado real del DEM, de modo que precisión y recall comparen
    conjuntos construidos con el mismo criterio.
    """
    from slgnn_v3 import V3Config
    from slgnn_v3.contact_kinematics import build_contacts

    cfg = V3Config()
    wall = surfaces.query(particles.q, particles.radius, particles.batch_id,
                          t, gap_off)
    edges = build_candidate_graph(particles, gap_off, skin)
    cs = build_contacts(particles, edges, wall, cfg)
    if cs.n_contacts == 0:
        return set(), particles.q.new_zeros(0)
    active = cs.gap <= 0
    keys = {tuple(int(x) for x in row)
            for row in cs.keys()[active].detach().cpu().numpy()}
    return keys, cs.gap.detach()


def active_contact_normals(state, surfaces, gap_off: float = 0.35,
                           skin: float = 0.15):
    """Normales de los contactos **activos** y el nodo al que aplica cada una.

    Neutral: se calcula igual para v3 y para GNS, de modo que la separación
    normal/tangencial del error sea la misma medida para ambos y no dependa de
    lo que cada modelo crea que es un contacto.
    """
    from slgnn_v3 import V3Config
    from slgnn_v3.contact_kinematics import build_contacts

    cfg = V3Config()
    pb = state.particles
    wall = surfaces.query(pb.q, pb.radius, pb.batch_id, state.time_scalar(), gap_off)
    edges = build_candidate_graph(pb, gap_off, skin)
    cs = build_contacts(pb, edges, wall, cfg)
    if cs.n_contacts == 0:
        return None
    active = cs.gap <= 0
    if not bool(active.any()):
        return None
    sub = cs.subset(active)
    return sub.n[sub.inc_contact].detach(), sub.inc_node
