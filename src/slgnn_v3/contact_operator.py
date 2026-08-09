"""Operadores de contacto `J`, `J^T` y matriz de Delassus normal.

Nunca se materializa `J` densa (`3C x 6N`). Las dos operaciones se implementan
como dispersión/reducción sobre la lista de incidencia del `ContactSet`, lo
que las hace `O(K)` con `K = 2 C_pp + C_pW`.

La matriz de Delassus normal `A_n = J_n M^-1 J_n^T` **sí** se materializa
densa, pero solo **por componente conexa** del grafo de contactos: es lo que
permite un solve acoplado exacto sin construir un sistema global (§5.2, §8.1
de la formulación). Resolver contacto por contacto cuando comparten partículas
está explícitamente prohibido (§22.6).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .contact_kinematics import ContactSet


# --------------------------------------------------------------------------
# J y J^T
# --------------------------------------------------------------------------

def J_times_velocity(
    contacts: ContactSet, v: torch.Tensor, omega: torch.Tensor,
    subtract_wall: bool = True,
) -> torch.Tensor:
    """`u = J nu (- v_W)`: velocidad relativa en cada punto de contacto, `[C,3]`."""
    C = contacts.n_contacts
    if C == 0:
        return torch.zeros(0, 3, dtype=v.dtype, device=v.device)
    node, arm, sign = contacts.inc_node, contacts.inc_arm, contacts.inc_sign
    contrib = sign.unsqueeze(-1) * (
        v[node] + torch.linalg.cross(omega[node], arm, dim=-1)
    )
    u = torch.zeros(C, 3, dtype=v.dtype, device=v.device).index_add_(
        0, contacts.inc_contact, contrib
    )
    return u - contacts.wall_velocity if subtract_wall else u


def JT_times_contact_vector(
    contacts: ContactSet, lam: torch.Tensor, n_nodes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(F, tau) = J^T lambda`, ambos `[N, 3]`.

    Aplicar un **único** vector por contacto en el punto común es lo que hace
    que acción–reacción y el momento angular orbital más spin se conserven por
    construcción, sin penalización de entrenamiento.
    """
    dev, dt_ = lam.device, lam.dtype
    force = torch.zeros(n_nodes, 3, dtype=dt_, device=dev)
    torque = torch.zeros(n_nodes, 3, dtype=dt_, device=dev)
    if contacts.n_contacts == 0:
        return force, torque
    node, arm, sign = contacts.inc_node, contacts.inc_arm, contacts.inc_sign
    lam_e = lam[contacts.inc_contact]
    f_e = sign.unsqueeze(-1) * lam_e
    t_e = sign.unsqueeze(-1) * torch.linalg.cross(arm, lam_e, dim=-1)
    return force.index_add_(0, node, f_e), torque.index_add_(0, node, t_e)


def JT_times_normal_scalars(
    contacts: ContactSet, lam_n: torch.Tensor, n_nodes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Versión proyectada: `lambda = lambda_n n`."""
    return JT_times_contact_vector(contacts, lam_n.unsqueeze(-1) * contacts.n, n_nodes)


def J_normal(contacts: ContactSet, v: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """Componente normal de la velocidad relativa, `[C]`."""
    return (J_times_velocity(contacts, v, omega) * contacts.n).sum(dim=-1)


# --------------------------------------------------------------------------
# Componentes conexas
# --------------------------------------------------------------------------

@dataclass
class ComponentLayout:
    """Empaquetado de contactos en componentes conexas, con padding."""

    component: torch.Tensor    # [C] long, id de componente por contacto
    slot: torch.Tensor         # [C] long, posición dentro de su componente
    n_components: int
    max_size: int
    valid: torch.Tensor        # [K, max_size] bool
    contact_of: torch.Tensor   # [K, max_size] long, índice global (0 en padding)
    sizes: torch.Tensor        # [K] long


def connected_components(contacts: ContactSet, n_nodes: int) -> ComponentLayout:
    """Componentes conexas del grafo de contactos (union-find sobre nodos).

    Dos contactos están acoplados si comparten una partícula. Un contacto de
    pared y uno partícula–partícula sobre la misma esfera caen en la misma
    componente: eso es exactamente lo que un solve por pares se pierde.
    """
    C = contacts.n_contacts
    dev = contacts.i.device
    if C == 0:
        z = torch.zeros(0, dtype=torch.long, device=dev)
        return ComponentLayout(
            component=z, slot=z, n_components=0, max_size=0,
            valid=torch.zeros(0, 0, dtype=torch.bool, device=dev),
            contact_of=torch.zeros(0, 0, dtype=torch.long, device=dev),
            sizes=torch.zeros(0, dtype=torch.long, device=dev),
        )

    parent = np.arange(n_nodes, dtype=np.int64)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    inc_c = contacts.inc_contact.detach().cpu().numpy()
    inc_n = contacts.inc_node.detach().cpu().numpy()
    # nodos incidentes al mismo contacto se unen
    first: dict[int, int] = {}
    for e in range(inc_c.shape[0]):
        c, node = int(inc_c[e]), int(inc_n[e])
        if c in first:
            union(first[c], node)
        else:
            first[c] = node

    roots = np.array([find(first[c]) for c in range(C)], dtype=np.int64)
    uniq, comp = np.unique(roots, return_inverse=True)
    k = len(uniq)
    sizes = np.bincount(comp, minlength=k)
    max_size = int(sizes.max())

    slot = np.zeros(C, dtype=np.int64)
    cursor = np.zeros(k, dtype=np.int64)
    for c in range(C):
        slot[c] = cursor[comp[c]]
        cursor[comp[c]] += 1

    comp_t = torch.as_tensor(comp, dtype=torch.long, device=dev)
    slot_t = torch.as_tensor(slot, dtype=torch.long, device=dev)
    valid = torch.zeros(k, max_size, dtype=torch.bool, device=dev)
    contact_of = torch.zeros(k, max_size, dtype=torch.long, device=dev)
    valid[comp_t, slot_t] = True
    contact_of[comp_t, slot_t] = torch.arange(C, device=dev)
    return ComponentLayout(
        component=comp_t, slot=slot_t, n_components=k, max_size=max_size,
        valid=valid, contact_of=contact_of,
        sizes=torch.as_tensor(sizes, dtype=torch.long, device=dev),
    )


# --------------------------------------------------------------------------
# Matriz de Delassus normal, densa por componente
# --------------------------------------------------------------------------

def _incidence_pairs(inc_node: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Todos los pares de entradas de incidencia que comparten nodo.

    Devuelve `(ea, eb)` de longitud `sum_p k_p^2`. Vectorizado: no hay bucle
    sobre nodos.
    """
    order = torch.argsort(inc_node, stable=True)
    sorted_nodes = inc_node[order]
    uniq, counts = torch.unique_consecutive(sorted_nodes, return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    sq = counts * counts
    total = int(sq.sum())
    if total == 0:
        z = torch.zeros(0, dtype=torch.long, device=inc_node.device)
        return z, z
    rep = torch.repeat_interleave(torch.arange(uniq.numel(), device=inc_node.device), sq)
    base = torch.repeat_interleave(torch.cumsum(sq, 0) - sq, sq)
    local = torch.arange(total, device=inc_node.device) - base
    k_rep = counts[rep]
    a = local // k_rep
    b = local % k_rep
    return order[starts[rep] + a], order[starts[rep] + b]


def assemble_normal_delassus(
    contacts: ContactSet,
    mass: torch.Tensor,
    inertia: torch.Tensor,
    layout: ComponentLayout,
) -> torch.Tensor:
    """`A_n` densa por componente, `[K, cmax, cmax]`.

        A_n[a,b] = sum_p [ (s_a n_a).(s_b n_b)/m_p
                         + (s_a r_a x n_a).(s_b r_b x n_b)/I_p ]

    donde la suma corre sobre las partículas `p` incidentes a ambos contactos.
    Simétrica y semidefinida positiva por construcción.
    """
    K, cmax = layout.n_components, layout.max_size
    dev, dt_ = contacts.n.device, contacts.n.dtype
    A = torch.zeros(K, cmax, cmax, dtype=dt_, device=dev)
    if contacts.n_contacts == 0:
        return A

    ea, eb = _incidence_pairs(contacts.inc_node)
    if ea.numel() == 0:
        return A
    p = contacts.inc_node[ea]
    ca, cb = contacts.inc_contact[ea], contacts.inc_contact[eb]
    na, nb = contacts.n[ca], contacts.n[cb]
    ta = torch.linalg.cross(contacts.inc_arm[ea], na, dim=-1)
    tb = torch.linalg.cross(contacts.inc_arm[eb], nb, dim=-1)
    s = contacts.inc_sign[ea] * contacts.inc_sign[eb]
    val = s * (
        (na * nb).sum(-1) / mass[p] + (ta * tb).sum(-1) / inertia[p]
    )
    A.index_put_(
        (layout.component[ca], layout.slot[ca], layout.slot[cb]), val, accumulate=True
    )
    return A


def pack_by_component(x: torch.Tensor, layout: ComponentLayout) -> torch.Tensor:
    """`[C]` -> `[K, cmax]` con ceros en el padding."""
    out = torch.zeros(layout.n_components, layout.max_size,
                      dtype=x.dtype, device=x.device)
    if x.numel():
        out = out.index_put((layout.component, layout.slot), x)
    return out


def unpack_by_component(packed: torch.Tensor, layout: ComponentLayout) -> torch.Tensor:
    """`[K, cmax]` -> `[C]`."""
    if layout.component.numel() == 0:
        return torch.zeros(0, dtype=packed.dtype, device=packed.device)
    return packed[layout.component, layout.slot]
