"""Router de régimen determinista y ciclo de vida de contacto (§9).

El router asigna a cada contacto exactamente un modo por paso:

    FREE       no produce respuesta de contacto, aunque exista la arista
    COMPLIANT  contribuye por V + Psi (y, más adelante, M)
    IMPULSIVE  contribuye por el solver de I

Aplicar la respuesta normal regular y la impulsiva con peso completo al mismo
contacto en el mismo intervalo está prohibido: los pesos `(1 - gamma)` y
`gamma` son complementarios y `assert_no_double_counting` lo verifica.

No hay router suave aprendido. Introducirlo antes de identificar los canales
físicos crea no identificabilidad: la red puede trasladar respuesta
arbitrariamente entre potencial e impulso (§17.2 de la formulación).

El ciclo de vida indexa por **clave estable** con histéresis geométrica
`g_on < g_off`, para que el ruido cerca de `g = 0` no cree y destruya un
contacto en cada paso. De él sale el indicador de nacimiento `iota`, que es lo
único que autoriza aplicar restitución.
"""

from __future__ import annotations

from enum import IntEnum

import torch

from .config import RouterConfig, RouterProfile
from .contact_kinematics import ContactSet


class ContactMode(IntEnum):
    FREE = 0
    COMPLIANT = 1
    IMPULSIVE = 2


class ContactLifecycle:
    """Edad y nacimiento por clave estable, con histéresis y periodo de gracia."""

    def __init__(self, cfg: RouterConfig):
        self.g_on = cfg.g_on
        self.g_off = cfg.g_off
        self.n_grace = cfg.n_grace
        self._age: dict[tuple, int] = {}
        self._missing: dict[tuple, int] = {}

    def reset(self) -> None:
        self._age.clear()
        self._missing.clear()

    def snapshot(self) -> dict:
        return {"age": dict(self._age), "missing": dict(self._missing)}

    def restore(self, state: dict) -> None:
        self._age = dict(state["age"])
        self._missing = dict(state["missing"])

    def step(self, contacts: ContactSet) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Avanza un paso. Devuelve `(active, birth, age)`, todos `[C]`.

        - `active`: la clave está en contacto según la histéresis;
        - `birth`: primera vez que esta clave está activa (`iota`);
        - `age`: pasos que lleva activa, `0` en el nacimiento.
        """
        C = contacts.n_contacts
        dev = contacts.gap.device
        active = torch.zeros(C, dtype=torch.bool, device=dev)
        birth = torch.zeros(C, dtype=torch.bool, device=dev)
        age = torch.zeros(C, dtype=torch.long, device=dev)
        if C == 0:
            self._decay(set())
            return active, birth, age

        keys = contacts.keys().detach().cpu().numpy()
        gaps = contacts.gap.detach().cpu().numpy()
        seen: set[tuple] = set()
        for c in range(C):
            key = (int(keys[c, 0]), int(keys[c, 1]), int(keys[c, 2]), int(keys[c, 3]))
            was_active = key in self._age
            threshold = self.g_off if was_active else self.g_on
            if gaps[c] <= threshold:
                seen.add(key)
                active[c] = True
                if was_active:
                    age[c] = self._age[key]
                    self._age[key] += 1
                    self._missing[key] = 0
                else:
                    birth[c] = True
                    self._age[key] = 1
                    self._missing[key] = 0
        self._decay(seen)
        return active, birth, age

    def _decay(self, seen: set[tuple]) -> None:
        for key in list(self._age):
            if key in seen:
                continue
            self._missing[key] = self._missing.get(key, 0) + 1
            if self._missing[key] > self.n_grace:
                del self._age[key]
                del self._missing[key]


def build_router(cfg: RouterConfig):
    """Selecciona el router del perfil. `v3-H` falla explícitamente."""
    if cfg.profile is RouterProfile.COMPLIANT:
        return compliant_router
    if cfg.profile is RouterProfile.IMPULSIVE:
        return impulsive_router
    raise NotImplementedError(
        "El perfil v3-H no está implementado. Requiere, en este orden: "
        "(1) fricción impulsiva con cono de Coulomb, (2) memoria tangencial "
        "persistente con transporte y lifecycle completo, y (3) una transición "
        "explícita y validada entre regímenes. Sin las tres, un router híbrido "
        "duplicaría física o la perdería según el contacto. No hay fallback "
        "silencioso a v3-C ni a v3-I: elige un perfil explícitamente."
    )


def compliant_router(
    contacts: ContactSet,
    cfg: RouterConfig,
    active: torch.Tensor,
    gap_free: torch.Tensor | None = None,
    u_n_free: torch.Tensor | None = None,
) -> torch.Tensor:
    """Perfil `v3-C`: todo contacto dentro de la ventana es COMPLIANT.

    La protección impulsiva está desactivada por defecto y solo debe activarse
    como red de seguridad numérica; con ella encendida, los experimentos de
    identificación de `v3-C` dejan de ser limpios.
    """
    mode = torch.full_like(contacts.i, int(ContactMode.FREE))
    mode = torch.where(
        contacts.window > 0, torch.full_like(mode, int(ContactMode.COMPLIANT)), mode
    )
    if cfg.impulsive_protection and gap_free is not None:
        tunneling = (gap_free <= cfg.g_ccd) & (contacts.gap > cfg.g_tol)
        mode = torch.where(tunneling, torch.full_like(mode, int(ContactMode.IMPULSIVE)), mode)
    return mode


def impulsive_router(
    contacts: ContactSet,
    cfg: RouterConfig,
    active: torch.Tensor,
    gap_free: torch.Tensor | None = None,
    u_n_free: torch.Tensor | None = None,
) -> torch.Tensor:
    """Perfil `v3-I`: contacto actual o cruce previsto, y que se esté cerrando
    o deba sostener una restricción persistente."""
    touching = contacts.gap <= cfg.g_tol
    crossing = torch.zeros_like(touching)
    if gap_free is not None:
        crossing = (contacts.gap > cfg.g_tol) & (gap_free <= cfg.g_ccd)
    closing = u_n_free < 0 if u_n_free is not None else contacts.u_n < 0
    candidate = (touching | crossing) & (closing | (touching & active))
    return torch.where(
        candidate,
        torch.full_like(contacts.i, int(ContactMode.IMPULSIVE)),
        torch.full_like(contacts.i, int(ContactMode.FREE)),
    )


def regular_weight(mode: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """`1 - gamma`: peso de la rama regular."""
    return (mode == int(ContactMode.COMPLIANT)).to(dtype)


def impulsive_mask(mode: torch.Tensor) -> torch.Tensor:
    """`gamma`: máscara de la rama impulsiva."""
    return mode == int(ContactMode.IMPULSIVE)


def assert_no_double_counting(mode: torch.Tensor) -> None:
    """Ningún contacto puede ser compliant e impulsivo a la vez."""
    both = (mode == int(ContactMode.COMPLIANT)) & (mode == int(ContactMode.IMPULSIVE))
    if bool(both.any()):
        raise AssertionError("Router: contacto marcado COMPLIANT e IMPULSIVE a la vez")
    valid = torch.isin(mode, torch.tensor([0, 1, 2], device=mode.device))
    if not bool(valid.all()):
        raise AssertionError(f"Router: modos fuera de rango {mode.unique().tolist()}")
