"""Cabeza `M`: memoria tangencial persistente. **Interfaz, no implementación.**

En un contacto persistente la fuerza tangencial no depende solo de la
velocidad actual: dos contactos con la misma configuración pueden haber
acumulado desplazamientos tangenciales distintos. Por eso `M` requiere un
estado `xi` que evolucione con `xi^{k+1} = U(xi^k, u_tau, n, dt)`, no una MLP
instantánea (§7 de la formulación, Decisión oficial 3).

Este módulo existe para fijar el contrato ahora y para que nada del MVP pueda
simular memoria por accidente. Toda función levanta `NotImplementedError` con
un mensaje que dice qué falta. **El antiguo canal `H` del legacy no se
reutiliza aquí**: es una función del estado instantáneo y llamarlo memoria fue
precisamente el error que v3 corrige.

Lo que hace falta para implementarlo (fase 10):

1. transporte entre planos tangentes con la rotación mínima `R^{k->k+1}`, y
   una regla robusta documentada cuando las normales son casi opuestas;
2. ley tangencial trial `lambda_trial = -k_tau xi~ - d_tau u_tau/||u_tau||`;
3. proyección al cono de Coulomb `||z|| <= mu f_n^reg` y corrección de `xi`
   en sliding, para que la ley elástica sea compatible con la fuerza
   proyectada;
4. lifecycle con histéresis y periodo de gracia (ya disponible en
   `router.ContactLifecycle`);
5. el término `F^M = sum_alpha J^T (lambda_tau^reg - lambda_tau^Psi)`, que
   evita sumar dos veces el amortiguamiento tangencial.
"""

from __future__ import annotations

import torch

from .contact_kinematics import ContactSet
from .state import ContactMemoryState

_MSG = (
    "La memoria tangencial M no está implementada en el MVP normal de SLGNN-v3. "
    "Requiere transporte entre planos tangentes, ley trial, proyección de "
    "Coulomb y corrección de xi en sliding. Ver docs/v3/IMPLEMENTATION_STATUS.md."
)


def transport(memory: ContactMemoryState, contacts: ContactSet, dt: float):
    raise NotImplementedError(_MSG)


def memory_force(memory: ContactMemoryState, contacts: ContactSet, *args, **kwargs):
    raise NotImplementedError(_MSG)


def finalize(memory: ContactMemoryState, *args, **kwargs):
    raise NotImplementedError(_MSG)


def zero_memory_force(n_nodes: int, dtype, device) -> tuple[torch.Tensor, torch.Tensor]:
    """`F^M = 0` mientras `M` esté desactivada.

    Se devuelve explícitamente cero y se reporta como `disabled` en los
    diagnósticos: no se rellena con un número que parezca un resultado válido.
    """
    z = torch.zeros(n_nodes, 3, dtype=dtype, device=device)
    return z, z.clone()
