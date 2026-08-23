"""SLGNN-v3: arquitectura gráfica energético-disipativa e impulsiva.

Implementación de la formulación oficial *SLGNN-v3* (Javier Tapia). Paquete
**independiente** del legacy `slgnn`: no importa `slgnn.model.SLGNN` ni su
integrador, y comparte con los baselines solo la infraestructura neutral de
`slgnn_experiments`.

Ecuación discreta central:

    M (nu_{k+1} - nu_k) = dt F_reg,k + J_k^T Lambda_k,
    F_reg = -grad_q V - d_nu Psi + F^M + F_ext.

Estado del MVP: `V`, `Psi` e `I` activos solo en dirección normal; `M`
(memoria tangencial) y `C` (cierre residual) son contratos declarados sin
implementar. Ver `docs/slgnn_v3/IMPLEMENTATION_STATUS.md`.
"""

from .config import (
    DissipationConfig,
    EncoderConfig,
    GraphConfig,
    ImpactConfig,
    PotentialConfig,
    RouterConfig,
    RouterProfile,
    SolverConfig,
    V3Config,
)
from .contact_kinematics import ContactSet, build_contacts
from .integrator import StepResult
from .model import SLGNNv3
from .router import ContactMode
from .state import ContactMemoryState, ParticleBatch, V3State
from .surfaces import (
    SurfaceSet,
    WallMotion,
    box_surfaces,
    half_space,
    rotating_cylinder_surfaces,
)

__all__ = [
    "ContactMemoryState", "ContactMode", "ContactSet", "DissipationConfig",
    "EncoderConfig", "GraphConfig", "ImpactConfig", "ParticleBatch",
    "PotentialConfig", "RouterConfig", "RouterProfile", "SLGNNv3",
    "SolverConfig", "StepResult", "SurfaceSet", "V3Config", "V3State",
    "WallMotion", "box_surfaces", "build_contacts", "half_space",
    "rotating_cylinder_surfaces",
]
