"""Configuración tipada de SLGNN-v3.

Todo hiperparámetro vive aquí y se serializa completo en cada checkpoint y
manifiesto. Las unidades son **adimensionales** (ver
`slgnn_experiments.nondimensionalization`): dentro del forward físico no se
convierten unidades.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any


class RouterProfile(str, Enum):
    """Perfil operativo. `HYBRID` está reservado y falla explícitamente."""

    COMPLIANT = "v3-C"
    IMPULSIVE = "v3-I"
    HYBRID = "v3-H"


@dataclass
class GraphConfig:
    """Grafo candidato. Los cortes se expresan sobre el **gap** en unidades de
    `L0`, no sobre la distancia entre centros: así generalizan a radios
    desiguales sin reescalar nada."""

    pp_gap_on: float = 0.02     # ventana C2: peso 1 por debajo
    pp_gap_off: float = 0.35    # ventana C2: peso 0 por encima
    pw_gap_on: float = 0.02
    pw_gap_off: float = 0.35
    skin: float = 0.15          # capa de seguridad de la neighbor list
    ccd: bool = True            # candidatos por cruce previsto con posición libre
    ccd_margin: float = 0.0     # gap libre por debajo del cual se marca candidato


@dataclass
class EncoderConfig:
    hidden: int = 64
    n_material_types: int = 4
    material_dim: int = 8
    n_message_steps: int = 2
    activation: str = "silu"


@dataclass
class PotentialConfig:
    """Cabeza conservativa `V`."""

    hidden: int = 64
    f0: float = 1.0             # escala de fuerza normal
    exponent: float = 1.0       # a(delta) = (delta/L0)^exponent; 1.5 = prior Hertz
    quad_nodes: int = 8         # nodos de Gauss-Legendre para U = int f
    eps_unilateral: float = 0.02  # ancho de la rampa C2 de p_eps, en L0


@dataclass
class DissipationConfig:
    """Cabeza disipativa convexa `Psi`.

    El canal tangencial es disipación continua sin memoria. Produce fuerzas y
    torques (spin) mediante `J^T`, pero no pretende representar por sí solo
    sticking/sliding ni el límite de Coulomb persistente de la cabeza `M`.
    """

    hidden: int = 64
    d0: float = 1.0
    tangential: bool = False    # Psi_tau continua; M sigue siendo necesaria para Coulomb
    rotational: bool = False    # rodadura/torsión: no implementado en el MVP
    eps_tangential: float = 1e-12  # regularización de u_tau / ||u_tau||
    # False conserva la semántica/checkpoints del MVP normal histórico. Todo
    # perfil nuevo con Psi_tau debe usar True para que c1,c2 no dependan de s.
    state_independent_coefficients: bool = False


@dataclass
class ImpactConfig:
    """Cabeza impulsiva `I`."""

    hidden: int = 64
    e_max: float = 1.0          # restitución pasiva
    kappa0: float = 1e-3        # escala de compliance
    friction: bool = False      # fricción impulsiva: no implementada en el MVP


@dataclass
class SolverConfig:
    """Solver normal acoplado."""

    max_iters: int = 40         # iteraciones desenrolladas en entrenamiento
    eval_max_iters: int = 200
    tol: float = 1e-9           # tolerancia de parada, solo en evaluación
    beta: float = 0.2           # estabilización de penetración (Baumgarte)
    kappa_floor: float = 1e-9   # regularización mínima de H
    power_iters: int = 12       # estimación de L = ||H||_2


@dataclass
class RouterConfig:
    profile: RouterProfile = RouterProfile.COMPLIANT
    g_tol: float = 0.0          # contacto impulsivo si g <= g_tol
    g_ccd: float = 0.0          # o si el gap libre previsto cruza g_ccd
    g_on: float = 0.0           # histéresis del lifecycle: nacimiento
    g_off: float = 0.05         # histéresis del lifecycle: ruptura
    n_grace: int = 1            # pasos de gracia antes de borrar una clave
    impulsive_protection: bool = False  # v3-C: red de seguridad, apagada por defecto


@dataclass
class V3Config:
    """Configuración completa de un modelo SLGNN-v3."""

    graph: GraphConfig = field(default_factory=GraphConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    potential: PotentialConfig = field(default_factory=PotentialConfig)
    dissipation: DissipationConfig = field(default_factory=DissipationConfig)
    impact: ImpactConfig = field(default_factory=ImpactConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    router: RouterConfig = field(default_factory=RouterConfig)

    eps: float = 1e-12          # regularización de divisiones, adimensional
    memory_enabled: bool = False   # cabeza M: contrato presente, sin implementar
    closure_enabled: bool = False  # cabeza C: congelada

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["router"]["profile"] = self.router.profile.value
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "V3Config":
        """Construye desde un dict anidado de YAML, rechazando claves ajenas."""
        raw = dict(raw or {})
        sub_types = {
            "graph": GraphConfig, "encoder": EncoderConfig,
            "potential": PotentialConfig, "dissipation": DissipationConfig,
            "impact": ImpactConfig, "solver": SolverConfig, "router": RouterConfig,
        }
        kwargs: dict[str, Any] = {}
        for name, klass in sub_types.items():
            block = dict(raw.pop(name, None) or {})
            if name == "router" and "profile" in block:
                block["profile"] = RouterProfile(block["profile"])
            _reject_unknown(klass, block, name)
            kwargs[name] = klass(**block)
        _reject_unknown(cls, raw, "v3")
        kwargs.update(raw)
        return cls(**kwargs)


def _reject_unknown(klass, block: dict, where: str) -> None:
    known = {f.name for f in fields(klass)}
    unknown = set(block) - known
    if unknown:
        raise ValueError(
            f"Claves desconocidas en la sección '{where}' de la configuración: "
            f"{sorted(unknown)}. Conocidas: {sorted(known)}"
        )
