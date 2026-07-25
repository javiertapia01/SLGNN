"""Operador `S` y programador de refresco (§7.5).

`S` es el arco que falta en el esqueleto de la propuesta: el camino de vuelta
macro → micro. En este banco de pruebas es sencillo porque la carga es
monodispersa, pero se implementa con la firma general para que el andamio ya
tenga la forma correcta cuando haya polidispersidad y medios de molienda.

Aquí vive también el **andamio declarado** de la retroalimentación PSD →
espectro. Está marcado como tal, no está validado por datos, y `AndamioWarning`
lo hace visible en cualquier resultado que lo use.
"""

import warnings
from dataclasses import dataclass, field, replace

import numpy as np

from .coarse import Spectrum
from .library import Regime, SpectrumLibrary
from .macro import MacroState, PBMConfig


class AndamioWarning(UserWarning):
    """Se emite al usar un arco no validado por datos.

    Existe para que un resultado que dependa del andamio no pueda presentarse
    como si tuviera el mismo estatus que el arco `omega → espectro`.
    """


@dataclass
class InitialCondition:
    """Condición inicial micro instanciada desde el estado macro."""

    n_particles: int
    radii: np.ndarray
    q: np.ndarray
    v: np.ndarray
    omega: np.ndarray
    meta: dict = field(default_factory=dict)


def instantiate_micro(z: MacroState, cfg: PBMConfig, template, rng) -> InitialCondition:
    """`z → (N, radios, posiciones, velocidades)`.

    En este banco de pruebas `N` y los radios vienen del dataset (carga
    monodispersa: el estado macro no puede cambiarlos), y lo único que se
    muestrea es la **fase** de la carga — qué instante de la trayectoria de
    referencia se toma como punto de partida. La firma es la general para que la
    versión polidispersa sea un reemplazo de cuerpo, no de interfaz.

    `template` es una trayectoria de referencia con `q, v, omega, radii`.
    """
    n_frames = template.q.shape[0]
    k = int(rng.integers(0, n_frames))
    to_np = lambda x: np.asarray(x.detach().cpu().numpy() if hasattr(x, "detach") else x,
                                 dtype=float)
    radii = to_np(template.radii)
    if cfg.n_classes > 1 and np.ptp(radii) == 0.0:
        # el estado macro tiene 6 clases de tamaño, la micro tiene una sola: el
        # mapeo z → radios NO es identificable con este dataset
        pass
    return InitialCondition(
        n_particles=int(radii.size), radii=radii,
        q=to_np(template.q[k]), v=to_np(template.v[k]),
        omega=to_np(template.omega[k]),
        meta={"phase_index": k, "monodisperse": True,
              "note": "los radios provienen del dataset; z no los determina"},
    )


# ---------------------------------------------------------------------------
# Andamio declarado: PSD → espectro
# ---------------------------------------------------------------------------

def psd_feedback_factor(state: MacroState, cfg: PBMConfig, *, exponent: float = -0.5,
                        warn: bool = True) -> float:
    """ANDAMIO SINTÉTICO — NO VALIDADO POR DATOS.

    Escala la tasa total del espectro por `g(d̄/d_ref) = (d̄/d_ref)^exponent`,
    donde `d̄` es el tamaño medio en masa de la carga. La forma funcional es
    plausible (una carga más fina ofrece más contactos por unidad de masa) y
    tiene un solo parámetro, pero **este dataset no puede verificarla**: la
    carga es monodispersa y el llenado es fijo, de modo que no existe ninguna
    observación con distinta PSD contra la cual contrastarla.

    Todo resultado que active esta función hereda el estatus de andamio.
    """
    if warn:
        warnings.warn(
            "psd_feedback_factor es un andamio sintético: la retroalimentación "
            "PSD → espectro no es verificable con este dataset",
            AndamioWarning, stacklevel=2,
        )
    total = state.M.sum()
    if total <= 0:
        return 1.0
    d_mean = float((state.M * cfg.sizes()).sum() / total)
    return float((d_mean / cfg.d_ref) ** exponent)


# ---------------------------------------------------------------------------
# Políticas de refresco
# ---------------------------------------------------------------------------

class RefreshPolicy:
    """Cuándo volver a consultar el acoplamiento micro-macro."""

    name = "base"

    def should_refresh(self, k: int, state: MacroState, u: np.ndarray) -> bool:
        raise NotImplementedError


class StaticPolicy(RefreshPolicy):
    name = "static"

    def should_refresh(self, k, state, u):
        return k == 0


class PeriodicPolicy(RefreshPolicy):
    name = "periodic"

    def __init__(self, every: int = 5):
        self.every = max(int(every), 1)

    def should_refresh(self, k, state, u):
        return k % self.every == 0


class TriggeredPolicy(RefreshPolicy):
    """Refresca cuando el estado se aleja del punto de la última consulta."""

    name = "triggered"

    def __init__(self, eps: float = 0.05):
        self.eps = float(eps)
        self._last = None

    def should_refresh(self, k, state, u):
        probe = np.concatenate([np.atleast_1d(u), [state.holdup]])
        if self._last is None:
            self._last = probe
            return True
        scale = np.maximum(np.abs(self._last), 1e-12)
        if np.max(np.abs(probe - self._last) / scale) > self.eps:
            self._last = probe
            return True
        return False


class LibraryPolicy(RefreshPolicy):
    """Política operativa del banco de pruebas: consultar siempre la tabla.

    Es barata precisamente porque la biblioteca ya pagó el costo del
    coarse-graining offline; las otras políticas existen para el experimento E3
    reducido, donde la fuente es un rollout y cada consulta cuesta.
    """

    name = "library"

    def should_refresh(self, k, state, u):
        return True


POLICIES = {
    "static": StaticPolicy,
    "periodic": PeriodicPolicy,
    "triggered": TriggeredPolicy,
    "library": LibraryPolicy,
}


def make_policy(name: str, **kwargs) -> RefreshPolicy:
    if name not in POLICIES:
        raise ValueError(f"política desconocida: {name}; opciones {sorted(POLICIES)}")
    return POLICIES[name](**kwargs)


def apply_feedback(regime: Regime, state: MacroState, cfg: PBMConfig, *,
                   enabled: bool, exponent: float = -0.5) -> Regime:
    """Aplica (o no) el andamio PSD → espectro sobre un régimen consultado."""
    if not enabled:
        return regime
    factor = psd_feedback_factor(state, cfg, exponent=exponent)
    scaled: Spectrum = regime.spectrum.with_rates(regime.spectrum.rates * factor)
    return replace(regime, spectrum=scaled,
                   meta={**regime.meta, "psd_feedback_factor": factor,
                         "scaffold": True})
