"""MPC restringido y baselines (§7.6).

Asimetría que hay que tener presente en todo lo que salga de aquí: **la única
acción con respaldo microdinámico es omega.** `f_feed` entra solo por el balance
macroscópico — no hay ningún dato que diga cómo la alimentación modifica el
espectro de colisiones. Las dos acciones aparecen juntas en `u`, pero no tienen
el mismo estatus, y `ControlResult.meta` lo arrastra hasta el reporte.

El objetivo económico está en unidades del tambor (§6): masa producida bajo el
tamaño objetivo por unidad de energía suministrada. No hay ficción de
escalamiento a unidades industriales.

Optimizador: proyección + Adam con gradiente por diferencias finitas. Son 10×2
variables y cada evaluación cuesta microsegundos, así que no justifica una
dependencia nueva; `scipy.optimize` (SLSQP) se usa automáticamente si está
disponible.
"""

from dataclasses import dataclass, field, replace

import numpy as np

from .coarse import Spectrum
from .confidence import ConfidenceMonitor, ConfidenceReport
from .coupling import RefreshPolicy, apply_feedback, make_policy
from .library import Regime, SpectrumLibrary
from .macro import PBM, MacroState, selection_rates

try:  # opcional: si está, se usa; si no, el optimizador propio basta
    from scipy.optimize import minimize as _scipy_minimize
except ImportError:  # pragma: no cover - depende del entorno
    _scipy_minimize = None


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

@dataclass
class MPCConfig:
    horizon: int = 10
    dt: float = 0.5                # s por paso macro

    # pesos del costo
    lambda_q: float = 1.0          # premio a la producción bajo tamaño
    lambda_e: float = 1.0e-4       # castigo a la energía específica [J/kg]
    lambda_du: float = 1.0e-2      # suavidad de la acción

    # restricciones
    omega_min: float = 0.0         # se recorta al rango cubierto por la rampa
    omega_max: float = 15.0        # rad/s
    feed_min: float = 0.0
    feed_max: float = 0.05         # kg/s
    p_max: float = np.inf          # W
    h_max: float = np.inf          # kg
    d_omega_max: float = 2.0       # rad/s por paso macro

    # cómo se obtiene el espectro dentro del horizonte
    #   "library"     : C_k = biblioteca(omega_k)  — el puente activo
    #   "frozen"      : C fijo al omega actual, refrescado entre resoluciones
    #   "static"      : C fijo al omega nominal, ignora el estado
    #   "fixed_rates" : S_b constante, el PBM no ve el espectro
    spectrum_mode: str = "library"
    omega_nominal: float = 6.0

    # andamio PSD → espectro (§7.5). Apagado por defecto: no está validado.
    psd_feedback: bool = False
    psd_exponent: float = -0.5

    # optimizador
    iters: int = 80
    lr: float = 0.15
    penalty: float = 1.0e3
    fd_eps: float = 1e-3
    seed: int = 0


@dataclass
class ControlResult:
    u: np.ndarray                  # [horizon, 2] acción planificada
    cost: float
    predicted: list = field(default_factory=list)
    confidence: ConfidenceReport | None = None
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Modelo de predicción F_psi
# ---------------------------------------------------------------------------

class MacroPlant:
    """`z_{k+1} = F_psi(z_k, u_k, C_k)` — barato porque `C` sale de la tabla.

    El compromiso de la propuesta se respeta literalmente: el núcleo granular
    nunca entra en el lazo. Lo que entra es una interpolación de resultados de
    coarse-graining calculados fuera de línea.
    """

    def __init__(self, pbm: PBM, library: SpectrumLibrary, cfg: MPCConfig):
        self.pbm = pbm
        self.library = library
        self.cfg = cfg
        self._cache: dict[float, Regime] = {}

    def regime(self, omega: float) -> Regime:
        # 1e-3 rad/s es más fino que cualquier resolución de control con sentido
        # físico, y acota la caché para que el optimizador no la haga inútil al
        # perturbar omega de forma continua.
        key = round(float(omega), 3)
        if key not in self._cache:
            self._cache[key] = self.library.query(key)
        return self._cache[key]

    def spectrum_for(self, omega: float, state: MacroState, frozen: Regime | None):
        mode = self.cfg.spectrum_mode
        if mode == "library":
            reg = self.regime(omega)
        elif mode == "frozen":
            reg = frozen if frozen is not None else self.regime(omega)
        elif mode in ("static", "fixed_rates"):
            reg = self.regime(self.cfg.omega_nominal)
        else:
            raise ValueError(f"spectrum_mode desconocido: {mode}")
        reg = apply_feedback(reg, state, self.pbm.cfg,
                             enabled=self.cfg.psd_feedback,
                             exponent=self.cfg.psd_exponent)
        return reg

    def rollout(self, state: MacroState, u: np.ndarray, frozen: Regime | None):
        """Simula el horizonte. Devuelve la traza y el costo acumulado."""
        cfg = self.cfg
        trace, cost, violation = [], 0.0, 0.0
        z = state.copy()
        u_prev = u[0]
        fixed_S = None
        if cfg.spectrum_mode == "fixed_rates":
            reg0 = self.regime(cfg.omega_nominal)
            fixed_S = selection_rates(reg0.spectrum, self.pbm.cfg)

        for k in range(u.shape[0]):
            omega, feed = float(u[k, 0]), float(u[k, 1])
            reg = self.spectrum_for(omega, z, frozen)
            S = fixed_S if fixed_S is not None else selection_rates(reg.spectrum,
                                                                    self.pbm.cfg)
            # Energía suministrada: se usa la potencia de IMPACTO, que sale
            # directamente de los eventos (Σ E·n·masa) y por tanto es positiva
            # por construcción y no depende de `kappa`. El balance `p_in` es
            # derivado y, sobre el cilindro, sale negativo en varias ventanas
            # porque `kappa` se calibró en otra geometría; usarlo aquí
            # convertiría el término de energía del costo en un premio.
            p_in = self.regime(omega).p_impact
            z = self.pbm.step(z, S, feed, cfg.dt)
            q = self.pbm.product_rate(z)

            specific = p_in / q if q > 1e-12 else cfg.penalty
            cost += -cfg.lambda_q * q + cfg.lambda_e * specific
            du = u[k] - u_prev
            cost += cfg.lambda_du * float(du @ du)
            u_prev = u[k]

            violation += max(0.0, p_in - cfg.p_max) ** 2
            violation += max(0.0, z.holdup - cfg.h_max) ** 2
            violation += max(0.0, abs(du[0]) - cfg.d_omega_max) ** 2
            trace.append({"k": k, "omega": omega, "feed": feed, "p_in": p_in,
                          "product": q, "holdup": z.holdup,
                          "p80": self.pbm.p80(z), "specific_energy": specific})
        return trace, cost + cfg.penalty * violation, z


# ---------------------------------------------------------------------------
# Optimizador
# ---------------------------------------------------------------------------

def _project(u: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return np.clip(u, bounds[:, 0], bounds[:, 1])


def _minimize(fun, u0: np.ndarray, bounds: np.ndarray, cfg: MPCConfig):
    """Adam proyectado con gradiente por diferencias finitas hacia adelante.

    Determinista. Si `scipy` está instalado se prefiere SLSQP, que converge en
    menos evaluaciones; el resultado no debería depender de cuál se use.
    """
    flat_bounds = [(bounds[i % bounds.shape[0], 0], bounds[i % bounds.shape[0], 1])
                   for i in range(u0.size)]
    if _scipy_minimize is not None:
        res = _scipy_minimize(fun, u0.ravel(), method="SLSQP", bounds=flat_bounds,
                              options={"maxiter": cfg.iters, "ftol": 1e-10})
        return res.x.reshape(u0.shape), float(res.fun)

    x = u0.ravel().copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    lo = np.array([b[0] for b in flat_bounds])
    hi = np.array([b[1] for b in flat_bounds])
    span = np.maximum(hi - lo, 1e-12)
    best, best_cost = x.copy(), fun(x)
    for it in range(1, cfg.iters + 1):
        f0 = fun(x)
        grad = np.zeros_like(x)
        for i in range(x.size):
            step = cfg.fd_eps * span[i]
            xp = x.copy()
            xp[i] = min(x[i] + step, hi[i])
            actual = xp[i] - x[i]
            grad[i] = (fun(xp) - f0) / actual if actual != 0 else 0.0
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad**2
        mh = m / (1 - 0.9**it)
        vh = v / (1 - 0.999**it)
        x = np.clip(x - cfg.lr * span * mh / (np.sqrt(vh) + 1e-8), lo, hi)
        c = fun(x)
        if c < best_cost:
            best, best_cost = x.copy(), c
    return best.reshape(u0.shape), float(best_cost)


# ---------------------------------------------------------------------------
# Controladores
# ---------------------------------------------------------------------------

class Controller:
    name = "base"

    def act(self, state: MacroState, plant: MacroPlant) -> ControlResult:
        raise NotImplementedError


class ConstantOmega(Controller):
    """Baseline 1: omega fijo. El punto de comparación mínimo."""

    name = "omega_constante"

    def __init__(self, omega: float, feed: float):
        self.u = np.array([omega, feed])

    def act(self, state, plant):
        u = np.tile(self.u, (plant.cfg.horizon, 1))
        trace, cost, _ = plant.rollout(state, u, plant.regime(self.u[0]))
        return ControlResult(u=u, cost=cost, predicted=trace,
                             meta={"controller": self.name})


class HoldupPI(Controller):
    """Baseline 2: PI sobre el hold-up actuando sobre la alimentación."""

    name = "pi_holdup"

    def __init__(self, setpoint: float, omega: float, kp: float = 0.02,
                 ki: float = 0.005, feed_bounds=(0.0, 0.05)):
        self.setpoint = setpoint
        self.omega = omega
        self.kp, self.ki = kp, ki
        self.bounds = feed_bounds
        self._integral = 0.0

    def act(self, state, plant):
        err = self.setpoint - state.holdup
        self._integral += err * plant.cfg.dt
        feed = np.clip(self.kp * err + self.ki * self._integral, *self.bounds)
        u = np.tile([self.omega, feed], (plant.cfg.horizon, 1))
        trace, cost, _ = plant.rollout(state, u, plant.regime(self.omega))
        return ControlResult(u=u, cost=cost, predicted=trace,
                             meta={"controller": self.name, "error": err})


class MPC(Controller):
    """MPC restringido sobre `u = [omega, F_feed]`."""

    name = "mpc"

    def __init__(self, cfg: MPCConfig, monitor: ConfidenceMonitor | None = None,
                 policy: RefreshPolicy | None = None):
        self.cfg = cfg
        self.monitor = monitor
        self.policy = policy or make_policy("library")
        self._warm = None
        self._frozen: Regime | None = None
        self._k = 0

    def act(self, state: MacroState, plant: MacroPlant) -> ControlResult:
        cfg = self.cfg
        omega_bounds = (cfg.omega_min, cfg.omega_max)
        d_omega = cfg.d_omega_max

        u_prev = self._warm[0] if self._warm is not None else np.array(
            [0.5 * (cfg.omega_min + cfg.omega_max), 0.5 * (cfg.feed_min + cfg.feed_max)]
        )
        report = None
        if self.monitor is not None:
            regime_now = plant.regime(float(u_prev[0]))
            report = self.monitor.evaluate(regime_now, state_com_r=regime_now.com_r)
            omega_bounds, d_omega = self.monitor.restrict(omega_bounds, d_omega, report)

        if self.policy.should_refresh(self._k, state, u_prev) or self._frozen is None:
            self._frozen = plant.regime(float(u_prev[0]))
        self._k += 1

        # [2, 2]: una fila por variable de control, que difunde contra u [H, 2]
        bounds = np.array([[omega_bounds[0], omega_bounds[1]],
                           [cfg.feed_min, cfg.feed_max]])

        if self._warm is not None:
            u0 = np.vstack([self._warm[1:], self._warm[-1:]])
        else:
            u0 = np.tile([np.mean(omega_bounds), np.mean([cfg.feed_min, cfg.feed_max])],
                         (cfg.horizon, 1))
        u0 = _project(u0, bounds)

        local = replace(cfg, d_omega_max=d_omega)
        plant_local = MacroPlant(plant.pbm, plant.library, local)
        plant_local._cache = plant._cache

        def objective(flat):
            u = _project(flat.reshape(cfg.horizon, 2), bounds)
            return plant_local.rollout(state, u, self._frozen)[1]

        u_opt, cost = _minimize(objective, u0, bounds, cfg)
        u_opt = _project(u_opt, bounds)
        self._warm = u_opt
        trace, cost, _ = plant_local.rollout(state, u_opt, self._frozen)
        return ControlResult(
            u=u_opt, cost=cost, predicted=trace, confidence=report,
            meta={"controller": self.name, "spectrum_mode": cfg.spectrum_mode,
                  "omega_bounds": omega_bounds, "d_omega_max": d_omega,
                  "solver": "slsqp" if _scipy_minimize else "adam_proyectado",
                  "omega_has_micro_support": True,
                  "feed_has_micro_support": False},
        )


# ---------------------------------------------------------------------------
# Lazo cerrado
# ---------------------------------------------------------------------------

def closed_loop(controller: Controller, plant: MacroPlant, state: MacroState,
                n_steps: int, *, truth: MacroPlant | None = None) -> dict:
    """Corre el lazo. `truth` permite que el controlador planifique con un
    modelo y la 'planta' avance con otro — así se mide el costo de que el
    espectro del controlador sea peor que el real, que es el objeto de E1."""
    sim = truth or plant
    history, total_cost, total_product, total_energy = [], 0.0, 0.0, 0.0
    z = state.copy()
    for k in range(n_steps):
        result = controller.act(z, plant)
        u0 = result.u[0]
        reg = sim.regime(float(u0[0]))
        reg = apply_feedback(reg, z, sim.pbm.cfg, enabled=sim.cfg.psd_feedback,
                             exponent=sim.cfg.psd_exponent)
        S = selection_rates(reg.spectrum, sim.pbm.cfg)
        z = sim.pbm.step(z, S, float(u0[1]), sim.cfg.dt)
        q = sim.pbm.product_rate(z)
        p_in = reg.p_impact
        total_product += q * sim.cfg.dt
        total_energy += p_in * sim.cfg.dt
        stage = (-plant.cfg.lambda_q * q
                 + plant.cfg.lambda_e * (p_in / q if q > 1e-12 else plant.cfg.penalty))
        total_cost += stage
        history.append({
            "k": k, "omega": float(u0[0]), "feed": float(u0[1]), "p_in": p_in,
            "product": q, "holdup": z.holdup, "p80": sim.pbm.p80(z),
            "stage_cost": stage,
            "ood": bool(result.confidence.triggered) if result.confidence else False,
        })
    return {
        "history": history,
        "total_cost": total_cost,
        "product_kg": total_product,
        "energy_J": total_energy,
        "specific_energy_J_per_kg": total_energy / max(total_product, 1e-12),
        "final_state": z.M.tolist(),
        "final_p80": sim.pbm.p80(z),
        "controller": controller.name,
        "spectrum_mode": plant.cfg.spectrum_mode,
    }
