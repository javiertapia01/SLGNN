"""PBM energético mínimo, tipo Hinde–Kalala (§7.4).

Seis clases log-espaciadas de razón 2, con tasas de selección informadas por el
**espectro** y no solo por la energía total. Esa distinción es el puente
micro-macro entero: si `S_b` fuera función únicamente de la potencia agregada,
el PBM sería ciego a `C_phi` y todos los experimentos posteriores darían falsos
negativos. `sensitivity_to_shape` existe para no dejar eso a la fe.

Los parámetros son **plausibles, no calibrados**: las PSD que produce este
módulo no son predicciones.

Convención de índices: `b = 0` es la clase más gruesa, `b = B-1` la más fina.
La progenie va siempre de `j` a `b > j`.
"""

from dataclasses import dataclass, field

import numpy as np

from .coarse import Spectrum


@dataclass
class PBMConfig:
    n_classes: int = 6
    d_max: float = 0.005        # m, borde superior de la clase más gruesa
    ratio: float = 2.0          # razón de tamaño entre clases consecutivas
    density: float = 4000.0     # kg/m³ (dataset)

    # --- selección: p_b(E) = 1 - exp[-(E/E*_b)^nu],  E*_b = E*_ref (d_b/d_ref)^alpha
    e_star_ref: float = 2.0e-6  # J, energía característica de rotura a d_ref
    d_ref: float = 0.005        # m
    nu: float = 1.0             # exponente de la ley de daño
    alpha: float = 2.0          # escalamiento de E* con el tamaño

    # --- progenie: Gaudin–Schuhmann truncada, un parámetro
    beta: float = 0.8

    # --- descarga por clasificación sigmoide sobre el tamaño
    k_discharge: float = 0.5    # 1/s
    d50: float = 1.0e-3         # m
    classifier_width: float = 0.15   # décadas de tamaño

    # --- alimentación
    feed_fractions: tuple = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # --- objetivo económico, en unidades del tambor (§6): masa bajo tamaño
    #     objetivo por unidad de energía suministrada
    d_target: float = 1.0e-3    # m

    def sizes(self) -> np.ndarray:
        """Tamaño representativo (media geométrica del intervalo) por clase."""
        upper = self.upper_edges()
        lower = upper / self.ratio
        return np.sqrt(upper * lower)

    def upper_edges(self) -> np.ndarray:
        return self.d_max * self.ratio ** (-np.arange(self.n_classes))

    def particle_masses(self) -> np.ndarray:
        return self.density * (np.pi / 6.0) * self.sizes() ** 3


# ---------------------------------------------------------------------------
# Núcleo del modelo
# ---------------------------------------------------------------------------

def breakage_probability(energies: np.ndarray, cfg: PBMConfig) -> np.ndarray:
    """`p_b(E)` para cada clase y cada bin de energía. Devuelve [B, n_bins]."""
    e_star = cfg.e_star_ref * (cfg.sizes() / cfg.d_ref) ** cfg.alpha
    x = energies[None, :] / e_star[:, None]
    return 1.0 - np.exp(-np.power(np.maximum(x, 0.0), cfg.nu))


def selection_rates(spectrum: Spectrum, cfg: PBMConfig) -> np.ndarray:
    """`S_b = m_b · Σ_bins n(E)·p_b(E)` en 1/s.

    `n(E)` viene de `C_phi` en eventos/(s·kg); multiplicar por la masa de una
    partícula de la clase convierte una tasa por unidad de masa en una tasa de
    impactos **por partícula**, que es la que consume la ley de daño. Es también
    lo que hace que las clases gruesas se seleccionen más rápido, sin necesidad
    de un parámetro extra.
    """
    n = spectrum.rates.sum(axis=1)                       # [n_bins]
    p = breakage_probability(spectrum.centers, cfg)      # [B, n_bins]
    return cfg.particle_masses() * (p * n[None, :]).sum(axis=1)


def progeny_matrix(cfg: PBMConfig) -> np.ndarray:
    """`b[b, j]`: fracción de la masa rota de `j` que aparece en `b`.

    Gaudin–Schuhmann truncada: P(x) = (x/u_j)^beta. La clase más fina absorbe el
    resto, y cada columna se normaliza exactamente a 1 para que la conservación
    de masa se cumpla a precisión de máquina y no aproximadamente.
    """
    B = cfg.n_classes
    upper = cfg.upper_edges()
    lower = upper / cfg.ratio
    mat = np.zeros((B, B))
    for j in range(B - 1):
        u = upper[j]
        hi = np.minimum(upper[j + 1:], u) / u
        lo = np.minimum(lower[j + 1:], u) / u
        frac = hi**cfg.beta - lo**cfg.beta
        frac[-1] += lo[-1] ** cfg.beta      # la clase más fina es el sumidero
        mat[j + 1:, j] = frac / frac.sum()
    return mat


def discharge_rates(cfg: PBMConfig) -> np.ndarray:
    """Clasificación sigmoide: lo fino descarga, lo grueso se retiene."""
    z = (np.log10(cfg.d50) - np.log10(cfg.sizes())) / cfg.classifier_width
    return cfg.k_discharge / (1.0 + np.exp(-z))


@dataclass
class MacroState:
    """`z = [M_1..M_B, H]` (§1). `H` es derivada, no un grado de libertad extra."""

    M: np.ndarray               # kg por clase

    @property
    def holdup(self) -> float:
        return float(self.M.sum())

    def as_vector(self) -> np.ndarray:
        return np.concatenate([self.M, [self.holdup]])

    def copy(self) -> "MacroState":
        return MacroState(M=self.M.copy())


@dataclass
class PBM:
    cfg: PBMConfig = field(default_factory=PBMConfig)

    def __post_init__(self):
        self._progeny = progeny_matrix(self.cfg)
        self._discharge = discharge_rates(self.cfg)
        self._feed = np.asarray(self.cfg.feed_fractions, dtype=float)
        if self._feed.size != self.cfg.n_classes:
            raise ValueError("feed_fractions debe tener n_classes entradas")
        s = self._feed.sum()
        if s > 0:
            self._feed = self._feed / s

    # -- dinámica ------------------------------------------------------------

    def rhs(self, M: np.ndarray, S: np.ndarray, f_feed: float) -> np.ndarray:
        death = S * M
        birth = self._progeny @ death
        return birth - death + f_feed * self._feed - self._discharge * M

    def step(self, state: MacroState, S: np.ndarray, f_feed: float,
             dt: float, substeps: int = 4) -> MacroState:
        """RK4 con submuestreo. El esquema es lineal en `M`, así que conserva la
        masa a precisión de máquina cuando no hay alimentación ni descarga."""
        M = state.M.astype(float)
        h = dt / substeps
        for _ in range(substeps):
            k1 = self.rhs(M, S, f_feed)
            k2 = self.rhs(M + 0.5 * h * k1, S, f_feed)
            k3 = self.rhs(M + 0.5 * h * k2, S, f_feed)
            k4 = self.rhs(M + h * k3, S, f_feed)
            M = M + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return MacroState(M=np.maximum(M, 0.0))

    # -- observables ---------------------------------------------------------

    def product_rate(self, state: MacroState) -> float:
        """Caudal descargado bajo el tamaño objetivo [kg/s]."""
        under = self.cfg.sizes() < self.cfg.d_target
        return float((self._discharge * state.M * under).sum())

    def discharge_rate(self, state: MacroState) -> float:
        return float((self._discharge * state.M).sum())

    def p80(self, state: MacroState) -> float:
        """P80 sobre la distribución acumulada de masa retenida [m]."""
        M = state.M
        total = M.sum()
        if total <= 0:
            return float("nan")
        # acumulado pasante, de fino a grueso
        sizes = self.cfg.sizes()[::-1]
        cum = np.cumsum(M[::-1]) / total
        if cum[-1] < 0.8:
            return float(sizes[-1])
        return float(np.interp(0.8, cum, sizes))


# ---------------------------------------------------------------------------
# Chequeo previo indispensable (§7.4)
# ---------------------------------------------------------------------------

def sensitivity_to_shape(cfg: PBMConfig, template: Spectrum) -> dict:
    """¿`S_b` responde a la FORMA del espectro o solo a su energía total?

    Construye dos espectros con **idéntica energía total** —uno concentrado en
    energías bajas, otro con cola alta— y compara las `S_b` resultantes. Si el
    cociente es cercano a 1, el PBM es ciego al puente micro-macro y todo
    experimento posterior daría un falso negativo. Es el modo de falla más
    traicionero del banco de pruebas, así que se mide antes de usarlo.
    """
    centers = template.centers
    n_bins = centers.size
    soft = np.zeros(n_bins)
    hard = np.zeros(n_bins)
    lo, hi = n_bins // 4, 3 * n_bins // 4
    soft[lo] = 1.0
    hard[hi] = 1.0
    # igualar la energía total transportada: Σ E·n
    soft *= 1.0 / centers[lo]
    hard *= 1.0 / centers[hi]

    def _spec(vec):
        rates = np.zeros_like(template.rates)
        rates[:, 0] = vec
        return template.with_rates(rates)

    s_soft = selection_rates(_spec(soft), cfg)
    s_hard = selection_rates(_spec(hard), cfg)
    ratio = s_hard / np.maximum(s_soft, 1e-300)
    return {
        "energy_soft": float((centers * soft).sum()),
        "energy_hard": float((centers * hard).sum()),
        "S_soft": s_soft, "S_hard": s_hard,
        "ratio_per_class": ratio,
        "max_ratio": float(np.max(ratio)),
        "min_ratio": float(np.min(ratio)),
        "shape_sensitive": bool(np.max(ratio) > 2.0 or np.min(ratio) < 0.5),
    }
