"""Biblioteca de regímenes `E^coll(omega)` (§7.3).

Dado el costo del doble backward en CPU, ésta es la forma **primaria** de
acoplamiento micro-macro del banco de pruebas, no un control de comparación: el
lazo de control consulta una tabla, nunca el núcleo granular.

La rampa de omega del archivo de 2073 esferas recorre el mismo rango de
velocidades dos veces, subiendo y bajando. Eso da dos ramas por nodo, y la
discrepancia entre ellas es exactamente la incertidumbre epistémica del
espectro (§7.7) — incertidumbre calibrada sin ensambles.
"""

from dataclasses import dataclass, field, replace

import numpy as np

from .coarse import MacroFeatures, Spectrum
from .harness import wasserstein

UP, DOWN, FLAT = "up", "down", "flat"


@dataclass
class Regime:
    """Respuesta de una consulta a la biblioteca."""

    omega: float
    spectrum: Spectrum
    p_in: float                # W, balance energético (DERIVADO, puede ser <0)
    p_impact: float            # W, potencia de impacto Σ E·n·masa (MEDIDA, >0)
    ke: float                  # J
    com_r: float               # m
    sigma_spec: float          # discrepancia entre ramas [décadas de Wasserstein]
    sigma_rate: float          # discrepancia relativa de la tasa total
    ood: bool                  # omega fuera del rango cubierto
    distance_to_node: float    # rad/s al nodo más cercano
    meta: dict = field(default_factory=dict)


def branch_of(omega_series: np.ndarray, lo: int, hi: int, tol: float = 1e-9) -> str:
    """Clasifica una ventana como rama ascendente, descendente o plana."""
    seg = omega_series[lo:hi]
    if seg.size < 2:
        return FLAT
    slope = float(np.polyfit(np.arange(seg.size), seg, 1)[0])
    span = float(seg.max() - seg.min())
    if span < tol or abs(slope) * seg.size < tol:
        return FLAT
    return UP if slope > 0 else DOWN


class SpectrumLibrary:
    """Tabla `omega → espectro` con interpolación lineal en log-tasa.

    La extrapolación está **prohibida**: fuera del rango cubierto por la rampa
    la consulta devuelve el nodo del extremo con `ood=True`, y el MPC está
    cableado para restringirse (§7.7). Un espectro extrapolado es exactamente el
    tipo de andamio silencioso que este banco de pruebas existe para evitar.
    """

    def __init__(self, min_events: int = 1):
        self.min_events = min_events
        self.nodes: list[MacroFeatures] = []
        self._omega = np.zeros(0)
        self._log_rates = np.zeros((0, 0, 0))
        self._scalars: dict[str, np.ndarray] = {}
        self._edges = None

    # -- construcción --------------------------------------------------------

    def fit(self, features: list[MacroFeatures]) -> "SpectrumLibrary":
        usable = [f for f in features if f.n_events >= self.min_events]
        if not usable:
            raise ValueError("ninguna ventana tiene eventos suficientes")
        edges = usable[0].spectrum.edges
        for f in usable:
            if not np.allclose(f.spectrum.edges, edges):
                raise ValueError("todas las ventanas deben compartir el binning")
        order = np.argsort([f.omega for f in usable], kind="stable")
        self.nodes = [usable[i] for i in order]
        self._edges = edges
        self._omega = np.array([f.omega for f in self.nodes])
        # log-tasa: la interpolación lineal en log preserva la positividad y
        # trata los órdenes de magnitud, que es la escala real del espectro
        rates = np.stack([f.spectrum.rates for f in self.nodes])   # [K, bins, tipos]
        self._log_rates = np.log(np.maximum(rates, 1e-300))
        self._scalars = {
            "p_in": np.array([f.p_in_balance for f in self.nodes]),
            "p_impact": np.array([f.spectrum.energy_rate() * f.spectrum.mass
                                  for f in self.nodes]),
            "ke": np.array([f.ke for f in self.nodes]),
            "com_r": np.array([f.com_r for f in self.nodes]),
            "duration": np.array([f.duration for f in self.nodes]),
            "mass": np.array([f.spectrum.mass for f in self.nodes]),
        }
        self._branches = np.array([f.branch for f in self.nodes])
        return self

    @property
    def omega_range(self) -> tuple[float, float]:
        return float(self._omega.min()), float(self._omega.max())

    def __len__(self) -> int:
        return len(self.nodes)

    # -- consulta ------------------------------------------------------------

    def _interp_log_rates(self, omega: float, mask=None) -> np.ndarray:
        om = self._omega if mask is None else self._omega[mask]
        lr = self._log_rates if mask is None else self._log_rates[mask]
        if om.size == 0:
            return np.full(self._log_rates.shape[1:], -np.inf)
        if om.size == 1:
            return lr[0]
        o = np.clip(omega, om.min(), om.max())
        k = int(np.clip(np.searchsorted(om, o) - 1, 0, om.size - 2))
        span = om[k + 1] - om[k]
        w = 0.0 if span <= 0 else (o - om[k]) / span
        return (1 - w) * lr[k] + w * lr[k + 1]

    def _interp_scalar(self, name: str, omega: float) -> float:
        return float(np.interp(np.clip(omega, *self.omega_range),
                               self._omega, self._scalars[name]))

    def query(self, omega: float) -> Regime:
        """Espectro interpolado más su medida de incertidumbre.

        La incertidumbre combina dos fuentes, ambas baratas: la dispersión entre
        la rama ascendente y la descendente a esa omega (histéresis medida), y
        la distancia al nodo más cercano.
        """
        if not self.nodes:
            raise RuntimeError("biblioteca vacía: llamar a fit() primero")
        lo, hi = self.omega_range
        ood = omega < lo or omega > hi
        rates = np.exp(self._interp_log_rates(omega))
        duration = self._interp_scalar("duration", omega)
        mass = self._interp_scalar("mass", omega)
        spectrum = Spectrum(
            edges=self._edges, rates=rates, duration=duration, mass=mass,
            underflow=np.zeros(rates.shape[1]), overflow=np.zeros(rates.shape[1]),
        )
        sigma_spec, sigma_rate = self._branch_discrepancy(omega, spectrum)
        return Regime(
            omega=float(omega), spectrum=spectrum,
            p_in=self._interp_scalar("p_in", omega),
            p_impact=self._interp_scalar("p_impact", omega),
            ke=self._interp_scalar("ke", omega),
            com_r=self._interp_scalar("com_r", omega),
            sigma_spec=sigma_spec, sigma_rate=sigma_rate, ood=ood,
            distance_to_node=float(np.min(np.abs(self._omega - omega))),
            meta={"n_nodes": len(self.nodes), "omega_range": (lo, hi)},
        )

    def _branch_discrepancy(self, omega: float, pooled: Spectrum):
        """Discrepancia entre ramas a la misma omega — la histéresis de §3."""
        up, down = self._branches == UP, self._branches == DOWN
        if up.sum() < 2 or down.sum() < 2:
            return float("nan"), float("nan")
        common_lo = max(self._omega[up].min(), self._omega[down].min())
        common_hi = min(self._omega[up].max(), self._omega[down].max())
        if not (common_lo <= omega <= common_hi):
            return float("nan"), float("nan")
        specs = []
        for mask in (up, down):
            rates = np.exp(self._interp_log_rates(omega, mask))
            specs.append(replace(pooled, rates=rates))
        return wasserstein(specs[0], specs[1]), _rel_gap(
            specs[0].total_rate(), specs[1].total_rate()
        )

    # -- diagnóstico ---------------------------------------------------------

    def branch_table(self) -> list[dict]:
        return [
            {"omega": f.omega, "branch": f.branch, "n_events": f.n_events,
             "total_rate": f.spectrum.total_rate(),
             "mean_energy": f.spectrum.mean_energy(),
             "p_in": f.p_in_balance}
            for f in self.nodes
        ]


def _rel_gap(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-30)
    return float(abs(a - b) / denom)


def hysteresis_verdict(report: dict, w_threshold: float, rate_threshold: float) -> dict:
    """Veredicto **resuelto en omega**, no por mediana.

    Una mediana puede quedar bajo el umbral mientras un extremo del rango está
    fuera por un orden de magnitud, y ese extremo es justamente el régimen donde
    el acoplamiento por tabla fallaría. El veredicto agregado solo declara
    "es una función" si **ningún** punto de sondeo supera el umbral, y en caso
    contrario reporta hasta qué omega llega la dependencia del camino.
    """
    if report.get("status") != "ok":
        return {"verdict": "indeterminado", "reason": report.get("status")}

    probes = report["probes"]
    bad = [p for p in probes
           if p["wasserstein_decades"] > w_threshold
           or p["rate_rel_error"] > rate_threshold]
    if not bad:
        return {"verdict": "es una función", "n_probes": len(probes),
                "n_violating": 0,
                "max_wasserstein": max(p["wasserstein_decades"] for p in probes),
                "max_rate_error": max(p["rate_rel_error"] for p in probes)}

    omegas_bad = [p["omega"] for p in bad]
    return {
        "verdict": "depende del camino",
        "n_probes": len(probes),
        "n_violating": len(bad),
        "omega_violating_range": (float(min(omegas_bad)), float(max(omegas_bad))),
        "max_wasserstein": max(p["wasserstein_decades"] for p in probes),
        "max_rate_error": max(p["rate_rel_error"] for p in probes),
        "worst_probe": max(bad, key=lambda p: p["wasserstein_decades"]),
    }


def hysteresis_report(features: list[MacroFeatures], n_probe: int = 12) -> dict:
    """Experimento H (§3): ¿`omega ↦ E^coll` es una función o depende del camino?

    Compara las ramas ascendente y descendente a la misma omega, con distancia
    de Wasserstein sobre el espectro y error relativo de la tasa total. No
    decide por el usuario: devuelve las cifras y el rango solapado.
    """
    up = [f for f in features if f.branch == UP]
    down = [f for f in features if f.branch == DOWN]
    if len(up) < 2 or len(down) < 2:
        return {"status": "insuficiente",
                "n_up": len(up), "n_down": len(down)}

    lib_up = SpectrumLibrary().fit(up)
    lib_down = SpectrumLibrary().fit(down)
    lo = max(lib_up.omega_range[0], lib_down.omega_range[0])
    hi = min(lib_up.omega_range[1], lib_down.omega_range[1])
    if hi <= lo:
        return {"status": "sin solape", "up": lib_up.omega_range,
                "down": lib_down.omega_range}

    probes = np.linspace(lo, hi, n_probe)
    rows = []
    for om in probes:
        a, b = lib_up.query(om).spectrum, lib_down.query(om).spectrum
        rows.append({
            "omega": float(om),
            "wasserstein_decades": wasserstein(a, b),
            "rate_rel_error": _rel_gap(a.total_rate(), b.total_rate()),
            "mean_energy_rel_error": _rel_gap(a.mean_energy(), b.mean_energy()),
            "rate_up": a.total_rate(), "rate_down": b.total_rate(),
        })
    return {
        "status": "ok",
        "omega_overlap": (float(lo), float(hi)),
        "n_up": len(up), "n_down": len(down),
        "probes": rows,
        "median_wasserstein": float(np.median([r["wasserstein_decades"] for r in rows])),
        "median_rate_error": float(np.median([r["rate_rel_error"] for r in rows])),
        "max_rate_error": float(np.max([r["rate_rel_error"] for r in rows])),
    }
