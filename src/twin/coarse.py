"""`C_phi` — el operador de coarse-graining (§7.2).

Contrato central del banco de pruebas: **una única implementación**, que recibe
trayectorias de MFiX o de un rollout de SLGNN indistintamente y no sabe cuál de
las dos le tocó. Esa indiferencia es lo que hace que T3 sustituya la fuente sin
tocar el operador.

Frontera de unidades (§6): la entrada es **adimensional**, la salida es **SI**.
Ninguna otra frontera del paquete convierte unidades.
"""

import math
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from slgnn.sdf import RotatingCylinderSDF

from .events import PP, PW, EventTable, close_pairs_frame, detect_events
from .units import Scaling

_TYPES = ("pp", "pw")


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

@dataclass
class CoarseConfig:
    """Parámetros de `C_phi`. Todo lo que no sea una escala vive aquí."""

    # detección de eventos (unidades adimensionales)
    delta: float = 0.0
    max_lookback: int = 32
    v_n_min: float = 0.0
    block: int = 256

    # binning del espectro (SI, joules)
    n_bins: int = 30
    e_min: float = 1e-12
    e_max: float = 1e-2

    # coeficientes de restitución de los metadatos del dataset (DATA_NOTES.md)
    restitution_pp: float = 0.95
    restitution_pw: float = 0.90

    # Factor de calibración de la disipación derivada (ver `calibrate_dissipation`).
    # 1.0 = estimador crudo (1-e²)·E_impacto, que en este dataset recupera solo
    # ~1/3 de la pérdida real de energía mecánica.
    kappa_diss: float = 1.0

    # cierre energético
    closure_stride: int = 5

    def bin_edges(self) -> np.ndarray:
        return np.geomspace(self.e_min, self.e_max, self.n_bins + 1)

    def restitution(self, kind: int) -> float:
        return self.restitution_pp if kind == PP else self.restitution_pw


# ---------------------------------------------------------------------------
# Espectro
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Spectrum:
    """Tasa de eventos por unidad de tiempo y masa, por bin de energía y canal.

    `edges` en joules, `rates` en eventos/(s·kg). El *underflow* y el
    *overflow* se reportan explícitamente en vez de descartarse en silencio:
    un espectro con overflow alto significa que `e_max` está mal elegido y las
    tasas de rotura del PBM estarán truncadas.
    """

    edges: np.ndarray          # [n_bins+1] J
    rates: np.ndarray          # [n_bins, n_types] eventos/(s·kg)
    duration: float            # s
    mass: float                # kg
    underflow: np.ndarray      # [n_types] eventos/(s·kg)
    overflow: np.ndarray       # [n_types] eventos/(s·kg)
    types: tuple = _TYPES

    @property
    def centers(self) -> np.ndarray:
        return np.sqrt(self.edges[:-1] * self.edges[1:])

    @property
    def counts(self) -> np.ndarray:
        return self.rates * self.duration * self.mass

    def total_rate(self, kind=None) -> float:
        r = self.rates if kind is None else self.rates[:, self.types.index(kind)]
        return float(r.sum())

    def energy_rate(self, kind=None) -> float:
        """Potencia de impacto por unidad de masa [W/kg]: Σ E·n(E)."""
        r = self.rates if kind is None else self.rates[:, [self.types.index(kind)]]
        return float((self.centers[:, None] * np.atleast_2d(r.reshape(len(self.centers), -1))).sum())

    def mean_energy(self) -> float:
        total = self.total_rate()
        if total <= 0:
            return 0.0
        return self.energy_rate() / total

    def normalized(self) -> np.ndarray:
        """Densidad de probabilidad discreta sobre bins, sumando ambos canales."""
        p = self.rates.sum(axis=1)
        s = p.sum()
        return p / s if s > 0 else p

    def with_rates(self, rates: np.ndarray) -> "Spectrum":
        return replace(self, rates=np.asarray(rates, dtype=float))


def _empty_spectrum(cfg: CoarseConfig, duration: float, mass: float) -> Spectrum:
    return Spectrum(
        edges=cfg.bin_edges(), rates=np.zeros((cfg.n_bins, len(_TYPES))),
        duration=duration, mass=mass,
        underflow=np.zeros(len(_TYPES)), overflow=np.zeros(len(_TYPES)),
    )


def build_spectrum(energies_si: np.ndarray, kinds: np.ndarray, cfg: CoarseConfig,
                   duration: float, mass: float) -> Spectrum:
    edges = cfg.bin_edges()
    rates = np.zeros((cfg.n_bins, len(_TYPES)))
    under = np.zeros(len(_TYPES))
    over = np.zeros(len(_TYPES))
    norm = duration * mass
    for col, kind in enumerate((PP, PW)):
        e = energies_si[kinds == kind]
        if e.size == 0:
            continue
        under[col] = float((e < edges[0]).sum()) / norm
        over[col] = float((e >= edges[-1]).sum()) / norm
        inside = e[(e >= edges[0]) & (e < edges[-1])]
        if inside.size:
            rates[:, col] = np.histogram(inside, bins=edges)[0] / norm
    return Spectrum(edges=edges, rates=rates, duration=duration, mass=mass,
                    underflow=under, overflow=over)


# ---------------------------------------------------------------------------
# Descriptores macroscópicos
# ---------------------------------------------------------------------------

@dataclass
class MacroFeatures:
    """Salida de `C_phi`, en unidades SI."""

    spectrum: Spectrum
    omega: float               # rad/s, velocidad angular media de la pared
    ke: float                  # J, energía cinética media (traslación + rotación)
    pe: float                  # J, energía potencial media
    p_diss: float              # W, DERIVADA vía kappa·(1-e²)·E_impacto, no medida
    p_mech: float              # W, d(KE+PE)/dt medido directamente
    p_in_balance: float        # W, p_mech + p_diss
    p_in_wall: float           # W, Σ F_pared·v_pared (subconjunto identificable)
    closure_gap: float         # residual del balance, normalizado por su escala
    fill: float                # fracción volumétrica de sólidos en el contenedor
    com: np.ndarray            # m, centro de masa
    com_r: float               # m, radio del centro de masa respecto del eje
    com_theta: float           # rad, ángulo polar del centro de masa
    theta_toe: float           # rad
    theta_shoulder: float      # rad
    t_start: float             # s
    t_end: float               # s
    n_events: int
    n_unresolved: int
    branch: str = "unknown"    # "up" | "down" | "flat" — rama de la rampa de omega
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    @property
    def energy_efficiency(self) -> float:
        """Potencia de impacto entregada por watt de entrada. Adimensional."""
        p_impact = self.spectrum.energy_rate() * self.spectrum.mass
        return p_impact / self.p_in_balance if self.p_in_balance > 0 else float("nan")


# ---------------------------------------------------------------------------
# Núcleo
# ---------------------------------------------------------------------------

def _kinetic_energy(v, w, m, radii):
    """Traslación + rotación. I = 2/5 m R² para esferas macizas."""
    ke_t = 0.5 * (m[None, :, None] * v**2).sum(axis=(1, 2))
    inertia = 0.4 * m * radii**2
    ke_r = 0.5 * (inertia[None, :, None] * w**2).sum(axis=(1, 2))
    return ke_t + ke_r


def _wall_omega(sdf, t_nondim: float, scaling: Scaling) -> float:
    if isinstance(sdf, RotatingCylinderSDF):
        return float(sdf.omega_fn(t_nondim)) / scaling.T
    return 0.0


def _load_descriptors(q, m, sdf, g_hat):
    """Centro de masa y extensión angular del lecho.

    `theta` se mide desde la dirección de la gravedad, en el plano perpendicular
    al eje del cilindro. Sin cilindro, los ángulos son NaN: no están definidos.
    """
    total_m = m.sum()
    com = (m[None, :, None] * q).sum(axis=1).mean(axis=0) / total_m
    if not isinstance(sdf, RotatingCylinderSDF):
        return com, float("nan"), float("nan"), float("nan"), float("nan")

    center = np.array([sdf.cx, sdf.cy, 0.0])
    rel = q - center
    rel[..., 2] = 0.0
    # base local: g_hat (hacia el pie del lecho) y su perpendicular en el plano
    axis = np.array([0.0, 0.0, 1.0])
    e_down = g_hat - (g_hat @ axis) * axis
    e_down = e_down / (np.linalg.norm(e_down) + 1e-30)
    e_perp = np.cross(axis, e_down)
    theta = np.arctan2(rel @ e_perp, rel @ e_down)

    r = np.linalg.norm(rel, axis=-1)
    near_wall = r > 0.85 * sdf.radius
    com_rel = com - center
    com_rel[2] = 0.0
    com_r = float(np.linalg.norm(com_rel))
    com_theta = float(np.arctan2(com_rel @ e_perp, com_rel @ e_down))
    if not near_wall.any():
        return com, com_r, com_theta, float("nan"), float("nan")
    th = theta[near_wall]
    # pie y hombro: percentiles robustos a partículas sueltas en vuelo
    return com, com_r, com_theta, float(np.percentile(th, 2)), float(np.percentile(th, 98))


def _wall_input_power(q, v, radii, m, sdf, g_vec, dt, t0, stride, eps=1e-8):
    """Ruta directa del cierre energético: Σ F_pared,i · v_pared,i.

    `F_pared,i` solo es identificable desde trayectorias para partículas cuyo
    **único** contacto es la pared: ahí F = m(a − g). En un lecho denso ese
    subconjunto es una fracción de las partículas en contacto con la pared, de
    modo que esta ruta es una **cota inferior** de la potencia de entrada, no su
    valor. Se reporta como tal en `closure_gap`.
    """
    T, N, _ = q.shape
    if T < 3:
        return float("nan"), 0.0
    qt = torch.from_numpy(q)
    total, n_samples, frac_used = 0.0, 0, []

    for k in range(1, T - 1, max(stride, 1)):
        with torch.no_grad():
            phi = sdf.phi(qt[k], t0 + k * dt).numpy()
        touching_wall = (phi - radii) < 0.0
        if not touching_wall.any():
            n_samples += 1
            frac_used.append(0.0)
            continue
        has_pp = np.zeros(N, dtype=bool)
        pairs = close_pairs_frame(q[k], radii, 0.0)
        if pairs.shape[0]:
            has_pp[pairs[:, 0]] = True
            has_pp[pairs[:, 1]] = True
        sel = np.nonzero(touching_wall & ~has_pp)[0]
        frac_used.append(float(sel.size) / max(int(touching_wall.sum()), 1))
        n_samples += 1
        if sel.size == 0:
            continue
        a = (v[k + 1, sel] - v[k - 1, sel]) / (2.0 * dt)
        f_wall = m[sel, None] * (a - g_vec[None, :])
        _, _, _, v_w = sdf.query(qt[k][sel].clone(), t0 + k * dt, eps)
        total += float((f_wall * v_w.detach().numpy()).sum())

    if n_samples == 0:
        return float("nan"), 0.0
    return total / n_samples, float(np.mean(frac_used)) if frac_used else 0.0


def coarse_grain(traj, sdf, scaling: Scaling, cfg: CoarseConfig, *,
                 g_vec=None, events: EventTable | None = None,
                 window=None, t0: float = 0.0) -> MacroFeatures:
    """ÚNICA implementación de `C_phi`.

    `traj` es cualquier objeto con `q, v, omega, m, radii, dt` en unidades
    adimensionales — un `slgnn.data.Trajectory` de MFiX o el resultado de un
    rollout de SLGNN, sin distinción. Determinista.

    `events` permite reutilizar una detección ya hecha sobre la trayectoria
    completa (la parte cara); `window = (lo, hi)` selecciona el rango de
    snapshots. `t0` es el tiempo adimensional del snapshot 0 de `traj`.
    """
    q = traj.q.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(traj.q) else np.asarray(traj.q, float)
    v = traj.v.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(traj.v) else np.asarray(traj.v, float)
    w = traj.omega.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(traj.omega) else np.asarray(traj.omega, float)
    m = traj.m.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(traj.m) else np.asarray(traj.m, float)
    radii = traj.radii.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(traj.radii) else np.asarray(traj.radii, float)
    dt = float(traj.dt)

    if events is None:
        events = detect_events(q, v, radii, m, sdf, delta=cfg.delta,
                               max_lookback=cfg.max_lookback, v_n_min=cfg.v_n_min,
                               dt=dt, t0=t0, block=cfg.block)
    lo, hi = window if window is not None else (0, q.shape[0])
    ev = events.window(lo, hi)

    qs, vs, ws = q[lo:hi], v[lo:hi], w[lo:hi]
    duration = (hi - lo) * dt * scaling.T
    mass = float(m.sum()) * scaling.M

    # --- espectro (adimensional → SI) ---
    spectrum = build_spectrum(ev.E_impact * scaling.energy, ev.kind, cfg,
                              duration, mass)

    # --- energías ---
    ke_series = _kinetic_energy(vs, ws, m, radii)
    if g_vec is None:
        g_vec = np.zeros(3)
    g_vec = np.asarray(g_vec, dtype=np.float64)
    pe_series = -(m[None, :, None] * qs * g_vec[None, None, :]).sum(axis=(1, 2))
    ke = float(ke_series.mean()) * scaling.energy
    pe = float(pe_series.mean()) * scaling.energy

    # --- disipación: DERIVADA, no medida (§2) ---
    diss = np.zeros(len(ev))
    for kind in (PP, PW):
        sel = ev.kind == kind
        diss[sel] = (1.0 - cfg.restitution(kind) ** 2) * ev.E_impact[sel]
    p_diss = cfg.kappa_diss * float(diss.sum()) * scaling.energy / max(duration, 1e-30)

    # --- cierre energético (§7.2) ---
    if hi - lo >= 3:
        mech = (ke_series + pe_series) * scaling.energy
        # Pendiente por mínimos cuadrados, no diferencia de extremos. En un lecho
        # que chapotea la energía mecánica fluctúa mucho más de lo que deriva, y
        # (mech[-1] - mech[0]) mide sobre todo el ruido de los dos instantes
        # elegidos: sobre la fase en reposo del cilindro esa versión daba factores
        # de calibración de signo alternante (-10, -10, +1.7, +5.6).
        t_axis = np.arange(mech.size) * dt * scaling.T
        p_mech = float(np.polyfit(t_axis, mech, 1)[0])
    else:
        p_mech = 0.0
    p_in_balance = p_mech + p_diss
    p_wall_nd, frac_ident = _wall_input_power(
        qs, vs, radii, m, sdf, g_vec, dt, t0 + lo * dt, cfg.closure_stride
    )
    p_in_wall = p_wall_nd * scaling.power if np.isfinite(p_wall_nd) else float("nan")
    # El residual se normaliza por la escala de los flujos de energía del balance,
    # no por p_in: en una caja estática p_in es cero por construcción y cualquier
    # cociente contra él sería infinito aunque el balance cierre perfecto.
    scale = max(abs(p_mech), abs(p_diss), 1e-30)
    closure_gap = (abs(p_in_balance - p_in_wall) / scale
                   if np.isfinite(p_in_wall) else float("nan"))

    # --- descriptores de carga ---
    g_hat = g_vec / (np.linalg.norm(g_vec) + 1e-30) if np.linalg.norm(g_vec) > 0 else np.array([0.0, -1.0, 0.0])
    com_nd, com_r, com_theta, toe, shoulder = _load_descriptors(qs, m, sdf, g_hat)
    solids = float((4.0 / 3.0) * math.pi * (radii**3).sum())
    fill = solids / _container_volume(sdf) if _container_volume(sdf) else float("nan")

    t_mid = t0 + 0.5 * (lo + hi) * dt
    omega = _wall_omega(sdf, t_mid, scaling)

    return MacroFeatures(
        spectrum=spectrum, omega=omega, ke=ke, pe=pe, p_diss=p_diss,
        p_mech=p_mech, p_in_balance=p_in_balance, p_in_wall=p_in_wall,
        closure_gap=closure_gap,
        fill=fill, com=com_nd * scaling.L, com_r=com_r * scaling.L,
        com_theta=com_theta, theta_toe=toe, theta_shoulder=shoulder,
        t_start=(t0 + lo * dt) * scaling.T, t_end=(t0 + hi * dt) * scaling.T,
        n_events=len(ev), n_unresolved=events.n_unresolved,
        meta={"identifiable_wall_fraction": frac_ident,
              "n_particles": int(q.shape[1]),
              "scaling": scaling.to_dict()},
    )


def _container_volume(sdf) -> float:
    if isinstance(sdf, RotatingCylinderSDF):
        return math.pi * sdf.radius**2 * (sdf.z_max - sdf.z_min)
    xmin = getattr(sdf, "xmin", None)
    if xmin is not None:
        extent = (sdf.xmax - sdf.xmin).numpy()
        if np.all(np.isfinite(extent)) and extent.max() < 1e3:
            return float(np.prod(extent))
    return 0.0


def calibrate_dissipation(traj, sdf, scaling: Scaling, cfg: CoarseConfig, windows,
                          *, g_vec=None, t0: float = 0.0) -> dict:
    """Calibra `kappa_diss` sobre un sistema cerrado (pared estática, v_W = 0).

    Motivación, medida y no supuesta. En `60Spheres_Gravity` con la caja fija no
    entra energía: `p_in ≡ 0` por construcción, de modo que el balance exige
    `d(KE+PE)/dt + p_diss = 0`. Con el estimador crudo `(1-e²)·E_impacto` el
    balance **no cierra**: recupera del orden de un tercio de la pérdida real de
    energía mecánica, de forma sistemática (no ruidosa) a lo largo de las
    ventanas.

    La razón es estructural, no un bug: la energía de impacto normal ignora la
    disipación tangencial por fricción y la que ocurre en contactos sostenidos,
    que no producen eventos discretos. Como las instrucciones ya advierten (§2),
    la disipación **no es recuperable de trayectorias sin la ley de contacto**;
    lo que sí se puede es fijar el factor faltante contra un caso donde el
    balance es exacto, y transferirlo.

    La transferencia al cilindro rotatorio es defendible porque comparte los
    mismos parámetros de contacto que el caso con gravedad (DATA_NOTES.md, §5),
    pero sigue siendo una transferencia: `kappa` es un parámetro calibrado, y
    todo `p_diss` que lo use hereda ese estatus.

    Devuelve {"kappa": ..., "residual": ..., "per_window": [...]}.
    """
    raw = replace(cfg, kappa_diss=1.0)
    feats = coarse_grain_windows(traj, sdf, scaling, raw, windows, g_vec=g_vec, t0=t0)
    losses = np.array([-f.p_mech for f in feats])   # > 0 si el sistema pierde energía
    derived = np.array([f.p_diss for f in feats])
    if derived.sum() <= 0:
        return {"kappa": 1.0, "residual": float("nan"), "per_window": []}
    kappa = float(losses.sum() / derived.sum())
    residual = float(np.abs(losses - kappa * derived).sum() / np.abs(losses).sum())
    return {
        "kappa": kappa,
        "residual": residual,
        "per_window": [float(l / d) if d > 0 else float("nan")
                       for l, d in zip(losses, derived)],
    }


def coarse_grain_windows(traj, sdf, scaling: Scaling, cfg: CoarseConfig, windows,
                         *, g_vec=None, t0: float = 0.0,
                         events: EventTable | None = None) -> list[MacroFeatures]:
    """`C_phi` sobre una lista de ventanas, detectando eventos una sola vez.

    La detección es la parte cara (O(T·N²)); las ventanas solo reparten la tabla.
    """
    q = traj.q.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(traj.q) else np.asarray(traj.q, float)
    v = traj.v.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(traj.v) else np.asarray(traj.v, float)
    m = traj.m.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(traj.m) else np.asarray(traj.m, float)
    radii = traj.radii.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(traj.radii) else np.asarray(traj.radii, float)
    if events is None:
        events = detect_events(q, v, radii, m, sdf, delta=cfg.delta,
                               max_lookback=cfg.max_lookback, v_n_min=cfg.v_n_min,
                               dt=float(traj.dt), t0=t0, block=cfg.block)
    return [
        coarse_grain(traj, sdf, scaling, cfg, g_vec=g_vec, events=events,
                     window=win, t0=t0)
        for win in windows
    ]
