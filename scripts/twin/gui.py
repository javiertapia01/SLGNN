"""Interfaz gráfica para el banco de pruebas del esqueleto del gemelo digital SAG.

Es una herramienta para **probar**, no un componente científico: no calcula nada
que `src/twin/` no calcule ya, solo expone sus operadores con perillas y gráficos
en vez de argumentos de línea de comandos.

Dos decisiones de diseño que vale la pena hacer explícitas:

1. **Cero dependencias nuevas.** Tkinter viene con Python y matplotlib ya está en
   `requirements.txt`. Streamlit habría dado una interfaz más bonita a cambio de
   una dependencia y un servidor; no vale la pena para un banco de pruebas local.

2. **La biblioteca se construye una vez y se cachea.** `C_phi` sobre la rampa del
   cilindro (2073 esferas x 2001 pasos, ~835 mil eventos) cuesta ~30 s. Todo lo
   que viene después —PBM, MPC, lazo cerrado— cuesta milisegundos. Así que el
   paso caro se paga al abrir, se guarda en `results/twin/cache/`, y la
   interactividad real ocurre aguas abajo.

La interfaz arrastra los estatus epistémicos del paquete hasta la pantalla: lo
que es andamio aparece marcado como andamio, `p_in` aparece marcado como
derivado, y `F_feed` aparece marcado como sin respaldo microdinámico. Una
interfaz que los escondiera traicionaría el punto entero de `src/twin/`.

Uso:
    python scripts/twin/gui.py
    python scripts/twin/gui.py --config configs/twin/twin_toy.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import queue
import sys
import threading
import traceback
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("TkAgg")

import tkinter as tk  # noqa: E402
from tkinter import messagebox, ttk  # noqa: E402

from matplotlib.backends.backend_tkagg import (  # noqa: E402
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure  # noqa: E402

from twin.confidence import ConfidenceMonitor  # noqa: E402
from twin.control import (MPC, ConstantOmega, HoldupPI, MacroPlant,  # noqa: E402
                          MPCConfig, closed_loop)
from twin.coupling import AndamioWarning, make_policy  # noqa: E402
from twin.library import hysteresis_report, hysteresis_verdict  # noqa: E402
from twin.macro import (PBM, MacroState, PBMConfig, selection_rates,  # noqa: E402
                        sensitivity_to_shape)
from twin.pipeline import (build_library, coarse_config, coarse_ramp,  # noqa: E402
                           load_config, load_cylinder, pbm_config,
                           positive_omega_features, project_root, resolve_kappa)

CACHE_SUBDIR = "results/twin/cache"

BANNER = ("Banco de pruebas del ESQUELETO — no es un gemelo digital: sin conexión "
          "a planta, carga monodispersa, tambor de laboratorio.")

# `SpectrumLibrary` interpola en log-tasa y rellena los bins vacíos con
# log(1e-300), de modo que al volver de la exponencial quedan valores
# subnormales. Graficarlos estira el eje logarítmico ~300 décadas y vuelve el
# espectro ilegible, así que se enmascaran: son ausencia de eventos, no eventos
# rarísimos.
SENTINEL_RATE = 1e-250


ARM_STYLE = {
    "baseline_omega_const": dict(ls="-", lw=1.2),
    "baseline_pi_holdup": dict(ls="-", lw=1.2),
    "a_tasas_fijas": dict(ls="-", lw=4.0, alpha=0.35),
    "b_espectro_estatico": dict(ls="--", lw=2.2),
    "c_biblioteca": dict(ls=":", lw=2.2),
    "d_biblioteca_conservador": dict(ls="-.", lw=1.6),
}


def visible_rates(rates: np.ndarray) -> np.ndarray:
    r = np.asarray(rates, dtype=float)
    return np.where(r > SENTINEL_RATE, r, np.nan)


def set_spectrum_ylim(ax, *series, decades: float = 12.0) -> None:
    """Acota el eje a un rango dinámico legible por debajo del máximo."""
    tops = [np.nanmax(s) for s in series if np.any(np.isfinite(s))]
    tops = [t for t in tops if np.isfinite(t) and t > 0]
    if tops:
        top = max(tops)
        ax.set_ylim(top * 10.0 ** (-decades), top * 3.0)


# ---------------------------------------------------------------------------
# Caché de la biblioteca
# ---------------------------------------------------------------------------

def cache_key(cfg: dict, width_s: float) -> str:
    """Huella de todo lo que entra en `C_phi`.

    Solo las secciones que afectan al coarse-graining: cambiar `control` o `pbm`
    no debe invalidar 30 s de cómputo.
    """
    relevant = {k: cfg.get(k) for k in ("data", "coarse")}
    relevant["width_s"] = width_s
    blob = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def build_or_load_features(cfg: dict, root: Path, width_s: float, progress) -> dict:
    """Devuelve las `MacroFeatures` de la rampa, desde caché si existe."""
    path = root / CACHE_SUBDIR / f"ramp_{cache_key(cfg, width_s)}.pkl"
    if path.exists():
        progress(f"caché encontrada: {path.name}")
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        progress(f"biblioteca cargada de caché ({len(payload['features'])} ventanas)")
        return payload

    progress("cargando trayectoria del cilindro (2073 esferas x 2001 pasos)…")
    traj, sdf, g_vec, scaling, profile = load_cylinder(cfg, root)

    progress("calibrando kappa de disipación sobre la caja estática…")
    cal = resolve_kappa(cfg, root)

    progress(f"kappa = {cal['kappa']:.3f} — corriendo C_phi sobre la rampa "
             f"(~30 s, solo esta vez)…")
    ccfg = coarse_config(cfg, kappa=cal["kappa"])
    feats = coarse_ramp(traj, sdf, scaling, ccfg, g_vec, width_s, profile)

    payload = {"features": feats, "calibration": cal, "profile": profile,
               "scaling": scaling.to_dict(), "width_s": width_s}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(payload, fh)
    progress(f"caché guardada en {path.relative_to(root)}")
    return payload


# ---------------------------------------------------------------------------
# Widgets auxiliares
# ---------------------------------------------------------------------------

class ParamSlider(ttk.Frame):
    """Deslizador con etiqueta y lectura del valor.

    `log=True` mueve el deslizador en log10 y reporta el valor lineal, que es lo
    único usable para parámetros como `e_star_ref` que viven en décadas.
    """

    def __init__(self, parent, label, lo, hi, init, *, log=False, fmt="{:.4g}",
                 unit="", command=None, width=170, variable=None):
        super().__init__(parent)
        self.log = log
        self.fmt = fmt
        self.unit = unit
        self._command = command

        raw_lo, raw_hi = (np.log10(lo), np.log10(hi)) if log else (lo, hi)
        raw_init = np.log10(init) if log else init
        # `variable` permite espejar el mismo parámetro en dos pestañas: Tk
        # mueve ambos deslizadores solo, y el trace mantiene las dos lecturas
        # numéricas en sincronía.
        self.var = variable if variable is not None else tk.DoubleVar(
            value=float(np.clip(raw_init, raw_lo, raw_hi)))
        self.var.trace_add("write", lambda *_: self._refresh())

        ttk.Label(self, text=label, width=18).grid(row=0, column=0, sticky="w")
        self.readout = ttk.Label(self, text="", width=14, anchor="e",
                                 font=("Consolas", 9))
        self.readout.grid(row=0, column=1, sticky="e")
        self.scale = ttk.Scale(self, from_=raw_lo, to=raw_hi, variable=self.var,
                               orient="horizontal", length=width,
                               command=self._on_move)
        self.scale.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        self.columnconfigure(0, weight=1)
        self._refresh()

    @property
    def value(self) -> float:
        raw = float(self.var.get())
        return float(10.0 ** raw) if self.log else raw

    def set(self, value: float) -> None:
        self.var.set(float(np.log10(value)) if self.log else float(value))
        self._refresh()

    def _refresh(self) -> None:
        self.readout.configure(text=f"{self.fmt.format(self.value)}{self.unit}")

    def _on_move(self, _evt=None) -> None:
        self._refresh()
        if self._command is not None:
            self._command(self.value)


class PlotPanel(ttk.Frame):
    """Figura de matplotlib empotrada, con barra de herramientas."""

    def __init__(self, parent, figsize=(7.4, 5.2)):
        super().__init__(parent)
        self.figure = Figure(figsize=figsize, dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill="x")

    def draw(self) -> None:
        self.canvas.draw_idle()


def readonly_text(parent, height=10, width=52) -> tk.Text:
    txt = tk.Text(parent, height=height, width=width, font=("Consolas", 9),
                  wrap="word", background="#f7f7f7", relief="flat")
    txt.configure(state="disabled")
    return txt


def set_text(widget: tk.Text, content: str) -> None:
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    widget.insert("1.0", content)
    widget.configure(state="disabled")


# ---------------------------------------------------------------------------
# Aplicación
# ---------------------------------------------------------------------------

class TwinGUI:
    def __init__(self, root: tk.Tk, config_path: str):
        self.root = root
        self.project = project_root()
        self.config_path = self.project / config_path
        self.cfg = load_config(self.config_path)

        self.payload: dict | None = None      # features + calibración
        self.library = None                   # SpectrumLibrary (rama de avance)
        self.sweep: dict | None = None        # barrido precomputado en omega
        self._queue: queue.Queue = queue.Queue()
        self._busy = False

        root.title("Banco de pruebas — gemelo digital SAG (esqueleto)")
        root.geometry("1320x880")
        root.minsize(1080, 720)

        self._build_header()
        self._build_notebook()
        self._build_statusbar()

        self._pump()
        self.root.after(400, self.load_library)

    # -- estructura ---------------------------------------------------------

    def _build_header(self) -> None:
        head = ttk.Frame(self.root, padding=(10, 8, 10, 4))
        head.pack(fill="x")

        ttk.Label(head, text=BANNER, foreground="#8a4b00",
                  font=("Segoe UI", 9, "bold"), wraplength=1200,
                  justify="left").pack(anchor="w")

        bar = ttk.Frame(head)
        bar.pack(fill="x", pady=(6, 0))
        ttk.Label(bar, text=f"config: {self.config_path.name}",
                  font=("Consolas", 9)).pack(side="left")
        self.lib_label = ttk.Label(bar, text="biblioteca: sin construir",
                                   font=("Consolas", 9))
        self.lib_label.pack(side="left", padx=18)
        self.rebuild_btn = ttk.Button(bar, text="Reconstruir biblioteca",
                                      command=lambda: self.load_library(force=True))
        self.rebuild_btn.pack(side="right")

    def _build_notebook(self) -> None:
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=10, pady=6)
        self._tab_library()
        self._tab_hysteresis()
        self._tab_pbm()
        self._tab_loop()
        self._tab_arms()
        self.nb.bind("<<NotebookTabChanged>>", lambda _e: self._on_omega())

    def _build_statusbar(self) -> None:
        bar = ttk.Frame(self.root, padding=(10, 4))
        bar.pack(fill="x")
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=140)
        self.progress.pack(side="left")
        self.status = ttk.Label(bar, text="listo", font=("Consolas", 9))
        self.status.pack(side="left", padx=12)

    # -- infraestructura de hilos -------------------------------------------

    def _emit(self, msg: str) -> None:
        """Progreso desde un hilo trabajador. No toca Tk directamente."""
        self._queue.put(("progress", msg))

    def run_async(self, fn, on_done, label: str) -> None:
        """Corre `fn(progress)` fuera del hilo de Tk y entrega el resultado.

        Tk no es reentrante desde otros hilos, así que el trabajador solo pone
        mensajes en una cola y `_pump` los consume en el hilo principal.
        """
        if self._busy:
            messagebox.showinfo("Ocupado", "Hay un cálculo en curso.")
            return
        self._busy = True
        self.progress.start(12)
        self.status.configure(text=label)

        def worker():
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    result = fn(self._emit)
                andamios = [str(w.message) for w in caught
                            if issubclass(w.category, AndamioWarning)]
                self._queue.put(("done", on_done, result, andamios))
            except Exception:  # noqa: BLE001 — se muestra íntegro al usuario
                self._queue.put(("error", traceback.format_exc()))

        threading.Thread(target=worker, daemon=True).start()

    def _pump(self) -> None:
        try:
            while True:
                item = self._queue.get_nowait()
                kind = item[0]
                if kind == "progress":
                    self.status.configure(text=item[1])
                elif kind == "done":
                    _, callback, result, andamios = item
                    self._finish()
                    callback(result)
                    if andamios:
                        self.status.configure(
                            text="ANDAMIO activo: " + andamios[0][:110])
                elif kind == "error":
                    self._finish()
                    self.status.configure(text="error — ver diálogo")
                    messagebox.showerror("Error en el cálculo", item[1][-2500:])
        except queue.Empty:
            pass
        self.root.after(80, self._pump)

    def _finish(self) -> None:
        self._busy = False
        self.progress.stop()
        self.status.configure(text="listo")

    # =====================================================================
    # Pestaña 1 — biblioteca omega -> espectro
    # =====================================================================

    def _tab_library(self) -> None:
        tab = ttk.Frame(self.nb, padding=8)
        self.nb.add(tab, text="1 · Biblioteca ω→espectro")

        side = ttk.Frame(tab)
        side.pack(side="left", fill="y", padx=(0, 10))

        ttk.Label(side, text="Consulta a la biblioteca",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Label(side, text="El único arco con datos: ω → E^coll.",
                  foreground="#2c6e2c", font=("Segoe UI", 8)).pack(anchor="w",
                                                                   pady=(0, 8))

        self.omega_slider = ParamSlider(side, "ω [rad/s]", 0.1, 14.0, 6.0,
                                        fmt="{:.3f}", command=self._on_omega,
                                        width=220)
        self.omega_slider.pack(fill="x")

        self.lib_readout = readonly_text(side, height=22, width=46)
        self.lib_readout.pack(fill="both", expand=True, pady=(8, 0))

        self.lib_plot = PlotPanel(tab, figsize=(8.4, 6.4))
        self.lib_plot.pack(side="left", fill="both", expand=True)

        fig = self.lib_plot.figure
        self.ax_spec = fig.add_subplot(2, 1, 1)
        self.ax_sweep = fig.add_subplot(2, 1, 2)
        self.ax_sweep_r = self.ax_sweep.twinx()

    def _on_omega(self, _value=None) -> None:
        """Redibuja solo la pestaña visible.

        Arrastrar el deslizador dispara esto en cada píxel; rehacer también las
        figuras ocultas haría el arrastre perceptiblemente lento sin que nadie
        vea el resultado. Las ocultas se ponen al día al seleccionarlas.
        """
        if self.library is None:
            return
        {0: self.refresh_library_tab, 2: self._refresh_pbm}.get(
            self.nb.index(self.nb.select()), lambda: None)()

    def refresh_library_tab(self) -> None:
        omega = self.omega_slider.value
        reg = self.library.query(omega)
        spec = reg.spectrum

        ax = self.ax_spec
        ax.clear()
        centers = spec.centers
        series = []
        for col, (name, style) in enumerate([("partícula–partícula", "-"),
                                             ("partícula–pared", "--")]):
            r = visible_rates(spec.rates[:, col])
            series.append(r)
            ax.step(centers, r, style, where="mid", label=name)
        set_spectrum_ylim(ax, *series)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("energía de impacto E [J]")
        ax.set_ylabel("tasa n(E) [ev/(s·kg)]")
        ax.set_title(f"Espectro interpolado en ω = {omega:.3f} rad/s"
                     + ("   [FUERA DE RANGO]" if reg.ood else ""))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")

        ax2 = self.ax_sweep
        ax2r = self.ax_sweep_r
        ax2.clear()
        ax2r.clear()
        # `clear()` devuelve el eje gemelo al lado izquierdo y sus etiquetas se
        # superponen con las del principal; hay que reponerlo a la derecha.
        ax2r.yaxis.tick_right()
        ax2r.yaxis.set_label_position("right")

        sw = self.sweep
        ax2.plot(sw["omega"], sw["p_impact"], "-", color="#1f77b4",
                 label="p_impact (MEDIDA)")
        # Los nodos se dibujan separados por rama: donde la ascendente y la
        # descendente tienen nodos a la misma ω con distinta potencia, la curva
        # interpolada salta. Ese salto ES la histéresis, no un artefacto.
        nodes_om = self.library._omega
        nodes_p = self.library._scalars["p_impact"]
        for br, marker, color in (("up", "^", "#1f77b4"), ("down", "v", "#9467bd")):
            m = self.library._branches == br
            if m.any():
                ax2.plot(nodes_om[m], nodes_p[m], marker, ms=6, ls="none",
                         color=color, label=f"nodo rama {br}")
        ax2.set_yscale("log")
        ax2.set_xlabel("ω [rad/s]")
        ax2.set_ylabel("potencia de impacto [W]", color="#1f77b4")
        ax2.axvline(omega, color="k", lw=1.0, ls=":")
        ax2r.plot(sw["omega"], sw["sigma_spec"], "-", color="#d62728",
                  label="σ_spec (histéresis)")
        ax2r.axhline(float(self.cfg["confidence"]["sigma_threshold"]),
                     color="#d62728", lw=0.8, ls="--")
        ax2r.set_ylabel("σ_spec [décadas]", color="#d62728")
        ax2.set_title("Barrido en ω — nodos por rama e incertidumbre entre ramas")
        ax2.legend(fontsize=7, loc="upper center", ncol=3)
        ax2.grid(alpha=0.3)

        self.lib_plot.draw()

        lo, hi = self.library.omega_range
        set_text(self.lib_readout, f"""ω consultada        {omega:12.4f} rad/s
rango cubierto      [{lo:.3f}, {hi:.3f}] rad/s
fuera de rango      {"SÍ — consulta saturada al extremo" if reg.ood else "no"}
distancia al nodo   {reg.distance_to_node:12.4f} rad/s
nodos en la tabla   {len(self.library):12d}

— espectro ————————————————————
tasa total          {spec.total_rate():12.4e} ev/(s·kg)
energía media       {spec.mean_energy():12.4e} J
potencia impacto    {spec.energy_rate():12.4e} W/kg
masa de la ventana  {spec.mass:12.4e} kg
underflow (pp/pw)   {spec.underflow[0]:.2e} / {spec.underflow[1]:.2e}
overflow  (pp/pw)   {spec.overflow[0]:.2e} / {spec.overflow[1]:.2e}

— potencias ———————————————————
p_impact  MEDIDA    {reg.p_impact:12.4e} W
p_in      DERIVADA  {reg.p_in:12.4e} W
  p_in usa kappa calibrada en OTRA geometría (la caja
  estática). Sale negativa en varias ventanas del
  cilindro: por eso el MPC usa p_impact, no p_in.

— incertidumbre ———————————————
σ_spec (ramas)      {reg.sigma_spec:12.4f} décadas
σ_rate (ramas)      {reg.sigma_rate:12.4f} relativo
  σ_spec ES la histéresis medida en el experimento H,
  reutilizada como incertidumbre epistémica.

— carga ———————————————————————
KE                  {reg.ke:12.4e} J
com_r               {reg.com_r:12.4e} m""")

    # =====================================================================
    # Pestaña 2 — histéresis (experimento H)
    # =====================================================================

    def _tab_hysteresis(self) -> None:
        tab = ttk.Frame(self.nb, padding=8)
        self.nb.add(tab, text="2 · Histéresis (Exp. H)")

        side = ttk.Frame(tab)
        side.pack(side="left", fill="y", padx=(0, 10))

        ttk.Label(side, text="¿ω ↦ E^coll es una función?",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Label(side, text="Compara la rama ascendente contra la descendente\n"
                             "a la misma ω. Si difieren, la tabla no basta.",
                  font=("Segoe UI", 8), justify="left").pack(anchor="w",
                                                             pady=(0, 8))

        self.hy_probes = ParamSlider(side, "n° de sondeos", 4, 40,
                                     float(self.cfg["hysteresis"]["n_probes"]),
                                     fmt="{:.0f}", width=220)
        self.hy_probes.pack(fill="x")
        self.hy_wthr = ParamSlider(
            side, "umbral Wasserstein", 0.01, 1.0,
            float(self.cfg["hysteresis"]["wasserstein_threshold"]),
            fmt="{:.3f}", unit=" déc", width=220)
        self.hy_wthr.pack(fill="x")
        self.hy_rthr = ParamSlider(
            side, "umbral tasa", 0.05, 1.0,
            float(self.cfg["hysteresis"]["rate_threshold"]),
            fmt="{:.3f}", width=220)
        self.hy_rthr.pack(fill="x")

        self.hy_btn = ttk.Button(side, text="Correr reporte H",
                                 command=self.run_hysteresis, state="disabled")
        self.hy_btn.pack(fill="x", pady=8)

        self.hy_readout = readonly_text(side, height=20, width=46)
        self.hy_readout.pack(fill="both", expand=True)

        self.hy_plot = PlotPanel(tab, figsize=(8.4, 6.4))
        self.hy_plot.pack(side="left", fill="both", expand=True)

    def run_hysteresis(self) -> None:
        feats = self.payload["features"]
        n_probe = int(round(self.hy_probes.value))
        w_thr, r_thr = self.hy_wthr.value, self.hy_rthr.value

        def job(progress):
            progress("comparando ramas ascendente y descendente…")
            rep = hysteresis_report(feats, n_probe=n_probe)
            return rep, hysteresis_verdict(rep, w_thr, r_thr)

        self.run_async(job, self._show_hysteresis, "experimento H…")

    def _show_hysteresis(self, result) -> None:
        rep, verdict = result
        fig = self.hy_plot.figure
        fig.clear()

        if rep.get("status") != "ok":
            set_text(self.hy_readout,
                     f"reporte indeterminado: {rep.get('status')}\n{rep}")
            self.hy_plot.draw()
            return

        probes = rep["probes"]
        om = [p["omega"] for p in probes]
        wd = [p["wasserstein_decades"] for p in probes]
        re_ = [p["rate_rel_error"] for p in probes]
        w_thr, r_thr = self.hy_wthr.value, self.hy_rthr.value

        ax1 = fig.add_subplot(3, 1, 1)
        ax1.plot(om, wd, "o-", color="#d62728")
        ax1.axhline(w_thr, ls="--", color="k", lw=0.9,
                    label=f"umbral {w_thr:.2f}")
        ax1.set_ylabel("Wasserstein [décadas]")
        ax1.set_title("Discrepancia entre ramas a la misma ω")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

        ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
        ax2.plot(om, re_, "s-", color="#ff7f0e")
        ax2.axhline(r_thr, ls="--", color="k", lw=0.9)
        ax2.set_ylabel("error relativo de tasa")
        ax2.grid(alpha=0.3)

        ax3 = fig.add_subplot(3, 1, 3, sharex=ax1)
        ax3.semilogy(om, [p["rate_up"] for p in probes], "^-", label="rama ↑")
        ax3.semilogy(om, [p["rate_down"] for p in probes], "v-", label="rama ↓")
        ax3.set_xlabel("ω [rad/s]")
        ax3.set_ylabel("tasa total [ev/(s·kg)]")
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3, which="both")
        self.hy_plot.draw()

        worst = verdict.get("worst_probe")
        extra = ""
        if worst:
            extra = (f"\npeor sondeo\n"
                     f"  ω               {worst['omega']:10.3f} rad/s\n"
                     f"  Wasserstein     {worst['wasserstein_decades']:10.3f} déc\n"
                     f"  error de tasa   {worst['rate_rel_error']:10.3f}\n"
                     f"  tasa ↑ / ↓      {worst['rate_up']:.3e} / "
                     f"{worst['rate_down']:.3e}\n")
        rng = verdict.get("omega_violating_range")
        rng_txt = (f"  ω violatorias   [{rng[0]:.2f}, {rng[1]:.2f}] rad/s\n"
                   if rng else "")

        set_text(self.hy_readout, f"""VEREDICTO: {verdict['verdict'].upper()}

ventanas ↑ / ↓      {rep['n_up']} / {rep['n_down']}
solape en ω         [{rep['omega_overlap'][0]:.3f}, {rep['omega_overlap'][1]:.3f}]
sondeos             {verdict['n_probes']}
sondeos violatorios {verdict['n_violating']}
{rng_txt}
máx. Wasserstein    {verdict['max_wasserstein']:10.3f} décadas
máx. error de tasa  {verdict['max_rate_error']:10.3f}
mediana Wasserstein {rep['median_wasserstein']:10.3f} décadas
{extra}
Ancho de ventana: {self.payload['width_s']:.2f} s (el de la caché).
Cambiarlo exige reconstruir la biblioteca.

Si el veredicto es "depende del camino", la tabla ω→espectro
es incompleta por construcción, y esa misma cifra alimenta
σ_spec en el monitor de confianza de la pestaña 4.""")

    # =====================================================================
    # Pestaña 3 — PBM
    # =====================================================================

    def _tab_pbm(self) -> None:
        tab = ttk.Frame(self.nb, padding=8)
        self.nb.add(tab, text="3 · PBM (espectro→rotura)")

        side = ttk.Frame(tab)
        side.pack(side="left", fill="y", padx=(0, 10))

        ttk.Label(side, text="PBM energético de 6 clases",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Label(side, text="PARÁMETROS PLAUSIBLES, NO CALIBRADOS.\n"
                             "Las PSD de aquí no son predicciones.",
                  foreground="#a00000", font=("Segoe UI", 8, "bold"),
                  justify="left").pack(anchor="w", pady=(0, 8))

        # Espejo del deslizador de la pestaña 1: el espectro que alimenta al PBM
        # se consulta a una ω, y tenerla que ir a cambiar a otra pestaña sería
        # absurdo. Comparten variable, así que no pueden desincronizarse.
        self.omega_slider_pbm = ParamSlider(
            side, "ω [rad/s]", 0.1, 14.0, 6.0, fmt="{:.3f}",
            command=self._on_omega, width=220, variable=self.omega_slider.var)
        self.omega_slider_pbm.pack(fill="x")
        ttk.Separator(side, orient="horizontal").pack(fill="x", pady=6)

        p = self.cfg["pbm"]
        self.pbm_sliders = {}
        specs = [
            ("e_star_ref", "E* de rotura [J]", 1e-8, 1e-4, p["e_star_ref"], True),
            ("alpha", "α (E* vs tamaño)", 0.5, 4.0, p["alpha"], False),
            ("nu", "ν (ley de daño)", 0.25, 3.0, p["nu"], False),
            ("beta", "β (progenie G–S)", 0.2, 2.0, p["beta"], False),
            ("k_discharge", "k_descarga [1/s]", 0.01, 3.0, p["k_discharge"], False),
            ("d50", "d50 clasificador [m]", 1e-4, 5e-3, p["d50"], True),
            ("d_target", "d objetivo [m]", 1e-4, 5e-3, p["d_target"], True),
        ]
        for key, label, lo, hi, init, log in specs:
            s = ParamSlider(side, label, lo, hi, init, log=log,
                            command=lambda _v: self._refresh_pbm(), width=220)
            s.pack(fill="x")
            self.pbm_sliders[key] = s

        ttk.Separator(side, orient="horizontal").pack(fill="x", pady=6)
        ttk.Button(side, text="Restaurar valores del YAML",
                   command=self._reset_pbm).pack(fill="x")
        self.sens_btn = ttk.Button(side, text="Chequeo: ¿S_b ve la FORMA?",
                                   command=self.run_sensitivity, state="disabled")
        self.sens_btn.pack(fill="x", pady=4)

        self.pbm_readout = readonly_text(side, height=14, width=46)
        self.pbm_readout.pack(fill="both", expand=True, pady=(6, 0))

        self.pbm_plot = PlotPanel(tab, figsize=(8.4, 6.4))
        self.pbm_plot.pack(side="left", fill="both", expand=True)

    def current_pbm_config(self) -> PBMConfig:
        base = pbm_config(self.cfg)
        overrides = {k: s.value for k, s in self.pbm_sliders.items()}
        return PBMConfig(**{**base.__dict__, **overrides})

    def _reset_pbm(self) -> None:
        for key, s in self.pbm_sliders.items():
            s.set(float(self.cfg["pbm"][key]))
        self._refresh_pbm()

    def _refresh_pbm(self) -> None:
        if self.library is None:
            return
        cfg = self.current_pbm_config()
        omega = self.omega_slider.value
        reg = self.library.query(omega)
        S = selection_rates(reg.spectrum, cfg)
        pbm = PBM(cfg)
        sizes = cfg.sizes()
        disc = pbm._discharge

        fig = self.pbm_plot.figure
        fig.clear()
        idx = np.arange(cfg.n_classes)

        ax1 = fig.add_subplot(2, 1, 1)
        ax1.bar(idx - 0.2, np.maximum(S, 1e-30), 0.4, label="S_b (selección)")
        ax1.bar(idx + 0.2, np.maximum(disc, 1e-30), 0.4, label="descarga")
        ax1.set_yscale("log")
        ax1.set_xticks(idx)
        ax1.set_xticklabels([f"{d*1e3:.2f} mm" for d in sizes], fontsize=8)
        ax1.set_ylabel("tasa [1/s]")
        ax1.set_title(f"Tasas por clase en ω = {omega:.3f} rad/s "
                      f"(b=0 la más gruesa)")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3, axis="y")

        ax2 = fig.add_subplot(2, 1, 2)
        centers = reg.spectrum.centers
        n = visible_rates(reg.spectrum.rates.sum(axis=1))
        ax2.step(centers, n, where="mid", color="k", label="n(E) del espectro")
        set_spectrum_ylim(ax2, n)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("energía E [J]")
        ax2.set_ylabel("n(E) [ev/(s·kg)]", color="k")
        e_star = cfg.e_star_ref * (sizes / cfg.d_ref) ** cfg.alpha
        # Dónde cae E*_b respecto del espectro decide si la clase se rompe: a la
        # izquierda del grueso de n(E) se rompe, a la derecha es inerte.
        top = ax2.get_ylim()[1]
        for b, es in enumerate(e_star):
            ax2.axvline(es, ls=":", lw=1.1, alpha=0.85, color="#7a3b8f")
            ax2.annotate(f"b={b}", (es, top), fontsize=7, color="#7a3b8f",
                         rotation=90, ha="right", va="top")
        ax2.plot([], [], ":", color="#7a3b8f", label="E*_b por clase")
        ax2.set_title("Espectro vs E*_b de cada clase")
        ax2.legend(fontsize=8, loc="lower left")
        ax2.grid(alpha=0.3, which="both")
        self.pbm_plot.draw()

        lines = ["clase   tamaño     S_b [1/s]   descarga   E*_b [J]", "-" * 52]
        for b in range(cfg.n_classes):
            lines.append(f"  {b}   {sizes[b]*1e3:7.3f} mm  {S[b]:10.3e}  "
                         f"{disc[b]:8.3e}  {e_star[b]:9.3e}")
        bajo = sizes < cfg.d_target
        lines.append("")
        lines.append(f"clases bajo el objetivo: {int(bajo.sum())} de {cfg.n_classes}")
        lines.append(f"masa de partícula (b=0): {cfg.particle_masses()[0]:.3e} kg")
        set_text(self.pbm_readout, "\n".join(lines))

    def run_sensitivity(self) -> None:
        cfg = self.current_pbm_config()
        template = self.library.nodes[0].spectrum

        def job(progress):
            progress("comparando espectro blando vs duro a energía igualada…")
            return sensitivity_to_shape(cfg, template)

        self.run_async(job, self._show_sensitivity, "chequeo de sensibilidad…")

    def _show_sensitivity(self, sens: dict) -> None:
        ratio = np.asarray(sens["ratio_per_class"])
        veredicto = ("SENSIBLE A LA FORMA — el puente micro-macro es visible"
                     if sens["shape_sensitive"] else
                     "CIEGO A LA FORMA — E1 daría un falso negativo por "
                     "construcción")
        msg = (f"{veredicto}\n\n"
               f"energía total igualada:\n"
               f"  blando {sens['energy_soft']:.4e}  vs  duro "
               f"{sens['energy_hard']:.4e}\n\n"
               f"razón S_duro/S_blando por clase:\n  "
               + "\n  ".join(f"b={b}: {r:.4e}" for b, r in enumerate(ratio))
               + f"\n\nmáx {sens['max_ratio']:.4e}   mín {sens['min_ratio']:.4e}\n\n"
                 "Si la razón fuera ≈1 en todas las clases, S_b dependería solo\n"
                 "de la energía total y el PBM no vería el espectro.")
        messagebox.showinfo("¿S_b responde a la forma del espectro?", msg)

    # =====================================================================
    # Pestaña 4 — lazo cerrado
    # =====================================================================

    def _tab_loop(self) -> None:
        tab = ttk.Frame(self.nb, padding=8)
        self.nb.add(tab, text="4 · Lazo cerrado (MPC)")

        side = ttk.Frame(tab)
        side.pack(side="left", fill="y", padx=(0, 10))

        ttk.Label(side, text="MPC restringido",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Label(side, text="ω tiene respaldo microdinámico.\n"
                             "F_feed NO: entra solo por el balance macro.",
                  foreground="#8a4b00", font=("Segoe UI", 8, "bold"),
                  justify="left").pack(anchor="w", pady=(0, 8))

        form = ttk.Frame(side)
        form.pack(fill="x")
        ttk.Label(form, text="controlador").grid(row=0, column=0, sticky="w")
        self.ctrl_var = tk.StringVar(value="mpc")
        ttk.Combobox(form, textvariable=self.ctrl_var, state="readonly", width=20,
                     values=["mpc", "omega_constante", "pi_holdup"]).grid(
            row=0, column=1, sticky="e", pady=2)
        ttk.Label(form, text="modo de espectro").grid(row=1, column=0, sticky="w")
        self.mode_var = tk.StringVar(value="library")
        ttk.Combobox(form, textvariable=self.mode_var, state="readonly", width=20,
                     values=["library", "static", "fixed_rates", "frozen"]).grid(
            row=1, column=1, sticky="e", pady=2)
        form.columnconfigure(1, weight=1)

        c = self.cfg["control"]
        self.loop_sliders = {}
        specs = [
            ("lambda_q", "λ_q (producción)", 0.05, 20.0, c["lambda_q"], False),
            ("lambda_e", "λ_e (energía)", 1e-4, 5.0, c["lambda_e"], True),
            ("lambda_du", "λ_du (suavidad)", 1e-5, 1.0, c["lambda_du"], True),
            ("horizon", "horizonte [pasos]", 2, 30, c["horizon"], False),
            ("dt", "dt macro [s]", 0.1, 3.0, c["dt"], False),
            ("d_omega_max", "Δω máx [rad/s]", 0.2, 8.0, c["d_omega_max"], False),
            ("feed_max", "F_feed máx [kg/s]", 0.0, 0.3, c["feed_max"], False),
            ("h_max", "hold-up máx [kg]", 0.5, 30.0, c["h_max"], False),
            ("omega_nominal", "ω nominal [rad/s]", 0.5, 14.0,
             c["omega_nominal"], False),
        ]
        for key, label, lo, hi, init, log in specs:
            s = ParamSlider(side, label, lo, hi, init, log=log,
                            fmt="{:.0f}" if key == "horizon" else "{:.4g}",
                            width=220)
            s.pack(fill="x")
            self.loop_sliders[key] = s

        self.steps_slider = ParamSlider(
            side, "pasos de lazo", 5, 120,
            float(self.cfg["experiment"]["n_closed_loop_steps"]),
            fmt="{:.0f}", width=220)
        self.steps_slider.pack(fill="x")
        self.holdup_slider = ParamSlider(
            side, "hold-up inicial [kg]", 0.1, 10.0,
            float(self.cfg["experiment"]["initial_holdup_kg"]),
            fmt="{:.3g}", width=220)
        self.holdup_slider.pack(fill="x")

        self.monitor_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(side, text="monitor de confianza (restringe ω)",
                        variable=self.monitor_var).pack(anchor="w", pady=(6, 0))
        self.psd_var = tk.BooleanVar(value=bool(c.get("psd_feedback", False)))
        ttk.Checkbutton(side, text="realimentación PSD→espectro  [ANDAMIO]",
                        variable=self.psd_var).pack(anchor="w")

        self.loop_btn = ttk.Button(side, text="Correr lazo cerrado",
                                   command=self.run_loop, state="disabled")
        self.loop_btn.pack(fill="x", pady=8)

        self.loop_readout = readonly_text(side, height=12, width=46)
        self.loop_readout.pack(fill="both", expand=True)

        self.loop_plot = PlotPanel(tab, figsize=(8.4, 6.4))
        self.loop_plot.pack(side="left", fill="both", expand=True)

    def mpc_config(self, **overrides) -> MPCConfig:
        """Igual que `_mpc_config` de exp_E1, pero leyendo los deslizadores."""
        lo, hi = self.library.omega_range
        s = {k: v.value for k, v in self.loop_sliders.items()}
        base = MPCConfig(
            horizon=int(round(s["horizon"])), dt=s["dt"],
            lambda_q=s["lambda_q"], lambda_e=s["lambda_e"],
            lambda_du=s["lambda_du"],
            omega_min=max(lo, 0.0), omega_max=hi,
            feed_min=float(self.cfg["control"]["feed_min"]),
            feed_max=s["feed_max"],
            p_max=float(self.cfg["control"]["p_max"]),
            h_max=s["h_max"], d_omega_max=s["d_omega_max"],
            omega_nominal=float(np.clip(s["omega_nominal"], lo, hi)),
            iters=int(self.cfg["control"]["iters"]),
            lr=float(self.cfg["control"]["lr"]),
            psd_feedback=bool(self.psd_var.get()),
            psd_exponent=float(self.cfg["control"].get("psd_exponent", -0.5)),
        )
        for k, v in overrides.items():
            setattr(base, k, v)
        return base

    def _initial_state(self, pcfg: PBMConfig) -> MacroState:
        h = self.holdup_slider.value
        return MacroState(M=np.array([h] + [0.0] * (pcfg.n_classes - 1)))

    def run_loop(self) -> None:
        pcfg = self.current_pbm_config()
        mcfg = self.mpc_config(spectrum_mode=self.mode_var.get())
        truth_cfg = self.mpc_config(spectrum_mode="library")
        kind = self.ctrl_var.get()
        use_monitor = bool(self.monitor_var.get())
        n_steps = int(round(self.steps_slider.value))
        init = self._initial_state(pcfg)
        lib = self.library
        conf = self.cfg["confidence"]

        def job(progress):
            pbm = PBM(pcfg)
            plant = MacroPlant(pbm, lib, mcfg)
            truth = MacroPlant(pbm, lib, truth_cfg)
            monitor = None
            if use_monitor:
                monitor = ConfidenceMonitor.from_library(
                    lib, ood_threshold=float(conf["ood_threshold"]),
                    sigma_threshold=float(conf["sigma_threshold"]),
                    shrink=float(conf["shrink"]))
            if kind == "omega_constante":
                ctrl = ConstantOmega(mcfg.omega_nominal,
                                     0.5 * (mcfg.feed_min + mcfg.feed_max))
            elif kind == "pi_holdup":
                sp = mcfg.h_max * 0.6 if np.isfinite(mcfg.h_max) else 3.0
                ctrl = HoldupPI(setpoint=sp, omega=mcfg.omega_nominal,
                                feed_bounds=(mcfg.feed_min, mcfg.feed_max))
            else:
                ctrl = MPC(mcfg, monitor=monitor, policy=make_policy("library"))
            progress(f"corriendo {n_steps} pasos de lazo cerrado…")
            return closed_loop(ctrl, plant, init, n_steps, truth=truth), mcfg

        self.run_async(job, self._show_loop, "lazo cerrado…")

    def _show_loop(self, result) -> None:
        res, mcfg = result
        h = res["history"]
        k = [r["k"] for r in h]

        fig = self.loop_plot.figure
        fig.clear()

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(k, [r["omega"] for r in h], "-o", ms=3)
        ax1.axhline(mcfg.omega_min, ls=":", color="k", lw=0.8)
        ax1.axhline(mcfg.omega_max, ls=":", color="k", lw=0.8)
        ood = [(r["k"], r["omega"]) for r in h if r["ood"]]
        if ood:
            ax1.plot([a for a, _ in ood], [b for _, b in ood], "rx", ms=7,
                     label="monitor disparado")
            ax1.legend(fontsize=8)
        ax1.set_ylabel("ω [rad/s]")
        ax1.set_title("acción ω  (CON respaldo micro)")
        ax1.grid(alpha=0.3)

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(k, [r["feed"] for r in h], "-o", ms=3, color="#8a4b00")
        ax2.set_ylabel("F_feed [kg/s]")
        ax2.set_title("acción F_feed  (SIN respaldo micro)")
        ax2.grid(alpha=0.3)

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(k, [r["holdup"] for r in h], "-", label="hold-up [kg]")
        ax3.set_xlabel("paso macro")
        ax3.set_ylabel("hold-up [kg]")
        ax3b = ax3.twinx()
        ax3b.plot(k, [r["p80"] * 1e3 for r in h], "--", color="#d62728",
                  label="P80 [mm]")
        ax3b.set_ylabel("P80 [mm]", color="#d62728")
        ax3.set_title("estado macro")
        ax3.grid(alpha=0.3)

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(k, [r["product"] for r in h], "-", label="producto [kg/s]")
        ax4.set_xlabel("paso macro")
        ax4.set_ylabel("producto [kg/s]")
        ax4b = ax4.twinx()
        ax4b.plot(k, np.cumsum([r["stage_cost"] for r in h]), "--",
                  color="#2ca02c")
        ax4b.set_ylabel("costo acumulado", color="#2ca02c")
        ax4.set_title("producción y costo")
        ax4.grid(alpha=0.3)
        self.loop_plot.draw()

        omegas = np.array([r["omega"] for r in h])
        n_ood = sum(1 for r in h if r["ood"])
        set_text(self.loop_readout, f"""controlador         {res['controller']}
modo de espectro    {res['spectrum_mode']}
pasos               {len(h)}   dt = {mcfg.dt:.3g} s

costo acumulado     {res['total_cost']:+12.5f}
producto            {res['product_kg']:12.5f} kg
energía             {res['energy_J']:12.4e} J
energía específica  {res['specific_energy_J_per_kg']:12.4e} J/kg
P80 final           {res['final_p80']*1e3:12.4f} mm

ω  media            {omegas.mean():12.3f} rad/s
ω  mín / máx        {omegas.min():.3f} / {omegas.max():.3f}
pasos con monitor   {n_ood} de {len(h)}

masa final por clase [kg]:
""" + "\n".join(f"  b={b}  {m:10.5f}"
                for b, m in enumerate(res["final_state"])))

    # =====================================================================
    # Pestaña 5 — comparación de arcos (E1)
    # =====================================================================

    def _tab_arms(self) -> None:
        tab = ttk.Frame(self.nb, padding=8)
        self.nb.add(tab, text="5 · Arcos E1")

        side = ttk.Frame(tab)
        side.pack(side="left", fill="y", padx=(0, 10))

        ttk.Label(side, text="¿El espectro cambia decisiones?",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Label(side, text="Los seis arcos planifican con su propio modelo\n"
                             "y avanzan contra la MISMA planta. La diferencia\n"
                             "de costo es el valor de información del puente.",
                  font=("Segoe UI", 8), justify="left").pack(anchor="w",
                                                             pady=(0, 8))

        self.arms_btn = ttk.Button(side, text="Correr los 6 arcos",
                                   command=self.run_arms, state="disabled")
        self.arms_btn.pack(fill="x", pady=4)
        ttk.Label(side, text="Usa los parámetros de las pestañas 3 y 4.",
                  font=("Segoe UI", 8)).pack(anchor="w")

        cols = ("arco", "costo", "Δ vs (c)", "ω medio", "prod [kg]")
        self.arms_tree = ttk.Treeview(side, columns=cols, show="headings",
                                      height=8)
        for c, w in zip(cols, (150, 80, 80, 70, 80)):
            self.arms_tree.heading(c, text=c)
            self.arms_tree.column(c, width=w, anchor="e")
        self.arms_tree.column("arco", anchor="w")
        self.arms_tree.pack(fill="x", pady=8)

        self.arms_readout = readonly_text(side, height=14, width=56)
        self.arms_readout.pack(fill="both", expand=True)

        self.arms_plot = PlotPanel(tab, figsize=(8.4, 6.4))
        self.arms_plot.pack(side="left", fill="both", expand=True)

    def run_arms(self) -> None:
        pcfg = self.current_pbm_config()
        n_steps = int(round(self.steps_slider.value))
        init = self._initial_state(pcfg)
        lib = self.library
        conf = self.cfg["confidence"]
        cfgs = {name: self.mpc_config(spectrum_mode=mode)
                for name, mode in [("a_tasas_fijas", "fixed_rates"),
                                   ("b_espectro_estatico", "static"),
                                   ("c_biblioteca", "library"),
                                   ("d_biblioteca_conservador", "library")]}
        truth_cfg = self.mpc_config(spectrum_mode="library")

        def job(progress):
            pbm = PBM(pcfg)
            truth = MacroPlant(pbm, lib, truth_cfg)
            monitor = ConfidenceMonitor.from_library(
                lib, ood_threshold=float(conf["ood_threshold"]),
                sigma_threshold=float(conf["sigma_threshold"]),
                shrink=float(conf["shrink"]))
            arms = {
                "baseline_omega_const": (
                    ConstantOmega(truth_cfg.omega_nominal,
                                  0.5 * (truth_cfg.feed_min + truth_cfg.feed_max)),
                    truth_cfg, None),
                "baseline_pi_holdup": (
                    HoldupPI(setpoint=(truth_cfg.h_max * 0.6
                                       if np.isfinite(truth_cfg.h_max) else 3.0),
                             omega=truth_cfg.omega_nominal,
                             feed_bounds=(truth_cfg.feed_min, truth_cfg.feed_max)),
                    truth_cfg, None),
                "a_tasas_fijas": (None, cfgs["a_tasas_fijas"], None),
                "b_espectro_estatico": (None, cfgs["b_espectro_estatico"], None),
                "c_biblioteca": (None, cfgs["c_biblioteca"], None),
                "d_biblioteca_conservador": (None, cfgs["d_biblioteca_conservador"],
                                             monitor),
            }
            out = {}
            for i, (name, (ctrl, mcfg, mon)) in enumerate(arms.items(), 1):
                progress(f"arco {i}/6 — {name}…")
                plant = MacroPlant(pbm, lib, mcfg)
                controller = ctrl or MPC(mcfg, monitor=mon,
                                         policy=make_policy("library"))
                out[name] = closed_loop(controller, plant, init.copy(), n_steps,
                                        truth=truth)
            return out

        self.run_async(job, self._show_arms, "corriendo los 6 arcos…")

    def _show_arms(self, results: dict) -> None:
        ref = results["c_biblioteca"]
        ref_omega = float(np.mean([r["omega"] for r in ref["history"]]))

        for row in self.arms_tree.get_children():
            self.arms_tree.delete(row)
        rows = {}
        for name, res in results.items():
            om = np.array([r["omega"] for r in res["history"]])
            d_cost = res["total_cost"] - ref["total_cost"]
            rows[name] = {"delta_cost": d_cost,
                          "delta_omega": float(om.mean() - ref_omega),
                          "omega_mean": float(om.mean()),
                          "cost": res["total_cost"],
                          "product": res["product_kg"]}
            self.arms_tree.insert("", "end", values=(
                name, f"{res['total_cost']:.4f}", f"{d_cost:+.4f}",
                f"{om.mean():.2f}", f"{res['product_kg']:.4f}"))

        fig = self.arms_plot.figure
        fig.clear()
        ax1 = fig.add_subplot(2, 1, 1)
        # Los arcos (a), (b) y (c) suelen coincidir casi exactamente — ése es
        # justamente el resultado de E1. Con un trazo uniforme el último dibujado
        # taparía a los demás y parecería que faltan, así que cada arco lleva un
        # grosor y un guionado distintos y todos quedan visibles al solaparse.
        for name, res in results.items():
            ax1.plot([r["k"] for r in res["history"]],
                     [r["omega"] for r in res["history"]], label=name,
                     **ARM_STYLE.get(name, {"ls": "-", "lw": 1.4}))
        ax1.set_ylabel("ω [rad/s]")
        ax1.set_xlabel("paso macro")
        ax1.set_title("Acción elegida por cada arco")
        ax1.legend(fontsize=7, ncol=2)
        ax1.grid(alpha=0.3)

        ax2 = fig.add_subplot(2, 1, 2)
        names = list(rows)
        ax2.bar(range(len(names)), [rows[n]["delta_cost"] for n in names],
                color=["#999" if "baseline" in n else "#1f77b4" for n in names])
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=20, ha="right", fontsize=7)
        ax2.axhline(0, color="k", lw=0.8)
        ax2.set_ylabel("Δ costo vs (c) biblioteca")
        ax2.set_title("Costo de tener una peor representación del espectro")
        ax2.grid(alpha=0.3, axis="y")
        self.arms_plot.draw()

        decision_gap = max(abs(rows[k]["delta_omega"])
                           for k in ("a_tasas_fijas", "b_espectro_estatico"))
        cost_gap = max(abs(rows[k]["delta_cost"])
                       for k in ("a_tasas_fijas", "b_espectro_estatico"))
        rel = cost_gap / max(abs(ref["total_cost"]), 1e-12)
        if decision_gap < 0.05 and rel < 0.01:
            verdict = ("El puente micro-macro NO cambia decisiones en este\n"
                       "régimen. Antes de invertir en fidelidad del surrogate,\n"
                       "buscar un régimen donde sí las cambie: transitorios\n"
                       "rápidos de ω, o dureza variable simulada.")
        else:
            verdict = ("El puente micro-macro SÍ cambia decisiones: un\n"
                       "controlador que ignora el espectro elige otra ω y\n"
                       "paga otro costo.")

        set_text(self.arms_readout, f"""máx. diferencia de acción vs (c): {decision_gap:.4f} rad/s
máx. diferencia de costo  vs (c): {cost_gap:.5f}  ({rel:.2%})

{verdict}

Recordatorio (§4): la única acción con respaldo micro-
dinámico es ω. F_feed entra solo por el balance macro —
no hay dato que ligue la alimentación al espectro.""")

    # =====================================================================
    # Carga de la biblioteca
    # =====================================================================

    def load_library(self, force: bool = False) -> None:
        cfg = self.cfg
        root = self.project
        width = float(cfg["hysteresis"]["reference_width_s"])
        if force:
            path = root / CACHE_SUBDIR / f"ramp_{cache_key(cfg, width)}.pkl"
            path.unlink(missing_ok=True)

        def job(progress):
            payload = build_or_load_features(cfg, root, width, progress)
            lib = build_library(positive_omega_features(payload["features"]))
            progress("precomputando el barrido en ω…")
            lo, hi = lib.omega_range
            grid = np.linspace(lo, hi, 160)
            regs = [lib.query(float(o)) for o in grid]
            sweep = {
                "omega": grid,
                "p_impact": np.array([r.p_impact for r in regs]),
                "p_in": np.array([r.p_in for r in regs]),
                "total_rate": np.array([r.spectrum.total_rate() for r in regs]),
                "sigma_spec": np.array([r.sigma_spec if np.isfinite(r.sigma_spec)
                                        else np.nan for r in regs]),
            }
            return payload, lib, sweep

        self.run_async(job, self._on_library, "construyendo la biblioteca…")

    def _on_library(self, result) -> None:
        self.payload, self.library, self.sweep = result
        lo, hi = self.library.omega_range
        kappa = self.payload["calibration"]
        self.lib_label.configure(
            text=f"biblioteca: {len(self.library)} nodos, "
                 f"ω ∈ [{lo:.2f}, {hi:.2f}] rad/s | κ = {kappa['kappa']:.3f} "
                 f"({kappa.get('source', '?')[:38]})")

        for sl in (self.omega_slider, self.omega_slider_pbm):
            sl.scale.configure(from_=max(lo * 0.6, 0.05), to=hi * 1.25)
        self.omega_slider.set(float(np.clip(self.cfg["control"]["omega_nominal"],
                                            lo, hi)))
        for btn in (self.hy_btn, self.sens_btn, self.loop_btn, self.arms_btn):
            btn.configure(state="normal")

        self.refresh_library_tab()
        self._refresh_pbm()
        self.status.configure(text=f"biblioteca lista — {len(self.library)} nodos")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/twin/twin_toy.yaml")
    args = ap.parse_args()

    root = tk.Tk()
    try:
        root.call("tk", "scaling", 1.2)
    except tk.TclError:
        pass
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    TwinGUI(root, args.config)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
