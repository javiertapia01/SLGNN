"""Experimento E1 — ¿el espectro cambia alguna decisión de control? (§8)

Corre el lazo macro completo con espectros derivados de las trayectorias MFiX y
compara cuatro arcos:

  (a) PBM con tasas fijas          — el puente micro-macro apagado
  (b) espectro estático a omega nominal
  (c) espectro consultado desde la biblioteca   ← el puente activo
  (d) biblioteca + incertidumbre, MPC conservador

Más dos baselines sin MPC: omega constante y PI sobre el hold-up.

Todos planifican contra su propio modelo y avanzan contra la **misma** planta
(la biblioteca), de modo que la diferencia de costo acumulado es exactamente el
costo de tener una representación peor del espectro. Eso es el valor de
información del puente.

Si (a), (b) y (c) producen decisiones equivalentes, el puente no tiene valor
decisional en este régimen, y la tarea inmediata es buscar un régimen donde lo
tenga antes de invertir en fidelidad del surrogate.

Uso:
    python experiments/exp_E1_value_of_information.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twin.confidence import ConfidenceMonitor  # noqa: E402
from twin.control import (MPC, ConstantOmega, HoldupPI, MacroPlant,  # noqa: E402
                          MPCConfig, closed_loop)
from twin.coupling import make_policy  # noqa: E402
from twin.harness import enable_utf8_stdout, save_report  # noqa: E402
from twin.macro import PBM, MacroState  # noqa: E402
from twin.macro import sensitivity_to_shape  # noqa: E402
from twin.pipeline import (build_library, coarse_config, coarse_ramp,  # noqa: E402
                           load_config, load_cylinder, pbm_config,
                           positive_omega_features, project_root, resolve_kappa)


def _mpc_config(cfg, lib, **overrides) -> MPCConfig:
    c = dict(cfg["control"])
    c.pop("psd_feedback", None), c.pop("psd_exponent", None)
    lo, hi = lib.omega_range
    base = MPCConfig(
        horizon=c["horizon"], dt=c["dt"], lambda_q=c["lambda_q"],
        lambda_e=c["lambda_e"], lambda_du=c["lambda_du"],
        omega_min=max(lo, 0.0), omega_max=hi,
        feed_min=c["feed_min"], feed_max=c["feed_max"],
        p_max=float(c["p_max"]), h_max=float(c["h_max"]),
        d_omega_max=c["d_omega_max"], omega_nominal=c["omega_nominal"],
        iters=c["iters"], lr=c["lr"],
        psd_feedback=bool(cfg["control"].get("psd_feedback", False)),
        psd_exponent=float(cfg["control"].get("psd_exponent", -0.5)),
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def main() -> int:
    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/twin_toy.yaml")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = project_root()
    cfg = load_config(root / args.config)
    np.random.seed(cfg.get("seed", 0))

    # ------------------------------------------------------------ biblioteca
    print("=" * 78)
    print("PASO 1 — biblioteca de regímenes desde la rampa de omega")
    print("=" * 78)
    t0 = time.time()
    traj, sdf, g_vec, scaling, profile = load_cylinder(cfg, root)
    cal = resolve_kappa(cfg, root)
    ccfg = coarse_config(cfg, kappa=cal["kappa"])
    feats = coarse_ramp(traj, sdf, scaling, ccfg, g_vec,
                        cfg["hysteresis"]["reference_width_s"], profile)
    forward = positive_omega_features(feats)
    lib = build_library(forward)
    print(f"  {len(lib)} nodos de avance, omega ∈ "
          f"[{lib.omega_range[0]:.2f}, {lib.omega_range[1]:.2f}] rad/s  "
          f"[{time.time() - t0:.1f} s]")
    print(f"  kappa_diss = {cal['kappa']:.3f}")

    # ------------------------------------------------- chequeo indispensable
    print("\n" + "=" * 78)
    print("PASO 2 — ¿es S_b sensible a la FORMA del espectro? (§7.4)")
    print("=" * 78)
    pcfg = pbm_config(cfg)
    sens = sensitivity_to_shape(pcfg, lib.nodes[0].spectrum)
    print(f"  energía total igualada: {sens['energy_soft']:.4e} vs "
          f"{sens['energy_hard']:.4e}")
    print(f"  razón S_dura/S_blanda por clase: "
          f"{np.round(sens['ratio_per_class'], 2).tolist()}")
    print(f"  sensible a la forma: {sens['shape_sensitive']}")
    if not sens["shape_sensitive"]:
        print("  ABORTA: el PBM es ciego al puente micro-macro; E1 daría un "
              "falso negativo por construcción.")
        return 1

    # ------------------------------------------------------------ arcos
    print("\n" + "=" * 78)
    print("PASO 3 — lazo cerrado, cuatro arcos + dos baselines")
    print("=" * 78)
    n_steps = args.steps or int(cfg["experiment"]["n_closed_loop_steps"])
    pbm = PBM(pcfg)
    holdup = float(cfg["experiment"]["initial_holdup_kg"])
    init = MacroState(M=np.array([holdup] + [0.0] * (pcfg.n_classes - 1)))

    truth_cfg = _mpc_config(cfg, lib, spectrum_mode="library")
    truth = MacroPlant(pbm, lib, truth_cfg)
    monitor = ConfidenceMonitor.from_library(
        lib, ood_threshold=cfg["confidence"]["ood_threshold"],
        sigma_threshold=cfg["confidence"]["sigma_threshold"],
        shrink=cfg["confidence"]["shrink"],
    )
    omega_nom = truth_cfg.omega_nominal

    arms = {
        "baseline_omega_constante": (
            ConstantOmega(omega_nom, 0.5 * (truth_cfg.feed_min + truth_cfg.feed_max)),
            _mpc_config(cfg, lib, spectrum_mode="library"), None),
        "baseline_pi_holdup": (
            HoldupPI(setpoint=truth_cfg.h_max * 0.6, omega=omega_nom,
                     feed_bounds=(truth_cfg.feed_min, truth_cfg.feed_max)),
            _mpc_config(cfg, lib, spectrum_mode="library"), None),
        "a_tasas_fijas": (None, _mpc_config(cfg, lib, spectrum_mode="fixed_rates"), None),
        "b_espectro_estatico": (None, _mpc_config(cfg, lib, spectrum_mode="static"), None),
        "c_biblioteca": (None, _mpc_config(cfg, lib, spectrum_mode="library"), None),
        "d_biblioteca_conservador": (
            None, _mpc_config(cfg, lib, spectrum_mode="library"), monitor),
    }

    results = {}
    for name, (ctrl, mcfg, mon) in arms.items():
        t0 = time.time()
        plant = MacroPlant(pbm, lib, mcfg)
        controller = ctrl or MPC(mcfg, monitor=mon, policy=make_policy("library"))
        res = closed_loop(controller, plant, init.copy(), n_steps, truth=truth)
        res["seconds"] = time.time() - t0
        results[name] = res
        omegas = [h["omega"] for h in res["history"]]
        print(f"\n  {name}")
        print(f"    costo acumulado      : {res['total_cost']:+.5f}")
        print(f"    producto             : {res['product_kg']:.5f} kg")
        print(f"    energía específica   : {res['specific_energy_J_per_kg']:.4e} J/kg")
        print(f"    P80 final            : {res['final_p80']:.4e} m")
        print(f"    omega  media/mín/máx : {np.mean(omegas):.2f} / "
              f"{np.min(omegas):.2f} / {np.max(omegas):.2f} rad/s")
        print(f"    [{res['seconds']:.1f} s]")

    # ------------------------------------------------------------ veredicto
    print("\n" + "=" * 78)
    print("VEREDICTO DE E1")
    print("=" * 78)
    ref = results["c_biblioteca"]
    print(f"  {'arco':>28} {'costo':>12} {'Δ vs (c)':>12} {'omega medio':>12} "
          f"{'Δomega':>10}")
    ref_omega = np.mean([h["omega"] for h in ref["history"]])
    table = {}
    for name, res in results.items():
        omegas = np.array([h["omega"] for h in res["history"]])
        d_cost = res["total_cost"] - ref["total_cost"]
        d_omega = float(np.mean(omegas) - ref_omega)
        table[name] = {"delta_cost": d_cost, "delta_omega_mean": d_omega}
        print(f"  {name:>28} {res['total_cost']:12.5f} {d_cost:+12.5f} "
              f"{np.mean(omegas):12.2f} {d_omega:+10.2f}")

    decision_gap = max(abs(table[k]["delta_omega_mean"])
                       for k in ("a_tasas_fijas", "b_espectro_estatico"))
    cost_gap = max(abs(table[k]["delta_cost"])
                   for k in ("a_tasas_fijas", "b_espectro_estatico"))
    rel_cost = cost_gap / max(abs(ref["total_cost"]), 1e-12)
    print(f"\n  máxima diferencia de acción frente a (c): {decision_gap:.3f} rad/s")
    print(f"  máxima diferencia de costo   frente a (c): {cost_gap:.5f} "
          f"({rel_cost:.1%})")
    if decision_gap < 0.05 and rel_cost < 0.01:
        print("\n  El puente micro-macro NO cambia decisiones en este régimen.")
        print("  Antes de invertir en fidelidad del surrogate, buscar un régimen")
        print("  donde sí las cambie: transitorios rápidos de omega, o dureza")
        print("  variable simulada.")
    else:
        print("\n  El puente micro-macro SÍ cambia decisiones: un controlador que")
        print("  ignora el espectro elige otra omega y paga otro costo.")

    print("\n  Recordatorio de estatus (§4): la única acción con respaldo")
    print("  microdinámico es omega. F_feed entra solo por el balance")
    print("  macroscópico — no hay dato que ligue la alimentación al espectro.")

    out = Path(args.out) if args.out else (
        root / cfg["experiment"]["out_dir"] / "exp_E1_value_of_information.json")
    save_report(out, {
        "experiment": "E1",
        "n_steps": n_steps,
        "dissipation_calibration": cal,
        "shape_sensitivity": {k: v for k, v in sens.items() if k != "ratio_per_class"},
        "library": {"n_nodes": len(lib), "omega_range": lib.omega_range,
                    "branches": lib.branch_table()},
        "arms": {k: {kk: vv for kk, vv in v.items() if kk != "history"}
                 for k, v in results.items()},
        "histories": {k: v["history"] for k, v in results.items()},
        "comparison": table,
        "scaling": scaling.to_dict(),
    })
    print(f"\n  reporte → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
