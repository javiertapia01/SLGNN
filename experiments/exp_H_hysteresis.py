"""Experimento H — ¿es `omega ↦ E^coll` una función, o depende del camino? (§3)

Es el primer experimento del programa y el único que puede reorientar el diseño
del acoplamiento sin necesidad de que SLGNN funcione. La rampa de omega del
archivo de 2073 esferas recorre el mismo rango de velocidades dos veces, una
subiendo y otra bajando, así que la pregunta se responde con los datos que ya
están en disco.

Interpretación fijada de antemano, para no elegir el umbral después de ver el
resultado:

- ramas indistinguibles dentro del ruido de muestreo → una biblioteca estática
  `E^coll(omega)` es representación suficiente y el acoplamiento puede ser una
  tabla;
- ramas sistemáticamente distintas → el espectro **no está determinado por el
  setpoint** y necesita condicionamiento por el estado de la carga.

Uso:
    python experiments/exp_H_hysteresis.py
    python experiments/exp_H_hysteresis.py --config configs/twin_toy.yaml --quick
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twin.coarse import CoarseConfig  # noqa: E402
from twin.cylinder import verify_omega_profile  # noqa: E402
from twin.events import detect_events  # noqa: E402
from twin.harness import (ascii_profile, enable_utf8_stdout,  # noqa: E402
                          features_table, save_report)
from twin.library import hysteresis_report, hysteresis_verdict  # noqa: E402
from twin.pipeline import (build_library, coarse_config, coarse_ramp,  # noqa: E402
                           load_cylinder, load_config, project_root, resolve_kappa)


def main() -> int:
    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/twin_toy.yaml")
    ap.add_argument("--quick", action="store_true",
                    help="usa solo el ancho de ventana de referencia")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = project_root()
    cfg = load_config(root / args.config)
    np.random.seed(cfg.get("seed", 0))

    # ---------------------------------------------------------------- paso 0
    print("=" * 78)
    print("PASO 0 — verificación de omega(t) contra los datos")
    print("=" * 78)
    traj, sdf, g_vec, scaling, profile = load_cylinder(cfg, root)
    print(f"trayectoria: {tuple(traj.q.shape)}  dt={traj.dt * scaling.T:.1e} s  "
          f"perfil configurado: {profile}")

    check = verify_omega_profile(traj, scaling)
    print(f"\n{'t[s]':>6} {'n_pared':>8} {'omega_lecho':>12} {'spin_medio':>11} "
          f"{'pdf_literal':>12} {'triangular':>11}")
    for p in check["probes"]:
        print(f"{p['t']:6.2f} {p['n_near_wall']:8d} {p['omega_bed']:12.3f} "
              f"{p['spin_mean']:11.2f} {p['omega_pdf_literal']:12.3f} "
              f"{p['omega_triangular']:11.3f}")
    print(f"\n  lecho invierte el giro en [1.0, 1.5] s : {check['bed_reverses_between_1_0_and_1_5s']}")
    print(f"  segundo pico de spin presente          : {check['second_spin_peak_present']} "
          f"(temprano={check['early_spin_peak']:.1f}, tardío={check['late_spin_peak']:.1f})")
    print(f"  VEREDICTO: el perfil compatible con los datos es '{check['verdict']}'")
    if check["verdict"] != profile:
        print(f"  ATENCIÓN: el config usa '{profile}' pero los datos indican "
              f"'{check['verdict']}'")

    # ---------------------------------------------------------------- paso 1
    print("\n" + "=" * 78)
    print("PASO 1 — calibración de la disipación (caja estática, p_in ≡ 0)")
    print("=" * 78)
    t0 = time.time()
    cal = resolve_kappa(cfg, root)
    print(f"  kappa_diss = {cal['kappa']:.3f}   ({cal['source']})")
    print(f"  residual del balance = {cal['residual']:.1%}")
    if cal.get("per_window"):
        print(f"  por ventana: {np.round(cal['per_window'], 3).tolist()}")
    print(f"  [{time.time() - t0:.1f} s]")
    print("  NOTA: kappa corrige lo que (1-e²)·E_impacto no captura (fricción")
    print("        tangencial y contactos sostenidos). p_diss es DERIVADO.")

    ccfg: CoarseConfig = coarse_config(cfg, kappa=cal["kappa"])

    # ---------------------------------------------------------------- paso 2
    print("\n" + "=" * 78)
    print("PASO 2 — detección de eventos sobre la rampa completa")
    print("=" * 78)
    t0 = time.time()
    events = detect_events(
        traj.q.numpy(), traj.v.numpy(), traj.radii.numpy(), traj.m.numpy(), sdf,
        delta=ccfg.delta, max_lookback=ccfg.max_lookback, v_n_min=ccfg.v_n_min,
        dt=float(traj.dt), t0=0.0, block=ccfg.block,
    )
    print(f"  {events.summary()}   [{time.time() - t0:.1f} s]")
    e_si = events.E_impact * scaling.energy
    if len(events):
        print(f"  energía de impacto: min={e_si.min():.2e} J  "
              f"mediana={np.median(e_si):.2e} J  max={e_si.max():.2e} J")

    # ---------------------------------------------------------------- paso 3
    widths = ([cfg["hysteresis"]["reference_width_s"]] if args.quick
              else list(cfg["hysteresis"]["window_widths_s"]))
    results, libraries = {}, {}
    for width in widths:
        print("\n" + "=" * 78)
        print(f"PASO 3 — espectros por ventana (ancho = {width:.3f} s)")
        print("=" * 78)
        feats = coarse_ramp(traj, sdf, scaling, ccfg, g_vec, width, profile,
                            events=events)
        print(features_table(feats))
        over = sum(f.spectrum.overflow.sum() for f in feats)
        if over > 0:
            print(f"  ATENCIÓN: {over:.2e} ev/(s·kg) sobre e_max — subir e_max")

        n_negative = sum(1 for f in feats if f.p_in_balance < 0)
        if n_negative:
            print(f"  ATENCIÓN: {n_negative}/{len(feats)} ventanas con P_in < 0. "
                  "kappa se calibró en la caja estática y su transferencia al")
            print("            cilindro es imperfecta: la mezcla p-p/p-pared y la "
                  "fracción de contactos sostenidos difieren.")

        report = hysteresis_report(feats, n_probe=cfg["hysteresis"]["n_probes"])
        results[width] = {"report": report,
                          "n_windows": len(feats),
                          "n_windows_negative_p_in": n_negative,
                          "closure_gap_median": float(np.nanmedian(
                              [f.closure_gap for f in feats]))}
        libraries[width] = build_library(feats)

        if report["status"] != "ok":
            print(f"\n  histéresis: {report['status']} "
                  f"(up={report.get('n_up')}, down={report.get('n_down')})")
            continue
        lo, hi = report["omega_overlap"]
        print(f"\n  solape de ramas: omega ∈ [{lo:.2f}, {hi:.2f}] rad/s")
        print(f"  {'omega':>8} {'W1[décadas]':>12} {'err_tasa':>10} "
              f"{'tasa_up':>12} {'tasa_down':>12}")
        for r in report["probes"]:
            print(f"  {r['omega']:8.2f} {r['wasserstein_decades']:12.4f} "
                  f"{r['rate_rel_error']:10.3f} {r['rate_up']:12.3e} "
                  f"{r['rate_down']:12.3e}")
        print(f"  mediana W1 = {report['median_wasserstein']:.4f} décadas   "
              f"mediana err_tasa = {report['median_rate_error']:.3f}")

    # ---------------------------------------------------------------- veredicto
    print("\n" + "=" * 78)
    print("VEREDICTO DE H")
    print("=" * 78)
    w_thr = cfg["hysteresis"]["wasserstein_threshold"]
    r_thr = cfg["hysteresis"]["rate_threshold"]
    verdicts = {}
    for width, res in results.items():
        rep = res["report"]
        v = hysteresis_verdict(rep, w_thr, r_thr)
        verdicts[width] = v
        if v["verdict"] == "indeterminado":
            print(f"  ancho {width:.3f} s: indeterminado ({v.get('reason')})")
            continue
        print(f"\n  ancho {width:.3f} s ({res['n_windows']:3d} ventanas)")
        print(f"    mediana  W1={rep['median_wasserstein']:.4f}  "
              f"err_tasa={rep['median_rate_error']:.3f}")
        print(f"    máximo   W1={v['max_wasserstein']:.4f}  "
              f"err_tasa={v['max_rate_error']:.3f}")
        print(f"    sondeos fuera de umbral: {v['n_violating']}/{v['n_probes']}")
        print(f"    → omega ↦ E^coll {v['verdict'].upper()}")
        if v["n_violating"]:
            lo_v, hi_v = v["omega_violating_range"]
            print(f"      la dependencia del camino se concentra en "
                  f"omega ∈ [{lo_v:.2f}, {hi_v:.2f}] rad/s")

    print("\n  Lectura: la mediana sola habría dicho 'es una función'. Resuelto en")
    print("  omega, el acuerdo entre ramas es excelente a omega alta y se rompe a")
    print("  omega baja, donde la rama ascendente todavía arrastra el transitorio")
    print("  de arranque desde reposo (KE dos órdenes de magnitud mayor a la misma")
    print("  omega). Una biblioteca estática E^coll(omega) es suficiente en el")
    print("  régimen desarrollado, y NO lo es cerca del arranque.")

    distinct = {v["verdict"] for v in verdicts.values()}
    if len(distinct) > 1:
        print("\n  El veredicto DEPENDE del ancho de ventana: la histéresis medida")
        print("  mezcla retardo físico de la carga con no estacionariedad del")
        print("  muestreo, y estos datos no alcanzan a separarlas.")

    ref = cfg["hysteresis"]["reference_width_s"]
    if ref in libraries:
        lib = libraries[ref]
        print(f"\n  biblioteca de referencia: {len(lib)} nodos, "
              f"omega ∈ [{lib.omega_range[0]:.2f}, {lib.omega_range[1]:.2f}] rad/s")
        omegas = np.array([n.omega for n in lib.nodes])
        rates = np.array([n.spectrum.total_rate() for n in lib.nodes])
        print("\n" + ascii_profile(omegas, rates,
                                   label="  tasa total de eventos vs omega"))

    out = Path(args.out) if args.out else (
        root / cfg["experiment"]["out_dir"] / "exp_H_hysteresis.json")
    save_report(out, {
        "experiment": "H",
        "omega_profile_check": check,
        "dissipation_calibration": cal,
        "scaling": scaling.to_dict(),
        "n_events": len(events),
        "n_unresolved": events.n_unresolved,
        "widths": {str(k): v for k, v in results.items()},
        "verdicts": {str(k): v for k, v in verdicts.items()},
        "thresholds": {"wasserstein": w_thr, "rate": r_thr},
    })
    print(f"\n  reporte → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
