"""Experimento E2 — ¿reproduce el rollout de SLGNN el espectro? (§8)

Este es el experimento que demuestra el contrato central del banco de pruebas:
**`C_phi` no sabe de dónde vienen las trayectorias.** El mismo operador que
produjo la biblioteca a partir de MFiX se aplica ahora a un rollout de SLGNN sin
cambiar una línea — T3 sustituye la fuente, no el operador.

Dos etapas, deliberadamente separadas (§8):

  E2-a  **en distribución**: 60 esferas con gravedad en caja, donde hay verdad
        de referencia y el modelo está entrenado. Aísla la fidelidad del
        surrogate.
  E2-b  **extrapolación**: cilindro rotatorio de 2073 esferas. La diferencia
        entre E2-b y E2-a es el costo de la extrapolación geométrica, que es
        precisamente la tesis de SLGNN.

Evaluar directamente en el cilindro mezclaría ambas fuentes de error, y el
resultado no sería atribuible.

ADVERTENCIA sobre E2-b: `slgnn.sdf.dynamical_cylinder_omega` implementa un
perfil de omega que los datos refutan (ver `twin/cylinder.py` y el paso 0 de
`exp_H_hysteresis.py`). Mientras no se corrija, un rollout sobre el cilindro
usará una velocidad de pared equivocada para t > 1 s, y E2-b medirá esa
discrepancia además de la extrapolación. Este script usa el perfil verificado
para la SDF del rollout y lo informa.

Uso:
    python experiments/twin/exp_E2_spectrum_estimators.py \
        --checkpoint checkpoints/slgnn_v2/gravity_rollout/xxx.pt
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from slgnn.data import default_scales, load_case  # noqa: E402
from slgnn.experiment import load_checkpoint  # noqa: E402
from slgnn.integrator import rollout  # noqa: E402
from slgnn.sdf import BoxSDF  # noqa: E402
from slgnn.state import Particles  # noqa: E402
from twin.coarse import coarse_grain  # noqa: E402
from twin.cylinder import make_cylinder_sdf  # noqa: E402
from twin.harness import enable_utf8_stdout, save_report, spectrum_distance  # noqa: E402
from twin.pipeline import (coarse_config, load_config, project_root,  # noqa: E402
                           resolve_kappa)
from twin.units import Scaling  # noqa: E402


class _Traj:
    """Trayectoria mínima con la interfaz que consume `C_phi`."""

    def __init__(self, q, v, omega, m, radii, dt):
        self.q, self.v, self.omega = q, v, omega
        self.m, self.radii, self.dt = m, radii, dt


def _rollout_traj(model, truth, wall, g_vec, n_steps, start=0):
    particles = Particles.uniform(
        truth.q.shape[1], m=float(truth.m[0]), radius=float(truth.radii[0]),
        dtype=truth.q.dtype,
    )
    q, v, w = rollout(
        model, truth.q[start].clone(), truth.v[start].clone(),
        truth.omega[start].clone(), particles, wall, dt=float(truth.dt),
        n_steps=n_steps, t0=start * float(truth.dt),
        g_vec=torch.as_tensor(g_vec, dtype=truth.q.dtype),
    )
    return _Traj(q, v, w, truth.m, truth.radii, truth.dt)


def _slice(traj, start, n):
    return _Traj(traj.q[start:start + n + 1], traj.v[start:start + n + 1],
                 traj.omega[start:start + n + 1], traj.m, traj.radii, traj.dt)


def _compare(name, model, truth, wall, g_vec, scaling, ccfg, n_steps, start):
    print(f"\n  --- {name} ---")
    t0 = time.time()
    pred = _rollout_traj(model, truth, wall, g_vec, n_steps, start)
    ref = _slice(truth, start, n_steps)
    print(f"    rollout de {n_steps} pasos sobre {truth.q.shape[1]} partículas "
          f"[{time.time() - t0:.1f} s]")

    t_start = start * float(truth.dt)
    f_ref = coarse_grain(ref, wall, scaling, ccfg, g_vec=g_vec, t0=t_start)
    f_pred = coarse_grain(pred, wall, scaling, ccfg, g_vec=g_vec, t0=t_start)

    dist = spectrum_distance(f_pred.spectrum, f_ref.spectrum)
    drift = float(torch.linalg.vector_norm(pred.q[-1] - ref.q[-1], dim=-1).mean())
    print(f"    eventos   MFiX={f_ref.n_events:6d}   SLGNN={f_pred.n_events:6d}")
    print(f"    tasa      MFiX={f_ref.spectrum.total_rate():.4e}   "
          f"SLGNN={f_pred.spectrum.total_rate():.4e}")
    print(f"    E_media   MFiX={f_ref.spectrum.mean_energy():.4e}   "
          f"SLGNN={f_pred.spectrum.mean_energy():.4e}")
    print(f"    W1 = {dist['wasserstein_decades']:.4f} décadas   "
          f"err_tasa = {dist['rate_rel_error']:.3f}")
    print(f"    deriva cinemática media al final del rollout = {drift:.4f} "
          "(diámetros)")
    return {"n_events_truth": f_ref.n_events, "n_events_pred": f_pred.n_events,
            "rate_truth": f_ref.spectrum.total_rate(),
            "rate_pred": f_pred.spectrum.total_rate(),
            "kinematic_drift_diameters": drift, **dist}


def main() -> int:
    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/twin/twin_toy.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--start", type=int, default=100)
    ap.add_argument("--skip-extrapolation", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = project_root()
    cfg = load_config(root / args.config)

    ckpt = args.checkpoint
    if ckpt is None:
        found = sorted((root / "checkpoints").rglob("*.pt"))
        ckpt = found[-1] if found else None
    if ckpt is None:
        print("E2 pertenece a T3 y necesita un checkpoint de SLGNN entrenado.")
        print("No se encontró ninguno en checkpoints/. Entrenar primero con:")
        print("  python scripts/slgnn_v2/train.py --config configs/slgnn_v2/gravity_rollout.yaml")
        print("\nT0-T2 (exp_H, exp_E1) no dependen de que la red entrene y ya")
        print("corren: ése es el punto del ordenamiento de hitos.")
        return 0

    print("=" * 78)
    print(f"E2 — checkpoint: {ckpt}")
    print("=" * 78)
    model, meta = load_checkpoint(Path(ckpt))
    cal = resolve_kappa(cfg, root)
    ccfg = coarse_config(cfg, kappa=cal["kappa"])
    scales = default_scales()
    scaling = Scaling.from_slgnn(scales)
    results = {}

    # ------------------------------------------------------------- E2-a
    print("\nE2-a — EN DISTRIBUCIÓN (60 esferas con gravedad, caja estática)")
    base = root / cfg["data"]["root"] / "60Spheres_Gravity_Inside_Cuboidal_Enclosure"
    truth = scales.nondim(load_case(base / "CASE07", dt=1e-4, dtype=torch.float32))
    side = scales.length(float(cfg["data"]["calibration_box_m"]))
    wall = BoxSDF([0.0] * 3, [side] * 3)
    g = np.zeros(3)
    g[1] = -scales.gravity(float(cfg["data"]["gravity"]))
    results["E2a_in_distribution"] = _compare(
        "CASE07 (held-out)", model, truth, wall, g, scaling, ccfg,
        args.steps, args.start)

    # ------------------------------------------------------------- E2-b
    if not args.skip_extrapolation:
        print("\nE2-b — EXTRAPOLACIÓN (2073 esferas, cilindro rotatorio)")
        print("  NOTA: la SDF usa el perfil de omega VERIFICADO ('pdf_literal'),")
        print("        no el de slgnn.sdf, que los datos refutan.")
        cyl = scales.nondim(load_case(
            root / cfg["data"]["root"] / cfg["data"]["cylinder_case"],
            dt=float(cfg["data"]["cylinder_dt"]), dtype=torch.float32))
        sdf = make_cylinder_sdf(scaling, cfg["data"].get("omega_profile",
                                                         "pdf_literal"))
        results["E2b_extrapolation"] = _compare(
            "CASE08 (cilindro)", model, cyl, sdf, g, scaling, ccfg,
            min(args.steps, 50), args.start)

    # ---------------------------------------------------------- atribución
    print("\n" + "=" * 78)
    print("ATRIBUCIÓN DEL ERROR")
    print("=" * 78)
    a = results.get("E2a_in_distribution")
    b = results.get("E2b_extrapolation")
    if a and b:
        gap = b["wasserstein_decades"] - a["wasserstein_decades"]
        print(f"  W1 en distribución  : {a['wasserstein_decades']:.4f} décadas")
        print(f"  W1 extrapolando     : {b['wasserstein_decades']:.4f} décadas")
        print(f"  costo de extrapolar : {gap:+.4f} décadas")
        print("\n  Lectura (§19.5, no identificabilidad energética): si el espectro")
        print("  del rollout es bueno y una cabeza aprendida es mala, el problema")
        print("  es de supervisión; si ambos fallan con la cinemática estable, el")
        print("  problema es estructural y afecta a H2.")
    elif a:
        print(f"  Solo E2-a: W1 = {a['wasserstein_decades']:.4f} décadas.")

    out = Path(args.out) if args.out else (
        root / cfg["experiment"]["out_dir"] / "exp_E2_spectrum_estimators.json")
    save_report(out, {"experiment": "E2", "checkpoint": str(ckpt),
                      "dissipation_calibration": cal, "results": results,
                      "scaling": scaling.to_dict()})
    print(f"\n  reporte → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
