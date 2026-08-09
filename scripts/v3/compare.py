"""Comparación controlada entre `v3-C`, `v3-I` y `GNSControlled`.

    python scripts/v3/compare.py \
        --experiment configs/experiments/gravity60_small.yaml \
        --models v3_c v3_i gns_controlled --seeds 0 1 2

Reglas de equidad implementadas aquí, no confiadas al operador (§16.1):

- exactamente los mismos splits, transiciones, sampler y cuotas;
- targets construidos una sola vez por la infraestructura común;
- mismas semillas y mismo presupuesto de actualizaciones y ejemplos;
- mejor checkpoint por validación, nunca por test;
- `CASE07` solo se evalúa al final, y solo si `--final-test` está activo;
- se reportan también parámetros y tiempos de entrenamiento e inferencia.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from _common import (  # noqa: E402
    REPO_ROOT, RunDirectory, build_model, load_yaml, make_manifest, make_run_id,
    prepare,
)
from slgnn_experiments.metrics import aggregate_seeds  # noqa: E402
from slgnn_experiments.runner import (  # noqa: E402
    evaluate_one_step, evaluate_rollout, train,
)

VARIANTS = {
    "v3_c": ("slgnn_v3", "configs/v3/mvp_c.yaml"),
    "v3_i": ("slgnn_v3", "configs/v3/mvp_i.yaml"),
    "gns_controlled": ("gns_controlled", "configs/gns/controlled.yaml"),
}


def run_one(variant: str, experiment: dict, exp_name: str, seed: int,
            out_root: Path, smoke: bool, final_test: bool) -> dict:
    model_key, cfg_path = VARIANTS[variant]
    model_cfg = load_yaml(REPO_ROOT / cfg_path).get("model")

    (scales, split, loaded, scene, train_ds, val_ds, test_ds,
     sampler, tcfg) = prepare(experiment, seed, smoke)

    torch.manual_seed(seed)
    model, resolved, profile = build_model(model_key, model_cfg)
    run_id = make_run_id(variant, profile, exp_name, seed)
    run = RunDirectory(out_root, run_id)
    run.write_environment()
    run.write_config({"variant": variant, "model_key": model_key,
                      "model": resolved, "experiment": experiment, "seed": seed})

    manifest = make_manifest(
        run_id=run_id, model_key=model_key, profile=profile, seed=seed, split=split,
        train_ds=train_ds, sampler=sampler, tcfg=tcfg, scales=scales, model=model,
        experiment_name=exp_name,
        notes="SMOKE" if smoke else "",
    )
    manifest.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    t0 = time.perf_counter()
    summary = train(model, scene, train_ds, val_ds, sampler, tcfg, run,
                    lambda u, v: run.save_checkpoint(
                        "best.pt", model, config=resolved, scales=scales.as_dict(),
                        extra={"update": u, "val": v}))
    manifest.duration_seconds = time.perf_counter() - t0
    run.save_checkpoint("last.pt", model, config=resolved, scales=scales.as_dict())
    manifest.write(run.path / "manifest.json")

    stride = max(1, len(train_ds.index) // 150)
    horizons = experiment.get("rollout_horizons", [1, 5, 10, 25])
    metrics: dict[str, float] = {}
    metrics.update({f"val/{k}": v for k, v in
                    evaluate_one_step(model, scene, val_ds, stride).items()})
    metrics.update({f"val/{k}": v for k, v in
                    evaluate_rollout(model, scene, val_ds, horizons).items()})
    if final_test and test_ds is not None:
        metrics.update({f"test/{k}": v for k, v in
                        evaluate_one_step(model, scene, test_ds, stride).items()})
        metrics.update({f"test/{k}": v for k, v in
                        evaluate_rollout(model, scene, test_ds, horizons).items()})

    total, trainable = model.n_parameters()
    metrics["n_parameters"] = total
    metrics["train_seconds"] = summary["train_seconds"]
    metrics["best_val_loss"] = summary["best_val_loss"] or float("nan")
    out = {"run_id": run_id, "variant": variant, "profile": profile, "seed": seed,
           "n_parameters": total, "n_trainable": trainable,
           "training": {k: v for k, v in summary.items() if k != "history"},
           "metrics": metrics}
    run.write_summary(out)
    return out


LR_GRID = (3e-3, 1e-3, 3e-4, 1e-4)


def select_lr(variant: str, experiment: dict, exp_name: str, out_root: Path,
              smoke: bool, seed: int = 0) -> tuple[float, dict[str, float]]:
    """Elige `lr` por validación con **presupuesto idéntico por familia**.

    La misma rejilla, la misma semilla y el mismo número de corridas para v3-C,
    v3-I y GNS. Sin esto la comparación sería tramposa en cualquiera de los dos
    sentidos: v3 arranca cerca del suelo de pérdida por sus priores
    estructurales y tolera mal un `lr` alto, mientras que GNS parte de cero y
    lo necesita. Fijar un solo `lr` para ambos penalizaría a uno de los dos por
    una razón que no tiene nada que ver con la arquitectura (§16.1).
    """
    scores: dict[str, float] = {}
    best_lr, best_score = LR_GRID[0], float("inf")
    for lr in LR_GRID:
        exp = json.loads(json.dumps(experiment))
        exp.setdefault("train", {})["lr"] = lr
        res = run_one(variant, exp, f"{exp_name}-lrsel", seed, out_root / "lr_selection",
                      smoke, final_test=False)
        score = res["metrics"].get("best_val_loss", float("inf"))
        scores[f"lr={lr:g}"] = score
        print(f"[compare]   {variant} lr={lr:g} -> val {score:.4g}", flush=True)
        if score < best_score:
            best_lr, best_score = lr, score
    return best_lr, scores


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", type=Path, required=True)
    ap.add_argument("--models", nargs="+", default=list(VARIANTS),
                    choices=list(VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tune-lr", action="store_true",
                    help="Rejilla de lr por validación, mismo presupuesto por familia.")
    ap.add_argument("--final-test", action="store_true",
                    help="Evalúa CASE07. Solo al cerrar la fase, nunca antes.")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results/v3_mvp/compare")
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    experiment = load_yaml(args.experiment)
    exp_name = experiment.get("name", args.experiment.stem)
    args.out.mkdir(parents=True, exist_ok=True)

    tuning: dict[str, dict] = {}
    results: dict[str, list[dict]] = {}
    for variant in args.models:
        exp = experiment
        if args.tune_lr:
            print(f"[compare] selección de lr para {variant} ...", flush=True)
            lr, scores = select_lr(variant, experiment, exp_name, args.out, args.smoke)
            tuning[variant] = {"selected_lr": lr, "grid": scores}
            exp = json.loads(json.dumps(experiment))
            exp.setdefault("train", {})["lr"] = lr
            print(f"[compare] {variant}: lr elegido {lr:g}", flush=True)
        results[variant] = []
        for seed in args.seeds:
            print(f"[compare] {variant} seed={seed} ...", flush=True)
            results[variant].append(
                run_one(variant, exp, exp_name, seed, args.out,
                        args.smoke, args.final_test)
            )

    aggregated = {
        v: aggregate_seeds([r["metrics"] for r in runs])
        for v, runs in results.items()
    }
    payload = {
        "experiment": exp_name,
        "seeds": args.seeds,
        "final_test_evaluated": args.final_test,
        "smoke": args.smoke,
        "lr_tuning": tuning or "no ejecutada: lr fijo del experimento",
        "lr_grid": list(LR_GRID) if args.tune_lr else None,
        "runs": {v: [r["run_id"] for r in runs] for v, runs in results.items()},
        "per_variant": aggregated,
    }
    path = args.out / f"comparison_{exp_name}.json"
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"\n[compare] {path}")
    _print_table(aggregated)
    return 0


def _print_table(agg: dict) -> None:
    keys = ["val/dp_rmse_all", "val/dp_rmse_free", "val/dp_rmse_pp",
            "val/dp_rmse_pw", "val/dp_rmse_mixed", "val/rollout_q_rmse_h1",
            "val/rollout_q_rmse_h25", "n_parameters", "train_seconds"]
    header = f"{'metric':<28}" + "".join(f"{v:>22}" for v in agg)
    print(header)
    print("-" * len(header))
    for k in keys:
        row = f"{k:<28}"
        for v in agg:
            e = agg[v].get(k)
            row += f"{'—':>22}" if e is None else f"{e['mean']:>13.4g}+-{e['std']:<7.2g}"
        print(row)


if __name__ == "__main__":
    raise SystemExit(main())
