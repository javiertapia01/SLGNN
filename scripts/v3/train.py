"""Entrenamiento de SLGNN-v3 y de los baselines GNS, con el mismo protocolo.

    python scripts/v3/train.py --model slgnn_v3 --config configs/v3/mvp_c.yaml \
        --experiment configs/experiments/micro_overfit.yaml --smoke

`--smoke` termina en segundos, toca forward, backward, checkpoint y
evaluación, y **no** es un resultado científico: el manifiesto lo marca.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from _common import (  # noqa: E402
    MODEL_KEYS, REPO_ROOT, RunDirectory, build_model, load_yaml, make_manifest,
    make_run_id, prepare,
)
from slgnn_experiments.runner import (  # noqa: E402
    evaluate_one_step, evaluate_rollout, train,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=MODEL_KEYS, required=True)
    ap.add_argument("--config", type=Path, required=True,
                    help="Configuración del modelo (perfil v3 o GNS).")
    ap.add_argument("--experiment", type=Path, required=True,
                    help="Configuración experimental: datos, sampler, presupuesto.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results/v3_mvp/runs")
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    model_cfg_raw = load_yaml(args.config)
    experiment = load_yaml(args.experiment)
    exp_name = experiment.get("name", args.experiment.stem)

    (scales, split, loaded, scene, train_ds, val_ds, test_ds,
     sampler, tcfg) = prepare(experiment, args.seed, args.smoke)

    torch.manual_seed(args.seed)
    model, resolved_model_cfg, profile = build_model(
        args.model, model_cfg_raw.get("model")
    )

    run_id = make_run_id(args.model, profile, exp_name, args.seed)
    run = RunDirectory(args.out, run_id)
    tee = run.tee_stdout()
    tee.__enter__()
    run.write_environment()
    run.write_config({
        "model_key": args.model, "model": resolved_model_cfg,
        "experiment": experiment, "seed": args.seed, "smoke": args.smoke,
    })

    manifest = make_manifest(
        run_id=run_id, model_key=args.model, profile=profile, seed=args.seed,
        split=split, train_ds=train_ds, sampler=sampler, tcfg=tcfg, scales=scales,
        model=model, experiment_name=exp_name,
        notes="SMOKE RUN: no es un resultado cientifico" if args.smoke else "",
    )
    manifest.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    total, trainable = model.n_parameters()
    print(f"[train] {run_id}")
    print(f"[train] parámetros: {total} ({trainable} entrenables)")
    print(f"[train] transiciones disponibles: {len(train_ds.index)}")
    print(f"[train] composición del índice: {train_ds.index.composition()}")
    print(f"[train] cuotas efectivas: {sampler.effective_quotas}")
    if sampler.dropped_strata:
        print(f"[train] estratos sin datos (cuota redistribuida): "
              f"{sampler.dropped_strata}")

    def save_best(update, val):
        run.save_checkpoint("best.pt", model, config=resolved_model_cfg,
                            scales=scales.as_dict(),
                            extra={"update": update, "val": val})

    t0 = time.perf_counter()
    summary = train(model, scene, train_ds, val_ds, sampler, tcfg, run, save_best)
    manifest.duration_seconds = time.perf_counter() - t0
    run.save_checkpoint("last.pt", model, config=resolved_model_cfg,
                        scales=scales.as_dict())
    manifest.write(run.path / "manifest.json")

    stride = max(1, len(train_ds.index) // 200)
    evals = {"train_one_step": evaluate_one_step(model, scene, train_ds, stride)}
    if val_ds is not None:
        evals["val_one_step"] = evaluate_one_step(model, scene, val_ds, stride)
        horizons = experiment.get("rollout_horizons")
        if horizons:
            evals["val_rollout"] = evaluate_rollout(model, scene, val_ds, horizons)

    out = {
        "run_id": run_id, "model": args.model, "profile": profile,
        "seed": args.seed, "smoke": args.smoke,
        "n_parameters": total, "n_trainable": trainable,
        "training": {k: v for k, v in summary.items() if k != "history"},
        "evaluation": evals,
    }
    run.write_summary(out)
    print(f"[train] pérdida de validación (mejor): {summary['best_val_loss']}")
    print(f"[train] {run.path / 'summary.json'}")
    tee.__exit__()
    print(json.dumps(
        {k: v for k, v in evals.get("val_one_step", {}).items()
         if k.startswith(("dp_rmse", "dL_rmse"))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
