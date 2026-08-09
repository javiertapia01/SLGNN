"""Evaluación de un checkpoint entrenado, con métricas por régimen y rollout.

    python scripts/v3/evaluate.py --run results/v3_mvp/runs/<run_id> \
        --cases CASE06 --horizons 1 5 10 25

Por defecto **no** evalúa `CASE07`: hay que pedirlo explícitamente con
`--allow-extrapolation`, y solo tiene sentido una vez, al cerrar la fase.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from _common import REPO_ROOT, build_model, prepare  # noqa: E402
from slgnn_experiments.checkpointing import load_checkpoint  # noqa: E402
from slgnn_experiments.nondimensionalization import default_scales  # noqa: E402
from slgnn_experiments.runner import (  # noqa: E402
    Dataset, evaluate_one_step, evaluate_rollout,
)
from slgnn_experiments.scene import build_scene  # noqa: E402
from slgnn_experiments.splits import Split, load_split  # noqa: E402

EXTRAPOLATION = {("sixty_gravity", "CASE07")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--cases", nargs="+", default=None)
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 10, 25])
    ap.add_argument("--stride", type=int, default=0)
    ap.add_argument("--allow-extrapolation", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    cfg = yaml.safe_load((args.run / "config_resolved.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    model_key = cfg.get("model_key")
    experiment = cfg["experiment"]
    dataset = experiment["data"]["dataset"]

    cases = args.cases or ([experiment["data"]["val_case"]]
                           if experiment["data"].get("val_case") else [])
    for c in cases:
        if (dataset, c) in EXTRAPOLATION and not args.allow_extrapolation:
            raise SystemExit(
                f"{dataset}/{c} es el caso de extrapolación. Evaluarlo requiere "
                "--allow-extrapolation y solo debe hacerse una vez, al cerrar la "
                "fase; nunca para elegir hiperparámetros ni checkpoints."
            )

    model, resolved, profile = build_model(model_key, cfg["model"])
    ck = load_checkpoint(args.run / args.checkpoint, type(model).__name__)
    model.load_state_dict(ck["model"])
    model.eval()

    scales = default_scales()
    scene = build_scene(dataset, scales)
    split = Split(dataset, tuple(cases), None)
    loaded = load_split(
        split, REPO_ROOT, scales,
        max_steps=experiment["data"].get("max_steps"),
        frame_start=int(experiment["data"].get("frame_start", 0)),
        frame_stop=experiment["data"].get("frame_stop"),
    )
    data = Dataset.build(loaded.train, scene)
    stride = args.stride or max(1, len(data.index) // 200)

    out = {
        "run": str(args.run), "checkpoint": args.checkpoint, "model": model_key,
        "profile": profile, "seed": manifest.get("seed"), "cases": cases,
        "n_parameters": model.n_parameters()[0],
        "one_step": evaluate_one_step(model, scene, data, stride),
        "rollout": evaluate_rollout(model, scene, data, args.horizons),
    }
    path = args.out or (args.run / f"evaluation_{'_'.join(cases)}.json")
    path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(json.dumps(out, indent=2, default=float))
    print(f"\n[evaluate] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
