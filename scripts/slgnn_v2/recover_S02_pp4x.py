"""Recuperación segura de PP-4x tras la consolidación S02.

Parte de best-joint.pt, restaura únicamente las cabezas PP desde el especialista
S00 y las readapta al backbone consolidado. El backbone y las cabezas de pared
permanecen congelados, por lo que la competencia PW1 no puede cambiar.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from train_S02_joint_consolidation import (
    ROOT,
    backward_rollout,
    load_checkpoint_model,
    pp_batch_loss,
    rollout_starts,
    save_checkpoint,
    set_trainable,
    suite_metrics,
)
from train_benchmark_wall_transfer import (
    active_indices,
    compute_scales,
    load_data,
    make_particles,
    pp_near_indices,
    randint,
)


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def evaluate(model, wall_data, pp_data, wall, cfg):
    threshold = float(cfg["evaluation"]["active_acceleration_threshold"])
    summary, wall_rows, pp_rows = suite_metrics(
        model, wall_data, pp_data, wall, threshold
    )
    return summary, wall_rows, pp_rows


def case_score(pp_rows, reference_rows):
    reference = {r["case"]: r for r in reference_rows}
    ratios = {
        r["case"]: r["rollout_v_rmse_post"]
        / max(reference[r["case"]]["rollout_v_rmse_post"], 1e-12)
        for r in pp_rows
    }
    return max(ratios.values()), ratios


def write_report(path, cfg, evaluations, best_tag, best_score):
    critical = str(cfg["recovery"]["critical_case"])
    limit = float(cfg["recovery"]["max_case_ratio_to_specialist"])
    best = next(e for e in evaluations if e["tag"] == best_tag)
    critical_row = next(r for r in best["pp_rows"] if r["case"] == critical)
    ref_row = next(r for r in evaluations[0]["reference_rows"] if r["case"] == critical)
    approved = best_score <= limit
    lines = [
        "# S02-R1 — Recuperación PP-4x",
        "",
        "El backbone y todas las cabezas PW permanecieron congelados. Sólo se modificaron las cabezas PP.",
        "",
        "| etapa | RMSE v pared mediana | RMSE v PP mediana | RMSE v 4x | peor razón vs especialista |",
        "|---|---:|---:|---:|---:|",
    ]
    for e in evaluations:
        row4 = next(r for r in e["pp_rows"] if r["case"] == critical)
        lines.append(
            f"| {e['tag']} | {e['summary']['wall_v_median']:.4g} | "
            f"{e['summary']['pp_v_median']:.4g} | {row4['rollout_v_rmse_post']:.4g} | "
            f"{e['score']:.4g} |"
        )
    lines += [
        "",
        f"- Mejor etapa: `{best_tag}`.",
        f"- 4x especialista: {ref_row['rollout_v_rmse_post']:.4g}.",
        f"- 4x recuperado: {critical_row['rollout_v_rmse_post']:.4g}.",
        f"- Criterio por caso (≤{limit:.2f}× especialista): **{'APROBADO' if approved else 'NO APROBADO'}**.",
        (
            "- Checkpoint promovido: `best-joint-recovered.pt`."
            if approved else
            "- `best-joint-recovered.pt` se conserva como mejor compromiso de "
            "recuperación, pero no se promueve todavía a H60."
        ),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/slgnn_v2/curriculum/S02_joint_pp2o_pw1.yaml")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    rc = cfg["recovery"]
    if args.quick:
        rc["one_step_iters"] = 10
        rc["rollout_horizons"] = [4]
        rc["rollout_iterations"] = [2]

    run_id = cfg["run_id"] + ("__recovery-quick" if args.quick else "")
    base_ckpt_dir = resolve(cfg["output"]["checkpoint_root"]) / cfg["run_id"]
    ckpt_dir = resolve(cfg["output"]["checkpoint_root"]) / run_id
    result_dir = resolve(cfg["output"]["result_root"]) / run_id
    source = base_ckpt_dir / rc["source_checkpoint"]
    specialist_path = resolve(rc["specialist_checkpoint"])
    wall_data, pp_data, wall = load_data(cfg)
    model, _ = load_checkpoint_model(source, cfg)
    specialist, _ = load_checkpoint_model(specialist_path, cfg)

    # Injerto seguro: sólo cabezas PP; pared queda matemáticamente invariante.
    state = model.state_dict()
    specialist_state = specialist.state_dict()
    for key in state:
        if key.startswith("head_pp_"):
            state[key] = specialist_state[key].detach().clone()
    model.load_state_dict(state)
    set_trainable(model, ["head_pp_"])
    parameters = [p for p in model.parameters() if p.requires_grad]

    threshold = float(cfg["evaluation"]["active_acceleration_threshold"])
    pp_scales = compute_scales(
        pp_data, list(pp_data),
        lambda tr: pp_near_indices(tr, float(cfg["model"]["r_off"])),
    )
    pp_particles = make_particles(next(iter(pp_data.values())))
    generator = torch.Generator().manual_seed(int(cfg["seed"]) + 30_000)
    history, evaluations = [], []
    _, _, reference_rows = evaluate(specialist, wall_data, pp_data, wall, cfg)

    def record(tag, optimizer=None):
        summary, wall_rows, pp_rows = evaluate(model, wall_data, pp_data, wall, cfg)
        score, ratios = case_score(pp_rows, reference_rows)
        item = {
            "tag": tag, "summary": summary, "wall_rows": wall_rows,
            "pp_rows": pp_rows, "reference_rows": reference_rows,
            "score": score, "case_ratios": ratios,
        }
        evaluations.append(item)
        save_checkpoint(
            ckpt_dir / f"stage-recovery-{tag}.pt", model, optimizer,
            cfg, f"recovery-{tag}", {**summary, "case_score": score},
        )
        return item

    print("== S02-R1: cabezas PP del especialista ==")
    record("hybrid")

    optimizer = torch.optim.Adam(parameters, lr=float(rc["lr_one_step"]))
    sequence = [str(x) for x in rc["case_sequence"]]
    model.train()
    for it in range(int(rc["one_step_iters"])):
        optimizer.zero_grad(set_to_none=True)
        loss = pp_batch_loss(
            model, pp_data, sequence, pp_particles, pp_scales, cfg,
            int(rc["batch_size"]), float(rc["active_probability"]),
            generator, it,
        )
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()
        history.append({
            "phase": "recovery-one-step", "iteration": it,
            "loss": float(loss.detach()), "grad_norm": float(grad),
        })
        if it % 100 == 0 or it == int(rc["one_step_iters"]) - 1:
            print(f"  one-step {it:4d}/{rc['one_step_iters']} loss={float(loss.detach()):.5f}", flush=True)
    record("one-step", optimizer)

    starts_cache = {}
    for horizon, iterations in zip(rc["rollout_horizons"], rc["rollout_iterations"]):
        optimizer = torch.optim.Adam(parameters, lr=float(rc["lr_rollout"]))
        starts_cache = {
            c: rollout_starts(tr, int(horizon), threshold)
            for c, tr in pp_data.items()
        }
        model.train()
        for it in range(int(iterations)):
            case = sequence[it % len(sequence)]
            choices = starts_cache[case]
            k0 = choices[randint(len(choices), generator)]
            optimizer.zero_grad(set_to_none=True)
            loss = backward_rollout(
                model, pp_data[case], pp_particles, None, k0, int(horizon),
                pp_scales, float(cfg["rollout"]["sigma_q"]),
                int(cfg["rollout"]["tbptt_chunk"]), 1.0,
                float(cfg["rollout"]["noise_q"]),
                float(cfg["rollout"]["noise_v"]), generator,
            )
            grad = torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            history.append({
                "phase": f"recovery-H{horizon}", "iteration": it,
                "loss": loss, "grad_norm": float(grad),
            })
            if it % 25 == 0 or it == int(iterations) - 1:
                print(f"  H={horizon:2d} {it:3d}/{iterations} loss={loss:.5f}", flush=True)
        record(f"H{horizon}", optimizer)

    best = min(evaluations, key=lambda x: x["score"])
    best_stage = ckpt_dir / f"stage-recovery-{best['tag']}.pt"
    best_path = ckpt_dir / "best-joint-recovered.pt"
    best_path.write_bytes(best_stage.read_bytes())
    (result_dir / "recovery_evaluations.json").write_text(
        json.dumps(evaluations, indent=2), encoding="utf-8"
    )
    with (result_dir / "recovery_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader(); writer.writerows(history)
    write_report(
        result_dir / "RECOVERY_PP4X.md", cfg, evaluations,
        best["tag"], best["score"],
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    tags = [e["tag"] for e in evaluations]
    for case in pp_data:
        ax.plot(tags, [next(r for r in e["pp_rows"] if r["case"] == case)["rollout_v_rmse_post"] for e in evaluations], "o-", label=case)
    ax.set_ylabel("RMSE velocidad post-choque"); ax.set_title("Recuperación PP por caso")
    ax.grid(alpha=0.2); ax.legend(); fig.tight_layout()
    fig.savefig(result_dir / "recovery_pp_cases.png", dpi=160); plt.close(fig)
    print(f"best recovery: {best['tag']} score={best['score']:.4f}")
    print(f"checkpoint: {best_path}")


if __name__ == "__main__":
    main()
