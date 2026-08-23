"""S02: consolidación multitarea PP2O + PW1 desde transfer_final.pt.

Etapas:
  1. recalibrar cabezas partícula-pared con backbone congelado;
  2. recalibrar cabezas partícula-partícula con backbone congelado;
  3. entrenamiento a un paso 50/50 con learning rates discriminativos;
  4. curriculum de rollout conjunto H=8,16,32;
  5. seleccionar best-pp, best-pw y best-joint contra referencias explícitas.

El directorio de ejecución y manifest.json preservan la secuencia completa de
entrenamientos. Nunca se sobrescribe el checkpoint padre.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from train_benchmark_wall_transfer import (
    active_indices,
    build_model,
    compute_scales,
    evaluate_pp,
    evaluate_wall,
    load_data,
    make_particles,
    normalized_acceleration_loss,
    pp_near_indices,
    predict,
    randint,
    targets,
    wall_near_indices,
)
from slgnn.integrator import semi_implicit_step


ROOT = Path(__file__).resolve().parents[2]


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_model(path, cfg):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def set_trainable(model, prefixes=None):
    for name, parameter in model.named_parameters():
        parameter.requires_grad = prefixes is None or any(
            name.startswith(prefix) for prefix in prefixes
        )


def sample_index(active, near, probability, generator):
    use_active = float(torch.rand((), generator=generator)) < probability
    choices = active if use_active else near
    return int(choices[randint(len(choices), generator)])


def wall_batch_loss(
    model, wall_data, cases, particles, wall, scales, cfg,
    batch_size, probability, generator, iteration,
):
    threshold = float(cfg["evaluation"]["active_acceleration_threshold"])
    total = torch.zeros(())
    for b in range(batch_size):
        case = cases[(iteration * batch_size + b) % len(cases)]
        tr = wall_data[case]
        active = active_indices(tr, threshold)
        near = wall_near_indices(tr, model.cfg.g_off)
        k = sample_index(active, near, probability, generator)
        a_ref, alpha_ref = targets(tr)
        out = model(tr.q[k], tr.v[k], tr.omega[k], particles, wall=wall)
        total = total + normalized_acceleration_loss(
            out, a_ref[k], alpha_ref[k], scales
        )
    return total / batch_size


def pp_batch_loss(
    model, pp_data, cases, particles, scales, cfg,
    batch_size, probability, generator, iteration,
):
    threshold = float(cfg["evaluation"]["active_acceleration_threshold"])
    total = torch.zeros(())
    for b in range(batch_size):
        case = cases[(iteration * batch_size + b) % len(cases)]
        tr = pp_data[case]
        active = active_indices(tr, threshold)
        near = pp_near_indices(tr, model.cfg.r_off)
        k = sample_index(active, near, probability, generator)
        a_ref, alpha_ref = targets(tr)
        out = model(tr.q[k], tr.v[k], tr.omega[k], particles)
        total = total + normalized_acceleration_loss(
            out, a_ref[k], alpha_ref[k], scales
        )
    return total / batch_size


def calibrate_head(
    model, task, wall_data, pp_data, wall, wall_particles, pp_particles,
    wall_scales, pp_scales, cfg, generator, history,
):
    cc = cfg["calibration"]
    if task == "wall":
        set_trainable(model, ["head_pw_"])
        parameters = [p for p in model.parameters() if p.requires_grad]
        iterations = int(cc["wall_head_iters"])
        optimizer = torch.optim.Adam(parameters, lr=float(cc["lr_wall_head"]))
    else:
        set_trainable(model, ["head_pp_"])
        parameters = [p for p in model.parameters() if p.requires_grad]
        iterations = int(cc["particle_head_iters"])
        optimizer = torch.optim.Adam(parameters, lr=float(cc["lr_particle_head"]))

    model.train()
    for it in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        if task == "wall":
            loss = wall_batch_loss(
                model, wall_data, list(wall_data), wall_particles, wall,
                wall_scales, cfg, int(cc["batch_size"]),
                float(cc["active_probability"]), generator, it,
            )
        else:
            loss = pp_batch_loss(
                model, pp_data, list(pp_data), pp_particles, pp_scales, cfg,
                min(int(cc["batch_size"]), len(pp_data)),
                float(cc["active_probability"]), generator, it,
            )
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()
        history.append({
            "phase": f"calibrate_{task}", "iteration": it,
            "wall_loss": float(loss.detach()) if task == "wall" else 0.0,
            "pp_loss": float(loss.detach()) if task == "particle" else 0.0,
            "total_loss": float(loss.detach()), "grad_norm": float(grad),
        })
        if it % 100 == 0 or it == iterations - 1:
            print(
                f"  calibrate-{task:8s} {it:4d}/{iterations} "
                f"loss={float(loss.detach()):.5f}", flush=True,
            )
    set_trainable(model, None)
    return optimizer


def joint_optimizer(model, cfg):
    jc = cfg["joint"]
    groups = {"wall": [], "particle": [], "processors": [], "material": []}
    for name, parameter in model.named_parameters():
        if name.startswith("head_pw_"):
            groups["wall"].append(parameter)
        elif name.startswith("head_pp_"):
            groups["particle"].append(parameter)
        elif name.startswith("material_encoder"):
            groups["material"].append(parameter)
        else:
            groups["processors"].append(parameter)
    return torch.optim.AdamW(
        [
            {"params": groups["wall"], "lr": float(jc["lr_wall_heads"]), "name": "wall"},
            {"params": groups["particle"], "lr": float(jc["lr_particle_heads"]), "name": "particle"},
            {"params": groups["processors"], "lr": float(jc["lr_processors"]), "name": "processors"},
            {"params": groups["material"], "lr": float(jc["lr_material_encoder"]), "name": "material"},
        ],
        weight_decay=float(jc["weight_decay"]),
    )


def joint_one_step(
    model, wall_data, pp_data, wall, wall_particles, pp_particles,
    wall_scales, pp_scales, cfg, wall_gen, pp_gen, history,
):
    jc = cfg["joint"]
    optimizer = joint_optimizer(model, cfg)
    iterations = int(jc["one_step_iters"])
    ww, wp = float(jc["task_weight_wall"]), float(jc["task_weight_particle"])
    model.train()
    for it in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        lw = wall_batch_loss(
            model, wall_data, list(wall_data), wall_particles, wall,
            wall_scales, cfg, int(jc["wall_batch_size"]),
            float(jc["active_probability"]), wall_gen, it,
        )
        lp = pp_batch_loss(
            model, pp_data, list(pp_data), pp_particles, pp_scales, cfg,
            int(jc["particle_batch_size"]),
            float(jc["active_probability"]), pp_gen, it,
        )
        loss = ww * lw + wp * lp
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Pérdida conjunta no finita en {it}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), float(jc["grad_clip"]))
        optimizer.step()
        history.append({
            "phase": "joint_one_step", "iteration": it,
            "wall_loss": float(lw.detach()), "pp_loss": float(lp.detach()),
            "total_loss": float(loss.detach()), "grad_norm": float(grad),
        })
        if it % int(jc["log_every"]) == 0 or it == iterations - 1:
            print(
                f"  joint one-step {it:4d}/{iterations} "
                f"wall={float(lw.detach()):.5f} pp={float(lp.detach()):.5f}",
                flush=True,
            )
    return optimizer


def rollout_starts(tr, horizon, threshold):
    active = active_indices(tr, threshold)
    lo = max(0, int(active[0]) - 5)
    hi = min(int(active[-1]), tr.q.shape[0] - horizon - 1)
    return list(range(lo, max(lo, hi) + 1))


def backward_rollout(
    model, tr, particles, wall, k0, horizon, scales,
    sigma_q, chunk_size, weight, noise_q, noise_v, generator,
):
    q = tr.q[k0] + noise_q * torch.randn(tr.q[k0].shape, generator=generator)
    v = tr.v[k0] + noise_v * torch.randn(tr.v[k0].shape, generator=generator)
    w = tr.omega[k0].clone()
    value = 0.0
    chunk = torch.zeros(())
    for s in range(horizon):
        q, v, w, _ = semi_implicit_step(
            model, q, v, w, particles, wall=wall, t=0.0, dt=tr.dt
        )
        k = k0 + s + 1
        step = ((q - tr.q[k]) / sigma_q).pow(2).mean()
        step = step + ((v - tr.v[k]) / scales["sigma_v"]).pow(2).mean()
        step = step + ((w - tr.omega[k]) / scales["sigma_w"]).pow(2).mean()
        step = weight * step / horizon
        chunk = chunk + step
        value += float(step.detach())
        if (s + 1) % chunk_size == 0 or s == horizon - 1:
            chunk.backward()
            chunk = torch.zeros(())
            if s != horizon - 1:
                q, v, w = q.detach(), v.detach(), w.detach()
    return value


def joint_rollout_stage(
    model, horizon, iterations, wall_data, pp_data, wall,
    wall_particles, pp_particles, wall_scales, pp_scales, cfg,
    wall_gen, pp_gen, history,
):
    rc, jc = cfg["rollout"], cfg["joint"]
    optimizer = joint_optimizer(model, cfg)
    threshold = float(cfg["evaluation"]["active_acceleration_threshold"])
    wall_starts = {c: rollout_starts(tr, horizon, threshold) for c, tr in wall_data.items()}
    pp_starts = {c: rollout_starts(tr, horizon, threshold) for c, tr in pp_data.items()}
    wall_cases, pp_cases = list(wall_data), list(pp_data)
    model.train()
    for it in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        wc = wall_cases[it % len(wall_cases)]
        pc = pp_cases[it % len(pp_cases)]
        wk = wall_starts[wc][randint(len(wall_starts[wc]), wall_gen)]
        pk = pp_starts[pc][randint(len(pp_starts[pc]), pp_gen)]
        lw = backward_rollout(
            model, wall_data[wc], wall_particles, wall, wk, horizon,
            wall_scales, float(rc["sigma_q"]), int(rc["tbptt_chunk"]),
            float(jc["task_weight_wall"]), float(rc["noise_q"]),
            float(rc["noise_v"]), wall_gen,
        )
        lp = backward_rollout(
            model, pp_data[pc], pp_particles, None, pk, horizon,
            pp_scales, float(rc["sigma_q"]), int(rc["tbptt_chunk"]),
            float(jc["task_weight_particle"]), float(rc["noise_q"]),
            float(rc["noise_v"]), pp_gen,
        )
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), float(jc["grad_clip"]))
        optimizer.step()
        history.append({
            "phase": f"rollout_H{horizon}", "iteration": it,
            "wall_loss": lw, "pp_loss": lp, "total_loss": lw + lp,
            "grad_norm": float(grad),
        })
        if it % int(rc["log_every"]) == 0 or it == iterations - 1:
            print(
                f"  joint rollout H={horizon:2d} {it:4d}/{iterations} "
                f"wall={lw:.5f} pp={lp:.5f}", flush=True,
            )
    return optimizer


def suite_metrics(model, wall_data, pp_data, wall, threshold):
    wall_particles = make_particles(next(iter(wall_data.values())))
    pp_particles = make_particles(next(iter(pp_data.values())))
    wall_rows, pp_rows = [], []
    model.eval()
    for case, tr in wall_data.items():
        metrics, _ = evaluate_wall(model, tr, wall_particles, wall, threshold)
        wall_rows.append({"case": case, **metrics})
    for case, tr in pp_data.items():
        pp_rows.append({"case": case, **evaluate_pp(model, tr, pp_particles, threshold)})
    summary = {
        "wall_v_median": float(np.median([r["rollout_v_rmse_post"] for r in wall_rows])),
        "wall_q_median": float(np.median([r["rollout_q_rmse_post"] for r in wall_rows])),
        "wall_spin_median": float(np.median([r["spin_change_relative_error"] for r in wall_rows])),
        "pp_v_median": float(np.median([r["rollout_v_rmse_post"] for r in pp_rows])),
        "pp_q_median": float(np.median([r["rollout_q_rmse_post"] for r in pp_rows])),
        "pp_spin_median": float(np.median([r["spin_change_relative_error"] for r in pp_rows])),
    }
    return summary, wall_rows, pp_rows


def save_checkpoint(path, model, optimizer, cfg, stage, evaluation):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "model_config": asdict(model.cfg),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "config": cfg, "run_id": cfg["run_id"], "lineage": cfg["lineage"],
        "parent_checkpoint": cfg["parent_checkpoint"], "stage": stage,
        "evaluation": evaluation,
    }, path)


class Selector:
    def __init__(self, run_dir, cfg, pp_target, wall_target):
        self.run_dir, self.cfg = run_dir, cfg
        self.pp_target, self.wall_target = pp_target, wall_target
        self.best_pp = self.best_wall = self.best_joint = float("inf")
        self.best_tags = {}
        self.best_summaries = {}

    def consider(self, model, optimizer, tag, summary):
        pp, wall = summary["pp_v_median"], summary["wall_v_median"]
        joint = max(pp / self.pp_target, wall / self.wall_target)
        summary["joint_minimax_score"] = joint
        if pp < self.best_pp:
            self.best_pp = pp; self.best_tags["pp"] = tag
            self.best_summaries["pp"] = dict(summary)
            save_checkpoint(self.run_dir / "best-pp.pt", model, optimizer, self.cfg, tag, summary)
        if wall < self.best_wall:
            self.best_wall = wall; self.best_tags["wall"] = tag
            self.best_summaries["wall"] = dict(summary)
            save_checkpoint(self.run_dir / "best-pw.pt", model, optimizer, self.cfg, tag, summary)
        if joint < self.best_joint:
            self.best_joint = joint; self.best_tags["joint"] = tag
            self.best_summaries["joint"] = dict(summary)
            save_checkpoint(self.run_dir / "best-joint.pt", model, optimizer, self.cfg, tag, summary)


def plot_stage_metrics(evaluations, path):
    tags = [e["tag"] for e in evaluations if e["kind"] == "candidate"]
    wall = [e["summary"]["wall_v_median"] for e in evaluations if e["kind"] == "candidate"]
    pp = [e["summary"]["pp_v_median"] for e in evaluations if e["kind"] == "candidate"]
    x = np.arange(len(tags))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(x, wall, "o-", label="Partícula–pared")
    ax.plot(x, pp, "s-", label="Partícula–partícula")
    ax.set_xticks(x, tags, rotation=25, ha="right")
    ax.set_ylabel("Mediana RMSE velocidad post-choque")
    ax.set_title("Consolidación por etapa")
    ax.grid(alpha=0.2); ax.legend(); fig.tight_layout()
    fig.savefig(path, dpi=160); plt.close(fig)


def plot_history(history, path):
    phases = []
    for row in history:
        if row["phase"] not in phases:
            phases.append(row["phase"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for phase in phases:
        rows = [r for r in history if r["phase"] == phase]
        axes[0].plot([r["iteration"] for r in rows], [r["wall_loss"] for r in rows], label=phase)
        axes[1].plot([r["iteration"] for r in rows], [r["pp_loss"] for r in rows], label=phase)
    for ax, title in zip(axes, ["Pérdida pared", "Pérdida partículas"]):
        ax.set_yscale("symlog", linthresh=1e-3); ax.set_xlabel("iteración local"); ax.set_title(title); ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def write_report(
    path, cfg, evaluations, selector, parent_summary, pp_ref, wall_ref,
    selected_pp_rows, pp_reference_rows,
):
    candidates = [e for e in evaluations if e["kind"] == "candidate"]
    selected = selector.best_summaries["joint"]
    allowed = float(cfg["evaluation"]["max_allowed_pp_degradation"])
    pp_limit = parent_summary["pp_v_median"] * (1 + allowed)
    critical_case = str(cfg["recovery"]["critical_case"])
    case_limit = float(cfg["recovery"]["max_case_ratio_to_specialist"])
    critical = next(r for r in selected_pp_rows if r["case"] == critical_case)
    critical_ref = next(r for r in pp_reference_rows if r["case"] == critical_case)
    critical_ratio = (
        critical["rollout_v_rmse_post"]
        / max(critical_ref["rollout_v_rmse_post"], 1e-12)
    )
    aggregate_pass = (
        selected["pp_v_median"] <= pp_limit
        and selected["wall_v_median"] < parent_summary["wall_v_median"]
    )
    promoted = aggregate_pass and critical_ratio <= case_limit
    lines = [
        "# S02 — Consolidación conjunta PP2O + PW1",
        "",
        f"**Run ID:** `{cfg['run_id']}`",
        "",
        f"**Padre:** `{cfg['parent_checkpoint']}`",
        "",
        "## Evolución por etapa",
        "",
        "| etapa | mediana RMSE v pared | mediana RMSE v pp | score minimax |",
        "|---|---:|---:|---:|",
    ]
    for item in candidates:
        s = item["summary"]
        lines.append(f"| {item['tag']} | {s['wall_v_median']:.4g} | {s['pp_v_median']:.4g} | {s['joint_minimax_score']:.4g} |")
    lines += [
        "",
        "## Referencias",
        "",
        f"- Padre `transfer_final`: pared={parent_summary['wall_v_median']:.4g}, pp={parent_summary['pp_v_median']:.4g}.",
        f"- Mejor referencia especializada pp: {pp_ref['pp_v_median']:.4g}.",
        f"- Mejor referencia especializada pared: {wall_ref['wall_v_median']:.4g}.",
        "",
        "## Selección",
        "",
        f"- `best-pp.pt`: etapa `{selector.best_tags.get('pp')}`.",
        f"- `best-pw.pt`: etapa `{selector.best_tags.get('wall')}`.",
        f"- `best-joint.pt`: etapa `{selector.best_tags.get('joint')}`.",
        f"- Criterio agregado: **{'APROBADO' if aggregate_pass else 'NO APROBADO'}**.",
        f"- Caso crítico `{critical_case}`: {critical['rollout_v_rmse_post']:.4g} "
        f"vs especialista {critical_ref['rollout_v_rmse_post']:.4g} "
        f"({critical_ratio:.4g}×; límite {case_limit:.2f}×).",
        f"- Criterio de promoción a H60: **{'APROBADO' if promoted else 'NO APROBADO'}**.",
        "",
        "La promoción exige mejorar pared respecto del padre sin degradar más de "
        f"{100*allowed:.0f}% la mediana pp y satisfacer el umbral por caso. "
        "`best-joint.pt` conserva el mejor resultado agregado; `final.pt` conserva "
        "el último estado aunque no sea necesariamente el mejor.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/slgnn_v2/curriculum/S02_joint_pp2o_pw1.yaml")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    if args.quick:
        cfg["run_id"] += "__quick"
        cfg["calibration"]["wall_head_iters"] = 10
        cfg["calibration"]["particle_head_iters"] = 10
        cfg["joint"]["one_step_iters"] = 15
        cfg["rollout"]["horizons"] = [4, 8]
        cfg["rollout"]["iterations"] = [2, 2]

    run_id = cfg["run_id"]
    ckpt_dir = resolve(cfg["output"]["checkpoint_root"]) / run_id
    result_dir = resolve(cfg["output"]["result_root"]) / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    parent_path = resolve(cfg["parent_checkpoint"])
    wall_data, pp_data, wall = load_data(cfg)
    threshold = float(cfg["evaluation"]["active_acceleration_threshold"])
    wall_particles = make_particles(next(iter(wall_data.values())))
    pp_particles = make_particles(next(iter(pp_data.values())))
    wall_scales = compute_scales(wall_data, list(wall_data), lambda tr: active_indices(tr, threshold))
    pp_scales = compute_scales(pp_data, list(pp_data), lambda tr: pp_near_indices(tr, float(cfg["model"]["r_off"])))

    torch.manual_seed(int(cfg["seed"]))
    wall_gen = torch.Generator().manual_seed(int(cfg["seed"]) + 10_000)
    pp_gen = torch.Generator().manual_seed(int(cfg["seed"]) + 20_000)
    model, parent_ck = load_checkpoint_model(parent_path, cfg)
    pp_reference, _ = load_checkpoint_model(resolve(cfg["reference_checkpoints"]["particle_particle"]), cfg)
    wall_reference, _ = load_checkpoint_model(resolve(cfg["reference_checkpoints"]["particle_wall"]), cfg)

    manifest = {
        "run_id": run_id, "stage": "S02-JOINT-PP2O-PW1",
        "lineage": cfg["lineage"], "parent_checkpoint": str(parent_path),
        "parent_sha256": sha256(parent_path),
        "datasets": [cfg["data"]["particle_dataset"], cfg["data"]["wall_dataset"]],
        "seed": cfg["seed"], "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": cfg,
    }
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (ckpt_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"run_id: {run_id}")
    print(f"parent: {parent_path}")
    started = time.time(); history = []; evaluations = []
    parent_summary, pw_rows, pp_rows = suite_metrics(model, wall_data, pp_data, wall, threshold)
    pp_ref_summary, _, pp_reference_rows = suite_metrics(pp_reference, wall_data, pp_data, wall, threshold)
    wall_ref_summary, _, _ = suite_metrics(wall_reference, wall_data, pp_data, wall, threshold)
    evaluations.extend([
        {"kind": "reference", "tag": "parent", "summary": parent_summary},
        {"kind": "reference", "tag": "pp-specialist", "summary": pp_ref_summary},
        {"kind": "reference", "tag": "pw-specialist", "summary": wall_ref_summary},
    ])
    pp_target = min(parent_summary["pp_v_median"], pp_ref_summary["pp_v_median"])
    wall_target = min(parent_summary["wall_v_median"], wall_ref_summary["wall_v_median"])
    selector = Selector(ckpt_dir, cfg, pp_target, wall_target)
    base = dict(parent_summary)
    base["joint_minimax_score"] = max(base["pp_v_median"] / pp_ref_summary["pp_v_median"], base["wall_v_median"] / wall_ref_summary["wall_v_median"])
    selector.consider(model, None, "parent", base)
    evaluations.append({"kind": "candidate", "tag": "parent", "summary": base})

    print("\n== Fase 1a: recalibración cabeza pared ==")
    optimizer = calibrate_head(model, "wall", wall_data, pp_data, wall, wall_particles, pp_particles, wall_scales, pp_scales, cfg, wall_gen, history)
    summary, _, _ = suite_metrics(model, wall_data, pp_data, wall, threshold)
    selector.consider(model, optimizer, "calibrate-pw", summary)
    evaluations.append({"kind": "candidate", "tag": "calibrate-pw", "summary": summary})
    save_checkpoint(ckpt_dir / "stage-calibrate-pw.pt", model, optimizer, cfg, "calibrate-pw", summary)

    print("\n== Fase 1b: recalibración cabeza partículas ==")
    optimizer = calibrate_head(model, "particle", wall_data, pp_data, wall, wall_particles, pp_particles, wall_scales, pp_scales, cfg, pp_gen, history)
    summary, _, _ = suite_metrics(model, wall_data, pp_data, wall, threshold)
    selector.consider(model, optimizer, "calibrate-pp", summary)
    evaluations.append({"kind": "candidate", "tag": "calibrate-pp", "summary": summary})
    save_checkpoint(ckpt_dir / "stage-calibrate-pp.pt", model, optimizer, cfg, "calibrate-pp", summary)

    print("\n== Fase 2: entrenamiento conjunto a un paso ==")
    optimizer = joint_one_step(model, wall_data, pp_data, wall, wall_particles, pp_particles, wall_scales, pp_scales, cfg, wall_gen, pp_gen, history)
    summary, _, _ = suite_metrics(model, wall_data, pp_data, wall, threshold)
    selector.consider(model, optimizer, "joint-one-step", summary)
    evaluations.append({"kind": "candidate", "tag": "joint-one-step", "summary": summary})
    save_checkpoint(ckpt_dir / "stage-joint-one-step.pt", model, optimizer, cfg, "joint-one-step", summary)

    print("\n== Fase 3: curriculum rollout conjunto ==")
    for horizon, iterations in zip(cfg["rollout"]["horizons"], cfg["rollout"]["iterations"]):
        optimizer = joint_rollout_stage(model, int(horizon), int(iterations), wall_data, pp_data, wall, wall_particles, pp_particles, wall_scales, pp_scales, cfg, wall_gen, pp_gen, history)
        tag = f"rollout-H{horizon}"
        summary, _, _ = suite_metrics(model, wall_data, pp_data, wall, threshold)
        selector.consider(model, optimizer, tag, summary)
        evaluations.append({"kind": "candidate", "tag": tag, "summary": summary})
        save_checkpoint(ckpt_dir / f"stage-{tag}.pt", model, optimizer, cfg, tag, summary)

    final_summary, final_wall_rows, final_pp_rows = suite_metrics(model, wall_data, pp_data, wall, threshold)
    save_checkpoint(ckpt_dir / "final.pt", model, optimizer, cfg, "final", final_summary)
    elapsed = time.time() - started
    (result_dir / "evaluation_history.json").write_text(json.dumps(evaluations, indent=2), encoding="utf-8")
    (result_dir / "final_wall_metrics.json").write_text(json.dumps(final_wall_rows, indent=2), encoding="utf-8")
    (result_dir / "final_pp_metrics.json").write_text(json.dumps(final_pp_rows, indent=2), encoding="utf-8")
    with (result_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
    plot_stage_metrics(evaluations, result_dir / "stage_metrics.png")
    plot_history(history, result_dir / "training_curves.png")
    best_joint, _ = load_checkpoint_model(ckpt_dir / "best-joint.pt", cfg)
    _, _, selected_pp_rows = suite_metrics(
        best_joint, wall_data, pp_data, wall, threshold
    )
    write_report(
        result_dir / "RESULTADOS.md", cfg, evaluations, selector,
        parent_summary, pp_ref_summary, wall_ref_summary,
        selected_pp_rows, pp_reference_rows,
    )
    metadata = {
        "run_id": run_id, "elapsed_seconds": elapsed,
        "python": platform.python_version(), "torch": torch.__version__,
        "device": "CPU", "best_tags": selector.best_tags,
    }
    (result_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nS02 completo en {elapsed / 60:.2f} min")
    print(f"checkpoints: {ckpt_dir}")
    print(f"resultados: {result_dir}")


if __name__ == "__main__":
    main()
