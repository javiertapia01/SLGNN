"""Cierre S02-R2 previo a N60-ZG con replay, L2-SP y auditoría multisemilla."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from train_S02_joint_consolidation import (
    ROOT, backward_rollout, load_checkpoint_model, rollout_starts,
    save_checkpoint, suite_metrics,
)
from train_benchmark_wall_transfer import (
    active_indices, compute_scales, load_data, make_particles,
    normalized_acceleration_loss, pp_near_indices, randint, targets,
    wall_near_indices,
)


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def interpolated_state(parent, specialist, alpha):
    a, b = parent.state_dict(), specialist.state_dict()
    return {
        k: (a[k] + alpha * (b[k] - a[k]) if k.startswith("head_pp_") else a[k]).detach().clone()
        for k in a
    }


def optimizer_for(model, cfg, multiplier=1.0):
    tc = cfg["training"]
    groups = {"pw": [], "pp": [], "processor": [], "material": []}
    for name, p in model.named_parameters():
        if name.startswith("head_pw_"):
            groups["pw"].append(p)
        elif name.startswith("head_pp_"):
            groups["pp"].append(p)
        elif name.startswith("material_encoder"):
            groups["material"].append(p)
        else:
            groups["processor"].append(p)
    return torch.optim.AdamW([
        {"params": groups["pw"], "lr": multiplier * float(tc["lr_wall_heads"]), "name": "pw"},
        {"params": groups["pp"], "lr": multiplier * float(tc["lr_particle_heads"]), "name": "pp"},
        {"params": groups["processor"], "lr": multiplier * float(tc["lr_processors"]), "name": "processor"},
        {"params": groups["material"], "lr": multiplier * float(tc["lr_material_encoder"]), "name": "material"},
    ], weight_decay=float(tc["weight_decay"]))


def split_indices(indices, modulus, train):
    values = [int(x) for x in indices]
    selected = [x for x in values if ((x % modulus) != 0) == train]
    return selected or values


def sampled_loss(model, data, cases, particles, wall, scales, cfg, batch, generator, iteration, is_wall):
    tc, ec = cfg["training"], cfg["evaluation"]
    total = torch.zeros(())
    for b in range(batch):
        case = cases[(iteration * batch + b) % len(cases)]
        tr = data[case]
        active = split_indices(active_indices(tr, float(ec["active_acceleration_threshold"])), int(tc["holdout_modulus"]), True)
        near_raw = wall_near_indices(tr, model.cfg.g_off) if is_wall else pp_near_indices(tr, model.cfg.r_off)
        near = split_indices(near_raw, int(tc["holdout_modulus"]), True)
        choices = active if float(torch.rand((), generator=generator)) < float(tc["active_probability"]) else near
        k = choices[randint(len(choices), generator)]
        a_ref, alpha_ref = targets(tr)
        out = model(tr.q[k], tr.v[k], tr.omega[k], particles, wall=wall)
        total = total + normalized_acceleration_loss(out, a_ref[k], alpha_ref[k], scales)
    return total / batch


def anchor_loss(model, anchor):
    values = []
    for name, p in model.named_parameters():
        ref = anchor[name]
        scale = ref.pow(2).mean().detach().clamp_min(1e-8)
        values.append((p - ref).pow(2).mean() / scale)
    return torch.stack(values).mean()


@torch.no_grad()
def holdout_loss(model, wall_data, pp_data, wall, wp, pp, ws, ps, cfg):
    modulus = int(cfg["training"]["holdout_modulus"])
    threshold = float(cfg["evaluation"]["active_acceleration_threshold"])
    rows = {"wall": [], "particle": []}
    model.eval()
    for kind, data, particles, boundary, scales in [
        ("wall", wall_data, wp, wall, ws), ("particle", pp_data, pp, None, ps)
    ]:
        for case, tr in data.items():
            idx = split_indices(active_indices(tr, threshold), modulus, False)
            a_ref, alpha_ref = targets(tr)
            vals = []
            for k in idx:
                out = model(tr.q[k], tr.v[k], tr.omega[k], particles, wall=boundary)
                vals.append(float(normalized_acceleration_loss(out, a_ref[k], alpha_ref[k], scales)))
            rows[kind].append({"case": case, "loss": float(np.mean(vals)), "n": len(vals)})
    return rows


def evaluate_candidate(model, wall_data, pp_data, wall, specialist_rows, source_summary, cfg):
    summary, wall_rows, pp_rows = suite_metrics(model, wall_data, pp_data, wall, float(cfg["evaluation"]["active_acceleration_threshold"]))
    ref = {r["case"]: r["rollout_v_rmse_post"] for r in specialist_rows}
    ratios = {r["case"]: r["rollout_v_rmse_post"] / max(ref[r["case"]], 1e-12) for r in pp_rows}
    ec = cfg["evaluation"]
    wall_ratio = summary["wall_v_median"] / (source_summary["wall_v_median"] * (1 + float(ec["max_wall_degradation"])))
    specialist_median = float(np.median(list(ref.values())))
    median_ratio = summary["pp_v_median"] / (specialist_median * (1 + float(ec["max_particle_median_degradation"])))
    case_gate_ratio = max(ratios.values()) / float(ec["max_case_ratio_to_specialist"])
    score = max(wall_ratio, median_ratio, case_gate_ratio)
    return {"summary": summary, "wall_rows": wall_rows, "pp_rows": pp_rows, "case_ratios": ratios, "gate_score": score, "passed": score <= 1.0}


def train_seed(seed, model, anchor, wall_data, pp_data, wall, wp, pp, ws, ps, cfg):
    tc, rc = cfg["training"], cfg["rollout"]
    wg = torch.Generator().manual_seed(seed + 10000); pg = torch.Generator().manual_seed(seed + 20000)
    history = []
    optimizer = optimizer_for(model, cfg)
    wall_cases = list(wall_data); pp_cases = [str(x) for x in tc["particle_case_sequence"]]
    model.train()
    for it in range(int(tc["one_step_iterations"])):
        optimizer.zero_grad(set_to_none=True)
        lw = sampled_loss(model, wall_data, wall_cases, wp, wall, ws, cfg, int(tc["wall_batch_size"]), wg, it, True)
        lp = sampled_loss(model, pp_data, pp_cases, pp, None, ps, cfg, int(tc["particle_batch_size"]), pg, it, False)
        la = anchor_loss(model, anchor)
        loss = float(tc["weight_wall"]) * lw + float(tc["weight_particle"]) * lp + float(tc["anchor_weight"]) * la
        loss.backward(); grad = torch.nn.utils.clip_grad_norm_(model.parameters(), float(tc["grad_clip"])); optimizer.step()
        history.append({"phase": "one-step", "iteration": it, "wall_loss": float(lw), "pp_loss": float(lp), "anchor_loss": float(la), "total_loss": float(loss), "grad_norm": float(grad)})
        if it % int(tc["log_every"]) == 0 or it == int(tc["one_step_iterations"]) - 1:
            print(f"seed={seed} one-step {it}/{tc['one_step_iterations']} wall={float(lw):.4g} pp={float(lp):.4g}", flush=True)
    threshold = float(cfg["evaluation"]["active_acceleration_threshold"])
    for horizon, iterations in zip(rc["horizons"], rc["iterations"]):
        optimizer = optimizer_for(model, cfg, float(rc["learning_rate_multiplier"]))
        ws0 = {c: rollout_starts(tr, int(horizon), threshold) for c, tr in wall_data.items()}
        ps0 = {c: rollout_starts(tr, int(horizon), threshold) for c, tr in pp_data.items()}
        for it in range(int(iterations)):
            optimizer.zero_grad(set_to_none=True); wc = wall_cases[it % len(wall_cases)]; pc = pp_cases[it % len(pp_cases)]
            wk = ws0[wc][randint(len(ws0[wc]), wg)]; pk = ps0[pc][randint(len(ps0[pc]), pg)]
            lw = backward_rollout(model, wall_data[wc], wp, wall, wk, int(horizon), ws, float(rc["sigma_q"]), int(rc["tbptt_chunk"]), float(tc["weight_wall"]), float(rc["noise_q"]), float(rc["noise_v"]), wg)
            lp = backward_rollout(model, pp_data[pc], pp, None, pk, int(horizon), ps, float(rc["sigma_q"]), int(rc["tbptt_chunk"]), float(tc["weight_particle"]), float(rc["noise_q"]), float(rc["noise_v"]), pg)
            la = anchor_loss(model, anchor); (float(tc["anchor_weight"]) * la).backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), float(tc["grad_clip"])); optimizer.step()
            history.append({"phase": f"H{horizon}", "iteration": it, "wall_loss": lw, "pp_loss": lp, "anchor_loss": float(la), "total_loss": lw + lp + float(tc["anchor_weight"]) * float(la), "grad_norm": float(grad)})
    return optimizer, history


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/slgnn_v2/curriculum/S02_R2_preN60.yaml"); parser.add_argument("--finalize-existing", action="store_true"); args = parser.parse_args()
    cfg = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8")); run_id = cfg["run_id"]
    ckpt_dir = resolve(cfg["output"]["checkpoint_root"]) / run_id; result_dir = resolve(cfg["output"]["result_root"]) / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True); result_dir.mkdir(parents=True, exist_ok=True)
    parent_path = resolve(cfg["parent_checkpoint"]); specialist_path = resolve(cfg["particle_specialist_checkpoint"])
    wall_data, pp_data, wall = load_data(cfg); wp = make_particles(next(iter(wall_data.values()))); pp = make_particles(next(iter(pp_data.values())))
    ws = compute_scales(wall_data, list(wall_data), lambda tr: active_indices(tr, float(cfg["evaluation"]["active_acceleration_threshold"])))
    ps = compute_scales(pp_data, list(pp_data), lambda tr: pp_near_indices(tr, float(cfg["model"]["r_off"])))
    parent, _ = load_checkpoint_model(parent_path, cfg); specialist, _ = load_checkpoint_model(specialist_path, cfg)
    source_summary, _, _ = suite_metrics(parent, wall_data, pp_data, wall, float(cfg["evaluation"]["active_acceleration_threshold"]))
    _, _, specialist_rows = suite_metrics(specialist, wall_data, pp_data, wall, float(cfg["evaluation"]["active_acceleration_threshold"]))
    if args.finalize_existing:
        candidates_path = result_dir / "candidates.json"
        candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
        ref_median = float(np.median([r["rollout_v_rmse_post"] for r in specialist_rows]))
        ec = cfg["evaluation"]
        for x in candidates:
            wall_ratio = x["summary"]["wall_v_median"] / (source_summary["wall_v_median"] * (1 + float(ec["max_wall_degradation"])))
            median_ratio = x["summary"]["pp_v_median"] / (ref_median * (1 + float(ec["max_particle_median_degradation"])))
            case_gate_ratio = max(x["case_ratios"].values()) / float(ec["max_case_ratio_to_specialist"])
            x["gate_score"] = max(wall_ratio, median_ratio, case_gate_ratio)
            x["passed"] = x["gate_score"] <= 1.0
        passing = [x for x in candidates if x["passed"]]
        best = min(passing or candidates, key=lambda x: x["gate_score"])
        seed_pass_fraction = sum(x["passed"] for x in candidates if x["seed"] is not None) / len(cfg["seeds"])
        promoted = best["passed"] and seed_pass_fraction + 1e-6 >= float(ec["minimum_seed_pass_fraction"])
        source_file = ckpt_dir / ("stage-alpha-baseline.pt" if best["seed"] is None else f"seed-{best['seed']}.pt")
        if promoted:
            promoted_path = ckpt_dir / "promoted-to-S03-N60-ZG-RP.pt"
            shutil.copy2(source_file, promoted_path)
        candidates_path.write_text(json.dumps(candidates, indent=2), encoding="utf-8")
        manifest_path = result_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update({"best_tag": best["tag"], "seed_pass_fraction": seed_pass_fraction, "promoted": promoted, "promoted_checkpoint_sha256": sha256(promoted_path) if promoted else None, "gate_definition": "PW median <= 1.05x source; every PP case and PP median <= 1.10x specialist."})
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        lines = ["# S02-R2 — Cierre previo a N60-ZG", "", f"- Mejor candidato: `{best['tag']}`.", f"- Semillas aprobadas: {seed_pass_fraction:.0%}.", f"- Promoción a S03-N60-ZG-RP: **{'APROBADA' if promoted else 'NO APROBADA'}**.", "", "| candidato | pared mediana | PP mediana | peor razón PP/especialista | gate score | aprobado |", "|---|---:|---:|---:|---:|---:|"]
        for x in candidates:
            lines.append(f"| {x['tag']} | {x['summary']['wall_v_median']:.4g} | {x['summary']['pp_v_median']:.4g} | {max(x['case_ratios'].values()):.4g} | {x['gate_score']:.4g} | {'sí' if x['passed'] else 'no'} |")
        lines += ["", "Gate: mediana PW ≤1.05× `best-joint`; cada caso PP y la mediana PP ≤1.10× el especialista.", "", "Limitación: sólo existe una trayectoria DEM por condición. El holdout temporal sirve como auditoría interna, no como prueba de generalización a condiciones iniciales nuevas."]
        (result_dir / "RESULTADOS.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"finalized best={best['tag']} seed_pass_fraction={seed_pass_fraction:.3f} promoted={promoted}")
        return
    initial = interpolated_state(parent, specialist, float(cfg["initialization"]["pp_head_extrapolation_alpha"])); parent.load_state_dict(initial)
    baseline = evaluate_candidate(parent, wall_data, pp_data, wall, specialist_rows, source_summary, cfg)
    baseline["tag"] = "alpha-baseline"; baseline["seed"] = None
    save_checkpoint(ckpt_dir / "stage-alpha-baseline.pt", parent, None, cfg, "alpha-baseline", baseline)
    candidates = [baseline]; all_history = []; started = time.time()
    for seed in cfg["seeds"]:
        torch.manual_seed(int(seed)); model, _ = load_checkpoint_model(parent_path, cfg); model.load_state_dict(deepcopy(initial))
        optimizer, history = train_seed(int(seed), model, initial, wall_data, pp_data, wall, wp, pp, ws, ps, cfg)
        audit = holdout_loss(model, wall_data, pp_data, wall, wp, pp, ws, ps, cfg)
        result = evaluate_candidate(model, wall_data, pp_data, wall, specialist_rows, source_summary, cfg)
        result.update({"tag": f"seed-{seed}", "seed": int(seed), "temporal_holdout": audit}); candidates.append(result)
        save_checkpoint(ckpt_dir / f"seed-{seed}.pt", model, optimizer, cfg, f"S02-R2-seed-{seed}", result)
        all_history.extend({"seed": int(seed), **row} for row in history)
        print(f"seed={seed} gate_score={result['gate_score']:.4f} passed={result['passed']}", flush=True)
    passing = [x for x in candidates if x["passed"]]; best = min(passing or candidates, key=lambda x: x["gate_score"])
    seed_pass_fraction = sum(x["passed"] for x in candidates if x["seed"] is not None) / len(cfg["seeds"])
    promoted = best["passed"] and seed_pass_fraction + 1e-6 >= float(cfg["evaluation"]["minimum_seed_pass_fraction"])
    source_file = ckpt_dir / ("stage-alpha-baseline.pt" if best["seed"] is None else f"seed-{best['seed']}.pt")
    promoted_path = ckpt_dir / "promoted-to-S03-N60-ZG-RP.pt"
    if promoted: shutil.copy2(source_file, promoted_path)
    (result_dir / "candidates.json").write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    with (result_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_history[0].keys()); writer.writeheader(); writer.writerows(all_history)
    manifest = {"run_id": run_id, "lineage": cfg["lineage"], "parent": str(parent_path), "parent_sha256": sha256(parent_path), "specialist": str(specialist_path), "config": cfg, "best_tag": best["tag"], "seed_pass_fraction": seed_pass_fraction, "promoted": promoted, "promoted_checkpoint_sha256": sha256(promoted_path) if promoted else None, "elapsed_seconds": time.time()-started, "python": platform.python_version(), "torch": torch.__version__, "data_limitation": "Una trayectoria DEM por condición; el holdout temporal no equivale a condiciones iniciales independientes."}
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = ["# S02-R2 — Cierre previo a N60-ZG", "", f"- Mejor candidato: `{best['tag']}`.", f"- Semillas aprobadas: {seed_pass_fraction:.0%}.", f"- Promoción a S03-N60-ZG-RP: **{'APROBADA' if promoted else 'NO APROBADA'}**.", "", "| candidato | pared mediana | PP mediana | peor razón PP/especialista | gate score | aprobado |", "|---|---:|---:|---:|---:|---:|"]
    for x in candidates:
        lines.append(f"| {x['tag']} | {x['summary']['wall_v_median']:.4g} | {x['summary']['pp_v_median']:.4g} | {max(x['case_ratios'].values()):.4g} | {x['gate_score']:.4g} | {'sí' if x['passed'] else 'no'} |")
    lines += ["", "Limitación: sólo existe una trayectoria DEM por condición. El holdout temporal sirve como auditoría interna, no como prueba de generalización a condiciones iniciales nuevas."]
    (result_dir / "RESULTADOS.md").write_text("\n".join(lines), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(9,4)); tags=[x['tag'] for x in candidates]; ax.plot(tags,[max(x['case_ratios'].values()) for x in candidates],"o-",label="peor PP/especialista"); ax.axhline(float(cfg['evaluation']['max_case_ratio_to_specialist']),color='r',ls='--',label='límite'); ax.set_ylabel('razón RMSE'); ax.tick_params(axis='x',rotation=20); ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(result_dir/'seed_robustness.png',dpi=160); plt.close(fig)
    print(f"best={best['tag']} seed_pass_fraction={seed_pass_fraction:.3f} promoted={promoted}")


if __name__ == "__main__":
    main()
