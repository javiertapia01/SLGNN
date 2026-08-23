"""S03: transferencia a 60 esferas homogéneas sin gravedad con replay S02.

La corrida es deliberadamente autocontenida: valida el checkpoint padre y los
datos antes de crear un optimizador nuevo, mantiene replay PP/PW, selecciona
con CASE06 y sólo consulta CASE07 tras cerrar la selección.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import shutil
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

from slgnn import (
    BoxSDF,
    Particles,
    SLGNN,
    SLGNNConfig,
    default_scales,
    finite_difference_accelerations,
    load_case,
)
from slgnn.integrator import semi_implicit_step
from slgnn.losses import (
    acceleration_loss,
    angular_acceleration_loss,
    passivity_loss,
    penetration_loss,
    residual_regularization,
)

from train_S02_joint_consolidation import suite_metrics
from train_benchmark_wall_transfer import (
    active_indices,
    compute_scales,
    load_data as load_replay_data,
    make_particles,
    normalized_acceleration_loss,
    pp_near_indices,
    randint,
    targets,
    wall_near_indices,
)


ROOT = Path(__file__).resolve().parents[2]
TIMESTEP_RE = re.compile(r"data_at_timestep_(\d+)\.csv$")


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def run_id(cfg, method: str, seed: int) -> str:
    return f"{cfg['campaign_id']}__{method}-seed-{seed}"


def run_slug(method: str, seed: int) -> str:
    """Nombre de carpeta corto; el run_id canónico completo vive en artefactos."""
    return f"{method}-seed-{seed}"


def strict_parent_model(cfg):
    path = resolve(cfg["parent_checkpoint"])
    actual_hash = sha256(path)
    expected_hash = str(cfg["parent_sha256"]).lower()
    if actual_hash.lower() != expected_hash:
        raise RuntimeError(f"SHA-256 padre incorrecto: {actual_hash} != {expected_hash}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    recorded = checkpoint.get("model_config")
    if not isinstance(recorded, dict):
        raise RuntimeError("El checkpoint padre no contiene model_config")
    expected = cfg["model"]
    for key in ("hidden", "layers", "use_history"):
        if recorded.get(key) != expected[key]:
            raise RuntimeError(
                f"Arquitectura incompatible en {key}: checkpoint={recorded.get(key)!r}, "
                f"S03={expected[key]!r}"
            )
    allowed = SLGNNConfig().__dict__
    model_cfg = {key: value for key, value in recorded.items() if key in allowed}
    model = SLGNN(SLGNNConfig(**model_cfg))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, checkpoint, path, actual_hash


def fresh_model(cfg, seed: int):
    torch.manual_seed(seed)
    allowed = SLGNNConfig().__dict__
    overrides = {key: value for key, value in cfg["model"].items() if key in allowed}
    return SLGNN(SLGNNConfig(**overrides))


def raw_case_audit(case_dir: Path, expected_particles: int, expected_snapshots: int):
    indexed = []
    for path in case_dir.glob("data_at_timestep_*.csv"):
        match = TIMESTEP_RE.match(path.name)
        if match:
            indexed.append((int(match.group(1)), path))
    indexed.sort(key=lambda item: item[0])
    steps = [step for step, _ in indexed]
    if steps != list(range(expected_snapshots)):
        raise RuntimeError(f"Timesteps inválidos en {case_dir}: {len(steps)} encontrados")
    expected_ids = list(range(1, expected_particles + 1))
    # Abrir 10.507 CSV individuales en una carpeta sincronizada domina por
    # completo el pre-flight. La forma [1501,60,3] sí se valida sobre todos los
    # frames cargados; aquí se audita el ID en anclas temporales estratificadas.
    audit_steps = sorted(set([0, expected_snapshots - 1] + list(range(0, expected_snapshots, 100))))
    paths_by_step = dict(indexed)
    for step in audit_steps:
        path = paths_by_step[step]
        with path.open(newline="") as stream:
            rows = csv.DictReader(stream)
            if "Particle_ID" not in (rows.fieldnames or []):
                raise RuntimeError(f"Falta Particle_ID en {path}")
            ids = sorted(int(float(row["Particle_ID"])) for row in rows)
        if ids != expected_ids:
            raise RuntimeError(f"Particle_ID inválido en {path}: {ids[:3]}...{ids[-3:]}")
    return {
        "snapshots": len(indexed),
        "particle_ids": [1, expected_particles],
        "particle_id_frames_audited": audit_steps,
        "particle_id_audit_mode": "stratified_raw_frames_plus_full_cached_tensor_shape",
    }


def load_n60_data(cfg, audit_raw: bool = True):
    dc = cfg["data"]
    scales = default_scales()
    base = ROOT / "data" / "extracted" / dc["dataset"]
    official = list(dc["train_cases"]) + [dc["validation_case"], dc["extrapolation_case"]]
    expected_n = int(dc["expected_particles"])
    expected_t = int(dc["expected_snapshots"])
    trajectories = {}
    report = {}
    for case in official:
        case_dir = base / case
        raw = raw_case_audit(case_dir, expected_n, expected_t) if audit_raw else {}
        si = load_case(case_dir, dt=float(dc["dt"]), cache=True)
        if si.q.shape != (expected_t, expected_n, 3):
            raise RuntimeError(f"Forma inesperada {case}: {tuple(si.q.shape)}")
        if si.v.shape != si.q.shape or si.omega.shape != si.q.shape:
            raise RuntimeError(f"Forma v/omega incompatible en {case}")
        if not torch.allclose(si.radii, torch.full_like(si.radii, 0.0025), atol=1e-9):
            raise RuntimeError(f"Radios inesperados en {case}")
        expected_mass = 4000.0 * (math.pi / 6.0) * 0.005**3
        if not torch.allclose(si.m, torch.full_like(si.m, expected_mass), rtol=1e-6):
            raise RuntimeError(f"Masas inesperadas en {case}")
        q_min = si.q.amin(dim=(0, 1))
        q_max = si.q.amax(dim=(0, 1))
        if (q_min < -1e-6).any() or (q_max > 0.030001).any():
            raise RuntimeError(f"Posiciones fuera de la caja SI en {case}: {q_min} {q_max}")
        nd = scales.nondim(si)
        if abs(nd.dt - 0.01) > 1e-12:
            raise RuntimeError(f"dt adimensional incorrecto en {case}: {nd.dt}")
        trajectories[case] = nd
        report[case] = {
            **raw,
            "shape": list(si.q.shape),
            "dt_s": si.dt,
            "dt_nondim": nd.dt,
            "radius_m": float(si.radii[0]),
            "radius_nondim": float(nd.radii[0]),
            "mass_kg": float(si.m[0]),
            "mass_nondim": float(nd.m[0]),
            "q_min_m": q_min.tolist(),
            "q_max_m": q_max.tolist(),
        }
    box_min = [scales.length(float(x)) for x in dc["box_min_m"]]
    box_max = [scales.length(float(x)) for x in dc["box_max_m"]]
    if box_min != [0.0, 0.0, 0.0] or box_max != [6.0, 6.0, 6.0]:
        raise RuntimeError(f"Caja adimensional incorrecta: {box_min}, {box_max}")
    gravity = torch.tensor(dc["gravity_vector"], dtype=torch.float32)
    if not torch.equal(gravity, torch.zeros(3, dtype=torch.float32)):
        raise RuntimeError(f"S03 exige gravedad exactamente nula, recibida {gravity.tolist()}")
    wall = BoxSDF(box_min, box_max)
    first = trajectories[dc["train_cases"][0]]
    particles = Particles.uniform(
        expected_n,
        m=float(first.m[0]),
        radius=float(first.radii[0]),
        dtype=first.q.dtype,
    )
    return trajectories, wall, gravity, particles, report


def compute_n60_sigmas(data, train_cases):
    values = {key: [] for key in ("a", "alpha", "v", "w")}
    for case in train_cases:
        tr = data[case]
        values["a"].append(finite_difference_accelerations(tr.v, tr.dt).reshape(-1))
        values["alpha"].append(finite_difference_accelerations(tr.omega, tr.dt).reshape(-1))
        values["v"].append(tr.v.reshape(-1))
        values["w"].append(tr.omega.reshape(-1))

    def std(name):
        return max(float(torch.cat(values[name]).std()), 1e-5)

    return {
        "sigma_a": std("a"),
        "sigma_alpha": std("alpha"),
        "sigma_q": 1.0,
        "sigma_v": std("v"),
        "sigma_w": std("w"),
    }


def kinetic_energy(v, w, particles):
    m = particles.m.to(v)
    inertia = particles.inertia.to(v)
    return float((0.5 * m * v.pow(2).sum(-1) + 0.5 * inertia * w.pow(2).sum(-1)).sum())


@torch.no_grad()
def evaluate_n60(model, tr, particles, wall, g_vec, horizons, divergence_penetration, arrays=False):
    requested = sorted(set(int(h) for h in horizons if int(h) <= tr.q.shape[0] - 1))
    q, v, w = tr.q[0], tr.v[0], tr.omega[0]
    se_q = se_v = se_w = 0.0
    max_pen = 0.0
    initial_energy = kinetic_energy(v, w, particles)
    max_energy = initial_energy
    metrics = {}
    q_hist = [q.detach().cpu()] if arrays else None
    v_hist = [v.detach().cpu()] if arrays else None
    w_hist = [w.detach().cpu()] if arrays else None
    stopped = None
    max_h = max(requested)
    for step in range(1, max_h + 1):
        q, v, w, out = semi_implicit_step(
            model, q, v, w, particles, wall=wall, t=(step - 1) * tr.dt,
            dt=tr.dt, g_vec=g_vec,
        )
        finite = torch.isfinite(q).all() and torch.isfinite(v).all() and torch.isfinite(w).all()
        if not bool(finite):
            stopped = f"non_finite_step_{step}"
            break
        pen = float((particles.radii.to(q) - out.diagnostics["phi"]).clamp_min(0).max())
        max_pen = max(max_pen, pen)
        if pen > divergence_penetration or float(q.abs().max()) > 1e6:
            stopped = f"divergence_step_{step}"
            break
        se_q += float((q - tr.q[step]).pow(2).mean())
        se_v += float((v - tr.v[step]).pow(2).mean())
        se_w += float((w - tr.omega[step]).pow(2).mean())
        energy = kinetic_energy(v, w, particles)
        max_energy = max(max_energy, energy)
        if arrays:
            q_hist.append(q.detach().cpu()); v_hist.append(v.detach().cpu()); w_hist.append(w.detach().cpu())
        if step in requested:
            metrics[str(step)] = {
                "reached": True,
                "rmse_q": math.sqrt(se_q / step),
                "rmse_v": math.sqrt(se_v / step),
                "rmse_w": math.sqrt(se_w / step),
                "max_penetration": max_pen,
                "energy_initial": initial_energy,
                "energy_final": energy,
                "energy_final_ratio": energy / max(initial_energy, 1e-12),
                "energy_peak_ratio": max_energy / max(initial_energy, 1e-12),
            }
    for horizon in requested:
        if str(horizon) not in metrics:
            metrics[str(horizon)] = {
                "reached": False,
                "stopped": stopped or "unknown",
                "steps_reached": len(q_hist) - 1 if arrays else step - 1,
                "rmse_q": float("inf"), "rmse_v": float("inf"), "rmse_w": float("inf"),
                "max_penetration": max_pen,
                "energy_peak_ratio": max_energy / max(initial_energy, 1e-12),
            }
    result = {"horizons": metrics, "stopped": stopped}
    if arrays:
        result["arrays"] = {
            "q_pred": torch.stack(q_hist).numpy(),
            "v_pred": torch.stack(v_hist).numpy(),
            "w_pred": torch.stack(w_hist).numpy(),
            "q_ref": tr.q[: len(q_hist)].numpy(),
            "v_ref": tr.v[: len(v_hist)].numpy(),
            "w_ref": tr.omega[: len(w_hist)].numpy(),
        }
    return result


def n60_score(metrics, sigmas):
    values = []
    for row in metrics["horizons"].values():
        if not row["reached"]:
            return 1e9
        values.append(
            math.log1p(row["rmse_q"] / sigmas["sigma_q"])
            + math.log1p(row["rmse_v"] / sigmas["sigma_v"])
            + 0.5 * math.log1p(row["rmse_w"] / sigmas["sigma_w"])
            + 0.25 * row["max_penetration"]
            + 0.1 * max(0.0, row["energy_peak_ratio"] - 1.5)
        )
    return float(np.mean(values))


def retention_audit(model, replay, specialist_rows, cfg):
    wall_data, pp_data, wall = replay
    summary, wall_rows, pp_rows = suite_metrics(
        model, wall_data, pp_data, wall,
        float(cfg["evaluation"]["active_acceleration_threshold"]),
    )
    refs = {row["case"]: row["rollout_v_rmse_post"] for row in specialist_rows}
    multipliers = {
        row["case"]: row["rollout_v_rmse_post"] / max(refs[row["case"]], 1e-12)
        for row in pp_rows
    }
    ec = cfg["evaluation"]
    pp4 = next(row["rollout_v_rmse_post"] for row in pp_rows if row["case"] == "4x")
    ratios = {
        "wall": summary["wall_v_median"] / float(ec["gate_wall_median"]),
        "pp_worst": max(multipliers.values()) / float(ec["gate_pp_case_multiplier"]),
        "pp4x": pp4 / float(ec["gate_pp4x_absolute"]),
    }
    return {
        "summary": summary,
        "wall_rows": wall_rows,
        "pp_rows": pp_rows,
        "pp_specialist": refs,
        "pp_multipliers": multipliers,
        "gate_ratios": ratios,
        "gate_score": max(ratios.values()),
        "passed": max(ratios.values()) <= 1.0,
    }


def activity_pools(data, cases, threshold):
    pools = {}
    refs = {}
    for case in cases:
        tr = data[case]
        a = finite_difference_accelerations(tr.v, tr.dt)
        alpha = finite_difference_accelerations(tr.omega, tr.dt)
        level = torch.maximum(a.norm(dim=-1).amax(-1), alpha.norm(dim=-1).amax(-1))
        active = torch.where(level > threshold)[0].tolist()
        low = torch.where(level <= threshold)[0].tolist()
        if not active:
            active = torch.topk(level, min(100, len(level))).indices.tolist()
        if not low:
            low = torch.topk(level, min(100, len(level)), largest=False).indices.tolist()
        pools[case] = {"active": active, "low": low, "level": level}
        refs[case] = (a, alpha)
    return pools, refs


def rollout_pools(data, cases, horizon, threshold):
    result = {}
    for case in cases:
        tr = data[case]
        a = finite_difference_accelerations(tr.v, tr.dt)
        alpha = finite_difference_accelerations(tr.omega, tr.dt)
        level = torch.maximum(a.norm(dim=-1).amax(-1), alpha.norm(dim=-1).amax(-1))
        hi = tr.q.shape[0] - horizon
        active, low = [], []
        for start in range(hi):
            target = active if bool((level[start:start + horizon] > threshold).any()) else low
            target.append(start)
        if not active:
            active = low[:]
        if not low:
            low = active[:]
        result[case] = {"active": active, "low": low}
    return result


def optimizer_for(model, cfg, multiplier=1.0):
    groups = {"processors": [], "pp": [], "pw": [], "material": []}
    parameter_names = {}
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(True)
        parameter_names[id(parameter)] = name
        if name.startswith("proc_"):
            groups["processors"].append(parameter)
        elif name.startswith("head_pp_"):
            groups["pp"].append(parameter)
        elif name.startswith("head_pw_"):
            groups["pw"].append(parameter)
        elif name.startswith("material_encoder"):
            groups["material"].append(parameter)
        else:
            raise RuntimeError(f"Parámetro sin grupo discriminativo: {name}")
    covered = {id(p) for values in groups.values() for p in values}
    if covered != set(parameter_names):
        raise RuntimeError("No todos los parámetros quedaron en el optimizador")
    oc = cfg["optimization"]
    optimizer = torch.optim.AdamW(
        [
            {"params": groups["processors"], "lr": multiplier * float(oc["lr_processors"]), "name": "processors"},
            {"params": groups["pp"], "lr": multiplier * float(oc["lr_particle_heads"]), "name": "particle_heads"},
            {"params": groups["pw"], "lr": multiplier * float(oc["lr_wall_heads"]), "name": "wall_heads"},
            {"params": groups["material"], "lr": multiplier * float(oc["lr_material_encoder"]), "name": "material_encoder"},
        ],
        weight_decay=float(oc["weight_decay"]),
    )
    if optimizer.state:
        raise RuntimeError("El optimizador S03 debe comenzar sin estado")
    return optimizer


def l2sp_loss(model, anchor):
    terms = []
    for name, parameter in model.named_parameters():
        reference = anchor[name].to(parameter)
        scale = reference.pow(2).mean().detach().clamp_min(1e-6)
        terms.append((parameter - reference).pow(2).mean() / scale)
    return torch.stack(terms).mean()


def auxiliary_loss(out, particles, cfg):
    lc = cfg["loss"]
    diagnostics = out.diagnostics
    return (
        float(lc["lambda_history"]) * residual_regularization(diagnostics["f_H"], diagnostics["tau_H"])
        + float(lc["lambda_passivity"]) * passivity_loss(diagnostics["P_hist_pp"], diagnostics["P_hist_pW"])
        + float(lc["lambda_penetration"]) * penetration_loss(
            diagnostics["phi"], particles.radii.to(out.a), out.a.new_tensor(20.0).item()
        )
    )


def replay_sample_loss(model, replay, replay_particles, replay_scales, cfg, iteration, generator):
    wall_data, pp_data, wall = replay
    wall_particles, pp_particles = replay_particles
    wall_scales, pp_scales = replay_scales
    choose_pp = iteration % 2 == 0
    if choose_pp:
        sequence = [str(value) for value in cfg["sampling"]["particle_case_sequence"]]
        case = sequence[(iteration // 2) % len(sequence)]
        tr = pp_data[case]
        choices = pp_near_indices(tr, model.cfg.r_off)
        k = int(choices[randint(len(choices), generator)])
        particles, boundary, scales = pp_particles, None, pp_scales
    else:
        cases = [str(value) for value in cfg["data"]["wall_cases"]]
        case = cases[(iteration // 2) % len(cases)]
        tr = wall_data[case]
        active = active_indices(tr, float(cfg["evaluation"]["active_acceleration_threshold"]))
        near = wall_near_indices(tr, model.cfg.g_off)
        choices = active if iteration % 4 == 1 else near
        k = int(choices[randint(len(choices), generator)])
        particles, boundary, scales = wall_particles, wall, wall_scales
    a_ref, alpha_ref = targets(tr)
    out = model(tr.q[k], tr.v[k], tr.omega[k], particles, wall=boundary)
    return normalized_acceleration_loss(out, a_ref[k], alpha_ref[k], scales) + auxiliary_loss(out, particles, cfg)


def save_checkpoint(path, model, optimizer, cfg, method, seed, stage, anchor_hash, evaluation):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "model_config": asdict(model.cfg),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "optimizer_reused": False,
        "config": cfg,
        "run_id": run_id(cfg, method, seed),
        "campaign_id": cfg["campaign_id"],
        "lineage": cfg["lineage"],
        "method": method,
        "seed": seed,
        "stage": stage,
        "parent_checkpoint": cfg["parent_checkpoint"],
        "parent_sha256": cfg["parent_sha256"],
        "anchor_sha256": anchor_hash,
        "evaluation": evaluation,
    }, path)


def make_stage_record(model, method, seed, stage, n60_val, sigmas, replay, specialist_rows, cfg):
    retention = retention_audit(model, replay, specialist_rows, cfg)
    score_n60 = n60_score(n60_val, sigmas)
    joint = score_n60 + math.log1p(retention["gate_score"]) + 10.0 * max(0.0, retention["gate_score"] - 1.0)
    h500 = n60_val["horizons"].get("500", {})
    stable = bool(
        h500.get("reached", False)
        and h500.get("max_penetration", float("inf")) <= float(cfg["evaluation"]["h64_stable_max_penetration"])
        and math.isfinite(score_n60)
    )
    return {
        "method": method, "seed": seed, "stage": stage,
        "n60": n60_val, "n60_score": score_n60,
        "retention": retention, "joint_score": joint, "stable": stable,
    }


def one_step_warmup(model, anchor, data, particles, wall, g_vec, replay,
                    replay_particles, replay_scales, sigmas, cfg, seed, iterations, history):
    optimizer = optimizer_for(model, cfg, 1.0)
    cases = list(cfg["data"]["train_cases"])
    pools, refs = activity_pools(data, cases, float(cfg["sampling"]["activity_threshold"]))
    generator = torch.Generator().manual_seed(seed + 10000)
    replay_generator = torch.Generator().manual_seed(seed + 20000)
    low_fraction = float(cfg["sampling"]["low_activity_fraction"])
    lc = cfg["loss"]
    model.train()
    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        n60_loss = torch.zeros(())
        # Tres muestras N60 y una de replay representan 75/25 por batch lógico.
        for batch_index in range(3):
            case = cases[(iteration * 3 + batch_index) % len(cases)]
            tr = data[case]
            use_low = float(torch.rand((), generator=generator)) < low_fraction
            choices = pools[case]["low" if use_low else "active"]
            k = choices[randint(len(choices), generator)]
            a_ref, alpha_ref = refs[case]
            out = model(tr.q[k], tr.v[k], tr.omega[k], particles, wall=wall, g_vec=g_vec)
            n60_loss = n60_loss + (
                float(lc["lambda_acceleration"]) * acceleration_loss(out.a, a_ref[k], sigmas["sigma_a"])
                + float(lc["lambda_angular_acceleration"]) * angular_acceleration_loss(out.alpha, alpha_ref[k], sigmas["sigma_alpha"])
                + auxiliary_loss(out, particles, cfg)
            )
        n60_loss = n60_loss / 3.0
        replay_loss = replay_sample_loss(
            model, replay, replay_particles, replay_scales, cfg, iteration, replay_generator
        )
        anchor_loss = l2sp_loss(model, anchor)
        total = 0.75 * n60_loss + 0.25 * replay_loss + float(lc["lambda_l2sp"]) * anchor_loss
        if not torch.isfinite(total):
            raise FloatingPointError(f"Warmup no finito seed={seed} iter={iteration}")
        total.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["optimization"]["grad_clip"]))
        optimizer.step()
        history.append({
            "method": "", "seed": seed, "stage": "warmup", "iteration": iteration,
            "n60_loss": float(n60_loss.detach()), "replay_loss": float(replay_loss.detach()),
            "l2sp_loss": float(anchor_loss.detach()), "total_loss": float(total.detach()),
            "grad_norm": float(grad), "validation_score": "", "lr_decay": 0,
        })
        if iteration % 50 == 0 or iteration == iterations - 1:
            print(
                f"  warmup {iteration:4d}/{iterations} "
                f"n60={float(n60_loss.detach()):.4g} "
                f"replay={float(replay_loss.detach()):.4g}",
                flush=True,
            )
    return optimizer


def rollout_backward(model, tr, particles, wall, g_vec, start, horizon, sigmas,
                     cfg, generator, weight):
    lc = cfg["loss"]
    chunk_size = int(cfg["curriculum"]["tbptt_chunk"])
    q = tr.q[start] + float(cfg["noise"]["sigma_q"]) * torch.randn(tr.q[start].shape, generator=generator)
    v = tr.v[start] + float(cfg["noise"]["sigma_v"]) * torch.randn(tr.v[start].shape, generator=generator)
    w = tr.omega[start].clone()
    chunk = torch.zeros(())
    total_value = 0.0
    for offset in range(horizon):
        q, v, w, out = semi_implicit_step(model, q, v, w, particles, wall=wall, t=0.0, dt=tr.dt, g_vec=g_vec)
        target = start + offset + 1
        decay = float(lc["rollout_time_decay"]) ** offset
        step = (
            float(lc["lambda_q"]) * ((q - tr.q[target]) / sigmas["sigma_q"]).pow(2).mean()
            + float(lc["lambda_v"]) * ((v - tr.v[target]) / sigmas["sigma_v"]).pow(2).mean()
            + float(lc["lambda_w"]) * ((w - tr.omega[target]) / sigmas["sigma_w"]).pow(2).mean()
            + auxiliary_loss(out, particles, cfg)
        )
        if offset == 0:
            a_ref = (tr.v[target] - tr.v[start]) / tr.dt
            alpha_ref = (tr.omega[target] - tr.omega[start]) / tr.dt
            step = step + float(lc["lambda_acceleration"]) * acceleration_loss(out.a, a_ref, sigmas["sigma_a"])
            step = step + float(lc["lambda_angular_acceleration"]) * angular_acceleration_loss(out.alpha, alpha_ref, sigmas["sigma_alpha"])
        step = weight * decay * step / horizon
        if not torch.isfinite(step):
            raise FloatingPointError(f"Loss rollout no finita H={horizon} paso={offset}")
        chunk = chunk + step
        total_value += float(step.detach())
        if (offset + 1) % chunk_size == 0 or offset == horizon - 1:
            chunk.backward()
            chunk = torch.zeros(())
            if offset != horizon - 1:
                q, v, w = q.detach(), v.detach(), w.detach()
    return total_value


def train_rollout_stage(model, anchor, data, particles, wall, g_vec, replay,
                        replay_particles, replay_scales, sigmas, cfg, method, seed,
                        horizon, iterations, history):
    multiplier = float(cfg["optimization"]["rollout_lr_multiplier"])
    optimizer = optimizer_for(model, cfg, multiplier)
    cases = list(cfg["data"]["train_cases"])
    pools = rollout_pools(data, cases, horizon, float(cfg["sampling"]["activity_threshold"]))
    generator = torch.Generator().manual_seed(seed + 30000 + horizon)
    replay_generator = torch.Generator().manual_seed(seed + 40000 + horizon)
    low_fraction = float(cfg["sampling"]["low_activity_fraction"])
    validation_case = data[cfg["data"]["validation_case"]]
    validation_scores = []
    decays = 0
    model.train()
    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        case = cases[iteration % len(cases)]
        use_low = float(torch.rand((), generator=generator)) < low_fraction
        starts = pools[case]["low" if use_low else "active"]
        start = starts[randint(len(starts), generator)]
        n60_value = rollout_backward(
            model, data[case], particles, wall, g_vec, start, horizon, sigmas,
            cfg, generator, 0.75,
        )
        replay_loss = replay_sample_loss(
            model, replay, replay_particles, replay_scales, cfg, iteration, replay_generator
        )
        anchor_loss = l2sp_loss(model, anchor)
        extra = 0.25 * replay_loss + float(cfg["loss"]["lambda_l2sp"]) * anchor_loss
        if not torch.isfinite(extra):
            raise FloatingPointError(f"Replay/L2-SP no finito H={horizon} iter={iteration}")
        extra.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["optimization"]["grad_clip"]))
        optimizer.step()
        val_score = ""
        lr_decay = 0
        if iteration % int(cfg["evaluation"]["validation_every"]) == 0 or iteration == iterations - 1:
            model.eval()
            validation = evaluate_n60(
                model, validation_case, particles, wall, g_vec, [100],
                float(cfg["evaluation"]["divergence_penetration"]),
            )
            val_score = n60_score(validation, sigmas)
            validation_scores.append(val_score)
            if len(validation_scores) >= 3 and validation_scores[-1] > validation_scores[-2] > validation_scores[-3] and decays < 2:
                for group in optimizer.param_groups:
                    group["lr"] *= float(cfg["optimization"]["oscillation_decay"])
                decays += 1
                lr_decay = 1
            model.train()
            print(f"  H={horizon:3d} {iteration:4d}/{iterations} train={n60_value:.4g} val={val_score:.4g} decay={lr_decay}", flush=True)
        history.append({
            "method": method, "seed": seed, "stage": f"H{horizon}", "iteration": iteration,
            "n60_loss": n60_value, "replay_loss": float(replay_loss.detach()),
            "l2sp_loss": float(anchor_loss.detach()),
            "total_loss": n60_value + 0.25 * float(replay_loss.detach()) + float(cfg["loss"]["lambda_l2sp"]) * float(anchor_loss.detach()),
            "grad_norm": float(grad), "validation_score": val_score, "lr_decay": lr_decay,
        })
    return optimizer


def train_base_run(method, seed, parent_state, data, particles, wall, g_vec, replay,
                   replay_particles, replay_scales, sigmas, specialist_rows, cfg,
                   ckpt_root, result_root, smoke=False):
    rid = run_id(cfg, method, seed)
    slug = run_slug(method, seed)
    run_ckpt = ckpt_root / slug
    run_result = result_root / slug
    run_ckpt.mkdir(parents=True, exist_ok=False)
    run_result.mkdir(parents=True, exist_ok=False)
    if method == "transfer":
        model = fresh_model(cfg, seed)
        model.load_state_dict(deepcopy(parent_state), strict=True)
        anchor = {key: value.detach().clone() for key, value in parent_state.items()}
        anchor_hash = cfg["parent_sha256"]
    else:
        model = fresh_model(cfg, seed)
        anchor = {key: value.detach().clone() for key, value in model.state_dict().items()}
        anchor_hash = "scratch-initialization"
    history = []
    candidates = []
    started = time.time()
    warmup_iterations = 2 if smoke else int(cfg["curriculum"]["warmup_iterations"])
    optimizer = one_step_warmup(
        model, anchor, data, particles, wall, g_vec, replay, replay_particles,
        replay_scales, sigmas, cfg, seed, warmup_iterations, history,
    )
    horizons = [4] if smoke else list(cfg["curriculum"]["horizons"])
    iterations = [1] if smoke else list(cfg["curriculum"]["iterations_per_horizon"])
    eval_horizons = [5, 10] if smoke else list(cfg["evaluation"]["horizons"])

    def audit_and_save(stage, opt):
        model.eval()
        validation = evaluate_n60(
            model, data[cfg["data"]["validation_case"]], particles, wall, g_vec,
            eval_horizons, float(cfg["evaluation"]["divergence_penetration"]),
        )
        record = make_stage_record(model, method, seed, stage, validation, sigmas, replay, specialist_rows, cfg)
        path = run_ckpt / f"stage-{stage}.pt"
        save_checkpoint(path, model, opt, cfg, method, seed, stage, anchor_hash, record)
        record["checkpoint"] = str(path)
        record["checkpoint_sha256"] = sha256(path)
        candidates.append(record)
        write_json(run_result / "phase_metrics.json", candidates)
        return record

    audit_and_save("warmup", optimizer)
    for horizon, count in zip(horizons, iterations):
        optimizer = train_rollout_stage(
            model, anchor, data, particles, wall, g_vec, replay, replay_particles,
            replay_scales, sigmas, cfg, method, seed, int(horizon), int(count), history,
        )
        audit_and_save(f"H{horizon}", optimizer)
    final_path = run_ckpt / "final.pt"
    save_checkpoint(final_path, model, optimizer, cfg, method, seed, candidates[-1]["stage"], anchor_hash, candidates[-1])
    with (run_result / "training_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader(); writer.writerows(history)
    write_json(run_result / "run_metadata.json", {
        "run_id": rid, "method": method, "seed": seed,
        "elapsed_seconds": time.time() - started, "final_checkpoint": str(final_path),
    })
    return {"method": method, "seed": seed, "run_id": rid, "candidates": candidates,
            "final_checkpoint": str(final_path), "history": history}


def continue_h128(run, parent_state, data, particles, wall, g_vec, replay,
                  replay_particles, replay_scales, sigmas, specialist_rows, cfg, smoke=False):
    if smoke:
        return None
    seed, method = run["seed"], run["method"]
    last = run["candidates"][-1]
    checkpoint = torch.load(last["checkpoint"], map_location="cpu", weights_only=False)
    model = fresh_model(cfg, seed)
    model.load_state_dict(checkpoint["model"], strict=True)
    if method == "transfer":
        anchor = {key: value.detach().clone() for key, value in parent_state.items()}
        anchor_hash = cfg["parent_sha256"]
    else:
        first_path = Path(run["candidates"][0]["checkpoint"])
        first = torch.load(first_path, map_location="cpu", weights_only=False)
        # Scratch usa su inicialización como referencia; queda almacenada en el primer
        # checkpoint sólo después de warmup, por lo que reconstruimos determinísticamente.
        anchor_model = fresh_model(cfg, seed)
        anchor = {key: value.detach().clone() for key, value in anchor_model.state_dict().items()}
        anchor_hash = "scratch-initialization"
    history = []
    optimizer = train_rollout_stage(
        model, anchor, data, particles, wall, g_vec, replay, replay_particles,
        replay_scales, sigmas, cfg, method, seed, 128,
        int(cfg["curriculum"]["h128_iterations"]), history,
    )
    validation = evaluate_n60(
        model, data[cfg["data"]["validation_case"]], particles, wall, g_vec,
        cfg["evaluation"]["horizons"], float(cfg["evaluation"]["divergence_penetration"]),
    )
    record = make_stage_record(model, method, seed, "H128", validation, sigmas, replay, specialist_rows, cfg)
    run_ckpt = Path(last["checkpoint"]).parent
    path = run_ckpt / "stage-H128.pt"
    save_checkpoint(path, model, optimizer, cfg, method, seed, "H128", anchor_hash, record)
    record["checkpoint"] = str(path); record["checkpoint_sha256"] = sha256(path)
    run["candidates"].append(record)
    run["final_checkpoint"] = str(run_ckpt / "final.pt")
    shutil.copy2(path, run["final_checkpoint"])
    suffix = "__smoke" if smoke else ""
    result_dir = (
        resolve(cfg["output"]["result_root"])
        / f"{cfg['campaign_id']}{suffix}"
        / run_slug(method, seed)
    )
    with (result_dir / "training_history.csv").open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys()); writer.writerows(history)
    write_json(result_dir / "phase_metrics.json", run["candidates"])
    return record


def copy_selected(record, destination):
    shutil.copy2(record["checkpoint"], destination)
    return {"source": record["checkpoint"], "sha256": sha256(destination), "record": record}


def verify_checkpoint(path, cfg):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_cfg = checkpoint["model_config"]
    if model_cfg["hidden"] != 16 or model_cfg["layers"] != 2 or not model_cfg["use_history"]:
        raise RuntimeError(f"Checkpoint seleccionado incompatible: {path}")
    model = SLGNN(SLGNNConfig(**model_cfg))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


def plot_training(runs, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for run in runs:
        rows = run["history"]
        label = f"{run['method']} {run['seed']}"
        axes[0].plot([row["total_loss"] for row in rows], label=label, alpha=0.8)
        values = [float(row["validation_score"]) for row in rows if row["validation_score"] != ""]
        axes[1].plot(values, marker="o", ms=3, label=label)
    axes[0].set_yscale("log"); axes[0].set_title("Pérdida de entrenamiento")
    axes[1].set_yscale("log"); axes[1].set_title("Score CASE06 H100")
    for axis in axes:
        axis.grid(alpha=0.2); axis.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def make_comparison_gif(arrays, path, title):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    count = arrays["q_pred"].shape[0]
    indices = np.linspace(0, count - 1, min(90, count), dtype=int)
    for index in indices:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        for axis, values, subtitle in [
            (axes[0], arrays["q_ref"][index], "DEM"),
            (axes[1], arrays["q_pred"][index], "SLGNN"),
        ]:
            scatter = axis.scatter(values[:, 0], values[:, 1], c=values[:, 2], s=18, vmin=0, vmax=6, cmap="viridis")
            axis.set_xlim(0, 6); axis.set_ylim(0, 6); axis.set_aspect("equal")
            axis.set_title(subtitle); axis.set_xlabel("x/dₚ"); axis.set_ylabel("y/dₚ")
        fig.suptitle(f"{title} — paso {index}")
        fig.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(rgba[:, :, :3].copy()))
        plt.close(fig)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=110, loop=0, optimize=True)


def write_report(path, cfg, runs, selected, case06, case07, h128_executed, promoted):
    candidates = [candidate for run in runs if run["method"] == "transfer" for candidate in run["candidates"]]
    scratch_candidates = [candidate for run in runs if run["method"] == "scratch" for candidate in run["candidates"]]
    lines = [
        "# S03-N60-ZG-RP — resultados", "",
        f"- Semillas transferidas: {', '.join(str(x) for x in cfg['seeds']['transfer'])}.",
        f"- Control scratch: {', '.join(str(x) for x in cfg['seeds']['scratch'])}.",
        f"- H128 condicional: **{'ejecutado' if h128_executed else 'no ejecutado; H64 no fue estable en todas las semillas transferidas'}**.",
        f"- Promoción: **{'APROBADA' if promoted else 'NO APROBADA'}**.", "",
        "## Candidatos transferidos", "",
        "| semilla | fase | score N60 | PW mediana | peor PP/especialista | PP-4x | gates | estable |",
        "|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for candidate in candidates:
        retention = candidate["retention"]
        pp4 = next(row["rollout_v_rmse_post"] for row in retention["pp_rows"] if row["case"] == "4x")
        lines.append(
            f"| {candidate['seed']} | {candidate['stage']} | {candidate['n60_score']:.4g} | "
            f"{retention['summary']['wall_v_median']:.4g} | {max(retention['pp_multipliers'].values()):.4g} | "
            f"{pp4:.4g} | {'sí' if retention['passed'] else 'no'} | {'sí' if candidate['stable'] else 'no'} |"
        )
    lines += [
        "", "## Control desde cero", "",
        "| semilla | fase | CASE06 H100 RMSE q | RMSE v | RMSE ω | alcanzó H500 | PW mediana | PP mediana |",
        "|---:|---|---:|---:|---:|---|---:|---:|",
    ]
    for candidate in scratch_candidates:
        h100 = candidate["n60"]["horizons"]["100"]
        h500 = candidate["n60"]["horizons"]["500"]
        summary = candidate["retention"]["summary"]
        lines.append(
            f"| {candidate['seed']} | {candidate['stage']} | {h100['rmse_q']:.4g} | "
            f"{h100['rmse_v']:.4g} | {h100['rmse_w']:.4g} | {'sí' if h500['reached'] else 'no'} | "
            f"{summary['wall_v_median']:.4g} | {summary['pp_v_median']:.4g} |"
        )
    best = selected["best_joint"]["record"]
    lines += [
        "", "## Selección", "",
        f"- `best-N60.pt`: semilla {selected['best_n60']['record']['seed']}, fase {selected['best_n60']['record']['stage']}.",
        f"- `best-retention.pt`: semilla {selected['best_retention']['record']['seed']}, fase {selected['best_retention']['record']['stage']}.",
        f"- `best-joint.pt`: semilla {best['seed']}, fase {best['stage']}, gates {'aprobados' if best['retention']['passed'] else 'fallidos'}.",
        "- CASE07 se evaluó después de fijar `best-joint.pt`; no intervino en la selección.",
        "", "## Evaluación final", "",
        f"- CASE06 rollout completo alcanzado: {case06['horizons'][str(cfg['evaluation']['full_rollout'])]['reached']}.",
        f"- CASE07 rollout completo alcanzado: {case07['horizons'][str(cfg['evaluation']['full_rollout'])]['reached']}.",
        "", "## Interpretación", "",
        "El checkpoint sólo se promovió si completó CASE06 sin NaN/divergencia y conservó simultáneamente los gates PW y PP. "
        "Una mejora en N60 por sí sola no se consideró suficiente.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def preflight(cfg, data, particles, wall, g_vec, report, replay, specialist_rows,
              parent_model, sigmas, result_root, smoke=False):
    horizons = [5, 10] if smoke else list(cfg["evaluation"]["horizons"])
    zero_n60 = {}
    for case in (cfg["data"]["validation_case"], cfg["data"]["extrapolation_case"]):
        zero_n60[case] = evaluate_n60(
            parent_model, data[case], particles, wall, g_vec, horizons,
            float(cfg["evaluation"]["divergence_penetration"]),
        )
    zero_retention = retention_audit(parent_model, replay, specialist_rows, cfg)
    payload = {
        "checkpoint": {"path": str(resolve(cfg["parent_checkpoint"])), "sha256": sha256(resolve(cfg["parent_checkpoint"]))},
        "architecture": asdict(parent_model.cfg), "parameter_count": sum(p.numel() for p in parent_model.parameters()),
        "data": report, "excluded_cases": cfg["data"]["excluded_cases"],
        "box_nondim": {"min": [0.0, 0.0, 0.0], "max": [6.0, 6.0, 6.0]},
        "gravity_vector": g_vec.tolist(), "sigmas": sigmas,
        "zero_shot_n60": zero_n60, "zero_shot_retention": zero_retention,
    }
    write_json(result_root / "preflight_baseline.json", payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/slgnn_v2/curriculum/S03_N60_ZG_RP.yaml")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-raw-id-audit", action="store_true", help="Sólo para desarrollo; no usar en la corrida oficial")
    args = parser.parse_args()
    cfg = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    if args.skip_raw_id_audit and not args.smoke:
        raise RuntimeError("La corrida oficial no permite omitir la auditoría de Particle_ID")
    suffix = "__smoke" if args.smoke else ""
    ckpt_root = resolve(cfg["output"]["checkpoint_root"]) / f"{cfg['campaign_id']}{suffix}"
    result_root = resolve(cfg["output"]["result_root"]) / f"{cfg['campaign_id']}{suffix}"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)

    parent_model, parent_checkpoint, parent_path, parent_hash = strict_parent_model(cfg)
    data, wall, g_vec, particles, data_report = load_n60_data(cfg, audit_raw=not args.skip_raw_id_audit)
    sigmas = compute_n60_sigmas(data, cfg["data"]["train_cases"])
    replay = load_replay_data(cfg)
    wall_data, pp_data, _ = replay
    replay_particles = (make_particles(next(iter(wall_data.values()))), make_particles(next(iter(pp_data.values()))))
    replay_scales = (
        compute_scales(wall_data, list(wall_data), lambda tr: active_indices(tr, float(cfg["evaluation"]["active_acceleration_threshold"]))),
        compute_scales(pp_data, list(pp_data), lambda tr: pp_near_indices(tr, float(cfg["model"]["r_off"]))),
    )
    specialist_path = resolve(cfg["particle_specialist_checkpoint"])
    specialist_checkpoint = torch.load(specialist_path, map_location="cpu", weights_only=False)
    specialist = fresh_model(cfg, 0)
    specialist.load_state_dict(specialist_checkpoint["model"], strict=True)
    _, _, specialist_rows = suite_metrics(
        specialist, wall_data, pp_data, replay[2],
        float(cfg["evaluation"]["active_acceleration_threshold"]),
    )
    print("== PRE-FLIGHT ==", flush=True)
    baseline = preflight(
        cfg, data, particles, wall, g_vec, data_report, replay, specialist_rows,
        parent_model, sigmas, result_root, smoke=args.smoke,
    )
    print(f"checkpoint={parent_hash} N={particles.radii.numel()} g={g_vec.tolist()}", flush=True)
    if args.preflight_only:
        return

    parent_state = {key: value.detach().clone() for key, value in parent_model.state_dict().items()}
    transfer_seeds = [cfg["seeds"]["transfer"][0]] if args.smoke else list(cfg["seeds"]["transfer"])
    scratch_seeds = [cfg["seeds"]["scratch"][0]] if args.smoke else list(cfg["seeds"]["scratch"])
    runs = []
    for method, seeds in (("transfer", transfer_seeds), ("scratch", scratch_seeds)):
        for seed in seeds:
            print(f"\n== {method.upper()} seed={seed} ==", flush=True)
            runs.append(train_base_run(
                method, int(seed), parent_state, data, particles, wall, g_vec, replay,
                replay_particles, replay_scales, sigmas, specialist_rows, cfg,
                ckpt_root, result_root, smoke=args.smoke,
            ))

    transfer_h64 = [
        run["candidates"][-1] for run in runs if run["method"] == "transfer"
    ]
    h128_executed = bool(
        not args.smoke and cfg["curriculum"]["conditional_h128"]
        and all(candidate["stage"] == "H64" and candidate["stable"] for candidate in transfer_h64)
    )
    if h128_executed:
        print("\n== H128 condicional para transfer y scratch (mismo presupuesto) ==", flush=True)
        for run in runs:
            record = continue_h128(
                run, parent_state, data, particles, wall, g_vec, replay,
                replay_particles, replay_scales, sigmas, specialist_rows, cfg,
            )
            run["history"].extend([] if record is None else [])
    else:
        print("\nH128 no ejecutado: H64 no fue estable en todas las semillas transferidas.", flush=True)

    transfer_candidates = [candidate for run in runs if run["method"] == "transfer" for candidate in run["candidates"]]
    best_n60_record = min(transfer_candidates, key=lambda row: row["n60_score"])
    best_retention_record = min(transfer_candidates, key=lambda row: row["retention"]["gate_score"])
    eligible = [row for row in transfer_candidates if row["retention"]["passed"] and row["stable"]]
    best_joint_record = min(eligible or transfer_candidates, key=lambda row: row["joint_score"])
    selected = {
        "best_n60": copy_selected(best_n60_record, ckpt_root / "best-N60.pt"),
        "best_retention": copy_selected(best_retention_record, ckpt_root / "best-retention.pt"),
        "best_joint": copy_selected(best_joint_record, ckpt_root / "best-joint.pt"),
    }
    final_record = next(
        run["candidates"][-1] for run in runs
        if run["method"] == "transfer" and run["seed"] == best_joint_record["seed"]
    )
    selected["final"] = copy_selected(final_record, ckpt_root / "final.pt")

    selected_model = verify_checkpoint(ckpt_root / "best-joint.pt", cfg)
    full = int(cfg["evaluation"]["full_rollout"])
    case06 = evaluate_n60(
        selected_model, data[cfg["data"]["validation_case"]], particles, wall, g_vec,
        sorted(set(cfg["evaluation"]["horizons"] + [full])),
        float(cfg["evaluation"]["divergence_penetration"]), arrays=True,
    )
    # CASE07 sólo se toca aquí, después de fijar best-joint con CASE06.
    case07 = evaluate_n60(
        selected_model, data[cfg["data"]["extrapolation_case"]], particles, wall, g_vec,
        sorted(set(cfg["evaluation"]["horizons"] + [full])),
        float(cfg["evaluation"]["divergence_penetration"]), arrays=True,
    )
    arrays06 = case06.pop("arrays"); arrays07 = case07.pop("arrays")
    write_json(result_root / "CASE06_metrics.json", case06)
    write_json(result_root / "CASE07_metrics.json", case07)
    np.savez_compressed(result_root / "CASE06_prediction.npz", **arrays06)
    np.savez_compressed(result_root / "CASE07_prediction.npz", **arrays07)
    final_retention = retention_audit(selected_model, replay, specialist_rows, cfg)
    write_json(result_root / "retention_PP2O_PW1.json", final_retention)

    full06 = case06["horizons"][str(full)]
    promoted = bool(best_joint_record["retention"]["passed"] and full06["reached"] and math.isfinite(best_joint_record["n60_score"]))
    if promoted:
        shutil.copy2(ckpt_root / "best-joint.pt", ckpt_root / "promoted-to-next-stage.pt")
        selected["promoted"] = {"sha256": sha256(ckpt_root / "promoted-to-next-stage.pt")}

    if not args.smoke:
        make_comparison_gif(arrays06, result_root / "CASE06_DEM_vs_SLGNN.gif", "CASE06")
        make_comparison_gif(arrays07, result_root / "CASE07_DEM_vs_SLGNN.gif", "CASE07")
    plot_training(runs, result_root / "training_validation_curves.png")
    write_json(result_root / "all_phase_metrics.json", [
        candidate for run in runs for candidate in run["candidates"]
    ])
    write_json(result_root / "transfer_vs_scratch.json", {
        "runs": [{"method": run["method"], "seed": run["seed"], "final": run["candidates"][-1]} for run in runs]
    })
    manifest = {
        "campaign_id": cfg["campaign_id"], "lineage": cfg["lineage"],
        "parent_checkpoint": str(parent_path), "parent_sha256": parent_hash,
        "parent_unchanged_sha256": sha256(parent_path),
        "specialist_checkpoint": str(specialist_path), "specialist_sha256": sha256(specialist_path),
        "seeds": {"transfer": transfer_seeds, "scratch": scratch_seeds},
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "config": cfg, "preflight": baseline,
        "selected": selected, "h128_executed": h128_executed, "promoted": promoted,
        "case07_used_for_selection": False,
    }
    write_json(result_root / "manifest.json", manifest)
    write_report(result_root / "RESULTADOS.md", cfg, runs, selected, case06, case07, h128_executed, promoted)

    for name in ("best-N60.pt", "best-retention.pt", "best-joint.pt", "final.pt"):
        verify_checkpoint(ckpt_root / name, cfg)
    if promoted:
        verify_checkpoint(ckpt_root / "promoted-to-next-stage.pt", cfg)
    if sha256(parent_path) != str(cfg["parent_sha256"]).lower():
        raise RuntimeError("El checkpoint padre cambió durante S03")
    print(f"\nS03 terminada. Promoción={'APROBADA' if promoted else 'NO APROBADA'}", flush=True)
    print(f"Resultados: {result_root}", flush=True)


if __name__ == "__main__":
    main()
