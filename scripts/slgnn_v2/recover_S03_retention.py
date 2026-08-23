"""S03-R1: recupera PP2O/PW1 sin sacrificar N60-ZG.

La selección usa exclusivamente N60-ZG CASE06 y los benchmarks de retención.
CASE07 permanece sellado durante diagnóstico, entrenamiento y selección.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import shutil
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_S03_N60_ZG_RP as s03  # noqa: E402
from train_S02_joint_consolidation import (  # noqa: E402
    backward_rollout as replay_backward_rollout,
    rollout_starts as replay_rollout_starts,
)
from slgnn import SLGNN, SLGNNConfig  # noqa: E402


PARAMETER_COUNT = 24_754
HEAD_PREFIXES = ("head_pp_", "head_pw_")


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def runtime_config(cfg):
    """Adapta sólo el nombre del CASE07 sellado al loader heredado."""
    value = deepcopy(cfg)
    value["data"]["extrapolation_case"] = value["data"]["sealed_extrapolation_case"]
    value["parent_checkpoint"] = value["sources"]["retention_checkpoint"]
    value["parent_sha256"] = value["sources"]["retention_sha256"]
    value["particle_specialist_checkpoint"] = value["sources"]["particle_specialist_checkpoint"]
    value["evaluation"]["h64_stable_max_penetration"] = value["evaluation"]["h500_max_penetration"]
    return value


def model_from_checkpoint(checkpoint, cfg):
    recorded = checkpoint.get("model_config")
    if not isinstance(recorded, dict):
        raise RuntimeError("Checkpoint sin model_config")
    for key, expected in (("hidden", 16), ("layers", 2), ("use_history", True)):
        if recorded.get(key) != expected or cfg["model"][key] != expected:
            raise RuntimeError(
                f"Arquitectura incompatible en {key}: checkpoint={recorded.get(key)!r}, "
                f"config={cfg['model'][key]!r}, esperado={expected!r}"
            )
    allowed = SLGNNConfig().__dict__
    model_cfg = {key: value for key, value in recorded.items() if key in allowed}
    model = SLGNN(SLGNNConfig(**model_cfg))
    model.load_state_dict(checkpoint["model"], strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != PARAMETER_COUNT:
        raise RuntimeError("El modelo no tiene 24 754 parámetros")
    return model


def load_and_verify_sources(cfg):
    source_cfg = cfg["sources"]
    specs = {
        "retention": (
            resolve(source_cfg["retention_checkpoint"]),
            str(source_cfg["retention_sha256"]).lower(),
        ),
        "n60": (
            resolve(source_cfg["n60_checkpoint"]),
            str(source_cfg["n60_sha256"]).lower(),
        ),
    }
    checkpoints = {}
    models = {}
    hashes = {}
    for name, (path, expected_hash) in specs.items():
        actual_hash = s03.sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"SHA-256 incorrecto para {name}: {actual_hash} != {expected_hash}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = model_from_checkpoint(checkpoint, cfg)
        checkpoints[name] = checkpoint
        models[name] = model
        hashes[name] = actual_hash

    left = checkpoints["retention"]["model"]
    right = checkpoints["n60"]["model"]
    if list(left) != list(right):
        raise RuntimeError("Los state_dict no tienen exactamente las mismas claves y orden")
    shape_mismatches = {
        key: [list(left[key].shape), list(right[key].shape)]
        for key in left
        if left[key].shape != right[key].shape
    }
    if shape_mismatches:
        raise RuntimeError(f"Shapes incompatibles entre fuentes: {shape_mismatches}")
    if checkpoints["retention"]["model_config"] != checkpoints["n60"]["model_config"]:
        raise RuntimeError("Los model_config fuente no son idénticos")
    return {
        "paths": {name: str(specs[name][0]) for name in specs},
        "hashes": hashes,
        "checkpoints": checkpoints,
        "models": models,
        "keys": len(left),
        "shapes_equal": True,
        "model_config_equal": True,
        "parameter_count": PARAMETER_COUNT,
        "strict_load": True,
    }


def clone_state(state):
    return {key: value.detach().clone() for key, value in state.items()}


def interpolate_states(retention_state, n60_state, alpha: float):
    result = {}
    for key in retention_state:
        left, right = retention_state[key], n60_state[key]
        if left.is_floating_point():
            result[key] = torch.lerp(left, right, float(alpha))
        else:
            result[key] = left.detach().clone() if alpha < 0.5 else right.detach().clone()
    return result


def replace_prefixes(base_state, donor_state, prefixes):
    result = clone_state(base_state)
    for key in result:
        if key.startswith(tuple(prefixes)):
            result[key] = donor_state[key].detach().clone()
    return result


def diagnostic_states(cfg, source_info):
    retention = source_info["checkpoints"]["retention"]["model"]
    n60 = source_info["checkpoints"]["n60"]["model"]
    states = {}
    kinds = {}
    for alpha in cfg["diagnostics"]["interpolation_alphas"]:
        name = f"interpolation-alpha-{float(alpha):.2f}"
        states[name] = interpolate_states(retention, n60, float(alpha))
        kinds[name] = "interpolation"

    recipes = {
        "s03_with_s02_pp_heads": (n60, retention, ("head_pp_",)),
        "s03_with_s02_pw_heads": (n60, retention, ("head_pw_",)),
        "s03_with_s02_all_heads": (n60, retention, HEAD_PREFIXES),
        "s03_with_s02_processors": (n60, retention, ("proc_",)),
        "s02_with_s03_pp_heads": (retention, n60, ("head_pp_",)),
        "s02_with_s03_pw_heads": (retention, n60, ("head_pw_",)),
        "s02_with_s03_all_heads": (retention, n60, HEAD_PREFIXES),
    }
    enabled = set(cfg["diagnostics"]["head_swaps"])
    for name, (base, donor, prefixes) in recipes.items():
        if name in enabled:
            states[name] = replace_prefixes(base, donor, prefixes)
            kinds[name] = "head_swap"
    unknown = enabled - set(recipes)
    if unknown:
        raise RuntimeError(f"Diagnósticos de trasplante desconocidos: {sorted(unknown)}")
    return states, kinds


def pareto_names(records):
    names = []
    for candidate in records:
        dominated = False
        for other in records:
            if other is candidate:
                continue
            no_worse = (
                other["n60_score"] <= candidate["n60_score"]
                and other["retention"]["gate_score"] <= candidate["retention"]["gate_score"]
            )
            strictly_better = (
                other["n60_score"] < candidate["n60_score"]
                or other["retention"]["gate_score"] < candidate["retention"]["gate_score"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            names.append(candidate["name"])
    return names


def diagnostic_rank(record):
    n60 = record["n60_score"]
    gate = record["retention"]["gate_score"]
    finite_n60 = n60 if math.isfinite(n60) and n60 < 1e8 else 1e8
    stability_penalty = 0.0 if record["stable"] else 3.0
    return (
        math.log1p(finite_n60)
        + math.log1p(gate)
        + 8.0 * max(0.0, gate - 1.0)
        + stability_penalty
    )


def evaluate_diagnostic(name, kind, state, cfg, context, smoke=False):
    model = s03.fresh_model(cfg, 0)
    model.load_state_dict(state, strict=True)
    model.eval()
    horizons = [5, 10] if smoke else list(cfg["diagnostics"]["horizons"])
    n60 = s03.evaluate_n60(
        model,
        context["data"][cfg["data"]["validation_case"]],
        context["particles"],
        context["wall"],
        context["gravity"],
        horizons,
        float(cfg["evaluation"]["divergence_penetration"]),
    )
    retention = s03.retention_audit(model, context["replay"], context["specialist_rows"], cfg)
    score = s03.n60_score(n60, context["sigmas"])
    h500 = n60["horizons"].get("500", {})
    stable = bool(
        h500.get("reached", False)
        and h500.get("max_penetration", float("inf"))
        <= float(cfg["evaluation"]["h500_max_penetration"])
    )
    record = {
        "name": name,
        "kind": kind,
        "n60": n60,
        "n60_score": score,
        "retention": retention,
        "stable": stable,
    }
    record["diagnostic_rank"] = diagnostic_rank(record)
    return record


def run_diagnostics(cfg, source_info, context, result_root, smoke=False):
    states, kinds = diagnostic_states(cfg, source_info)
    records = []
    for index, (name, state) in enumerate(states.items(), 1):
        print(f"  diagnóstico {index:02d}/{len(states):02d}: {name}", flush=True)
        records.append(evaluate_diagnostic(name, kinds[name], state, cfg, context, smoke=smoke))
    pareto = pareto_names(records)
    candidates = sorted(
        (record for record in records if record["name"] in pareto),
        key=diagnostic_rank,
    )
    top_k = min(int(cfg["diagnostics"]["pareto_initializations"]), len(candidates))
    selected = [record["name"] for record in candidates[:top_k]]
    if not selected:
        raise RuntimeError("El diagnóstico no produjo inicializaciones Pareto")
    common = {
        "pareto_candidates": pareto,
        "selected_initializations": selected,
        "selection_uses_case07": False,
    }
    write_json(
        result_root / "interpolation_audit.json",
        {**common, "records": [row for row in records if row["kind"] == "interpolation"]},
    )
    write_json(
        result_root / "head_swap_diagnostics.json",
        {**common, "records": [row for row in records if row["kind"] == "head_swap"]},
    )
    return records, selected, states


def load_or_run_diagnostics(cfg, source_info, context, result_root, smoke=False):
    interpolation_path = result_root / "interpolation_audit.json"
    head_path = result_root / "head_swap_diagnostics.json"
    states, _ = diagnostic_states(cfg, source_info)
    cache_root = cfg["diagnostics"].get("cache_result_root")
    if (
        not interpolation_path.exists()
        and not head_path.exists()
        and cache_root
    ):
        cached = resolve(cache_root)
        cached_interpolation = cached / "interpolation_audit.json"
        cached_head = cached / "head_swap_diagnostics.json"
        if cached_interpolation.exists() and cached_head.exists():
            interpolation = load_json(cached_interpolation)
            head = load_json(cached_head)
            records = interpolation["records"] + head["records"]
            pareto = pareto_names(records)
            ranked = sorted(
                (record for record in records if record["name"] in pareto),
                key=diagnostic_rank,
            )
            selected = [
                record["name"]
                for record in ranked[
                    : int(cfg["diagnostics"]["pareto_initializations"])
                ]
            ]
            shared = {
                "pareto_candidates": pareto,
                "selected_initializations": selected,
                "selection_uses_case07": False,
                "reused_from": str(cached),
            }
            write_json(
                interpolation_path,
                {
                    **shared,
                    "records": [
                        record
                        for record in records
                        if record["kind"] == "interpolation"
                    ],
                },
            )
            write_json(
                head_path,
                {
                    **shared,
                    "records": [
                        record for record in records if record["kind"] == "head_swap"
                    ],
                },
            )
            print(f"  reutilizando diagnóstico verificado de {cached}", flush=True)
            return records, selected, states
    if not smoke and interpolation_path.exists() and head_path.exists():
        interpolation = load_json(interpolation_path)
        head = load_json(head_path)
        records = interpolation["records"] + head["records"]
        pareto = pareto_names(records)
        ranked = sorted(
            (record for record in records if record["name"] in pareto),
            key=diagnostic_rank,
        )
        selected = [
            record["name"]
            for record in ranked[: int(cfg["diagnostics"]["pareto_initializations"])]
        ]
        if any(name not in states for name in selected):
            raise RuntimeError("Una inicialización diagnóstica ya no es reproducible")
        print("  reutilizando diagnóstico preflight verificado", flush=True)
        return records, selected, states
    return run_diagnostics(cfg, source_info, context, result_root, smoke=smoke)


def load_context(cfg):
    data, wall, gravity, particles, data_report = s03.load_n60_data(cfg, audit_raw=True)
    sigmas = s03.compute_n60_sigmas(data, cfg["data"]["train_cases"])
    replay = s03.load_replay_data(cfg)
    wall_data, pp_data, replay_wall = replay
    replay_particles = (
        s03.make_particles(next(iter(wall_data.values()))),
        s03.make_particles(next(iter(pp_data.values()))),
    )
    replay_scales = (
        s03.compute_scales(
            wall_data,
            list(wall_data),
            lambda trajectory: s03.active_indices(
                trajectory, float(cfg["evaluation"]["active_acceleration_threshold"])
            ),
        ),
        s03.compute_scales(
            pp_data,
            list(pp_data),
            lambda trajectory: s03.pp_near_indices(trajectory, float(cfg["model"]["r_off"])),
        ),
    )
    specialist_checkpoint = torch.load(
        resolve(cfg["sources"]["particle_specialist_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    specialist = s03.fresh_model(cfg, 0)
    specialist.load_state_dict(specialist_checkpoint["model"], strict=True)
    _, _, specialist_rows = s03.suite_metrics(
        specialist,
        wall_data,
        pp_data,
        replay_wall,
        float(cfg["evaluation"]["active_acceleration_threshold"]),
    )
    return {
        "data": data,
        "wall": wall,
        "gravity": gravity,
        "particles": particles,
        "data_report": data_report,
        "sigmas": sigmas,
        "replay": replay,
        "replay_particles": replay_particles,
        "replay_scales": replay_scales,
        "specialist_rows": specialist_rows,
    }


def random_choice(values, generator):
    return int(values[int(torch.randint(len(values), (), generator=generator))])


def replay_domain_loss(model, context, cfg, domain, iteration, generator):
    wall_data, pp_data, wall = context["replay"]
    wall_particles, pp_particles = context["replay_particles"]
    wall_scales, pp_scales = context["replay_scales"]
    if domain == "particle":
        sequence = [str(value) for value in cfg["sampling"]["particle_case_sequence"]]
        case = sequence[iteration % len(sequence)]
        trajectory = pp_data[case]
        choices = s03.pp_near_indices(trajectory, model.cfg.r_off)
        index = random_choice(choices, generator)
        particles, boundary, scales = pp_particles, None, pp_scales
    elif domain == "wall":
        cases = [str(value) for value in cfg["data"]["wall_cases"]]
        case = cases[iteration % len(cases)]
        trajectory = wall_data[case]
        active = s03.active_indices(
            trajectory, float(cfg["evaluation"]["active_acceleration_threshold"])
        )
        near = s03.wall_near_indices(trajectory, model.cfg.g_off)
        choices = active if iteration % 2 == 0 else near
        index = random_choice(choices, generator)
        particles, boundary, scales = wall_particles, wall, wall_scales
    else:
        raise ValueError(domain)
    acceleration, angular = s03.targets(trajectory)
    output = model(
        trajectory.q[index],
        trajectory.v[index],
        trajectory.omega[index],
        particles,
        wall=boundary,
    )
    return (
        s03.normalized_acceleration_loss(
            output, acceleration[index], angular[index], scales
        )
        + s03.auxiliary_loss(output, particles, cfg)
    )


def one_step_recovery(model, anchor, context, cfg, seed, iterations, history):
    optimizer = s03.optimizer_for(model, cfg, 1.0)
    pools, refs = s03.activity_pools(
        context["data"],
        cfg["data"]["train_cases"],
        float(cfg["sampling"]["activity_threshold"]),
    )
    n60_generator = torch.Generator().manual_seed(seed + 10_000)
    replay_generator = torch.Generator().manual_seed(seed + 20_000)
    low_fraction = float(cfg["sampling"]["low_activity_fraction"])
    fractions = cfg["sampling"]
    loss_cfg = cfg["loss"]
    cases = list(cfg["data"]["train_cases"])
    model.train()
    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        case = cases[iteration % len(cases)]
        trajectory = context["data"][case]
        use_low = float(torch.rand((), generator=n60_generator)) < low_fraction
        index = random_choice(pools[case]["low" if use_low else "active"], n60_generator)
        acceleration, angular = refs[case]
        output = model(
            trajectory.q[index],
            trajectory.v[index],
            trajectory.omega[index],
            context["particles"],
            wall=context["wall"],
            g_vec=context["gravity"],
        )
        n60_loss = (
            float(loss_cfg["lambda_acceleration"])
            * s03.acceleration_loss(output.a, acceleration[index], context["sigmas"]["sigma_a"])
            + float(loss_cfg["lambda_angular_acceleration"])
            * s03.angular_acceleration_loss(
                output.alpha, angular[index], context["sigmas"]["sigma_alpha"]
            )
            + s03.auxiliary_loss(output, context["particles"], cfg)
        )
        particle_loss = replay_domain_loss(
            model, context, cfg, "particle", iteration, replay_generator
        )
        wall_loss = replay_domain_loss(
            model, context, cfg, "wall", iteration, replay_generator
        )
        anchor_loss = s03.l2sp_loss(model, anchor)
        total = (
            float(fractions["n60_fraction"]) * n60_loss
            + float(fractions["replay_particle_fraction"]) * particle_loss
            + float(fractions["replay_wall_fraction"]) * wall_loss
            + float(loss_cfg["lambda_l2sp"]) * anchor_loss
        )
        if not torch.isfinite(total):
            raise FloatingPointError(f"Warmup no finito seed={seed} iter={iteration}")
        total.backward()
        grad = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(cfg["optimization"]["grad_clip"])
        )
        optimizer.step()
        history.append(
            {
                "stage": "one-step",
                "iteration": iteration,
                "n60_loss": float(n60_loss.detach()),
                "particle_loss": float(particle_loss.detach()),
                "wall_loss": float(wall_loss.detach()),
                "l2sp_loss": float(anchor_loss.detach()),
                "total_loss": float(total.detach()),
                "grad_norm": float(grad),
                "validation_score": "",
            }
        )
        if iteration % 50 == 0 or iteration == iterations - 1:
            print(
                f"    one-step {iteration:4d}/{iterations}: "
                f"N60={float(n60_loss.detach()):.4g} "
                f"PP={float(particle_loss.detach()):.4g} "
                f"PW={float(wall_loss.detach()):.4g}",
                flush=True,
            )
    return optimizer


def rollout_recovery(model, anchor, context, cfg, seed, horizon, iterations, history):
    optimizer = s03.optimizer_for(
        model, cfg, float(cfg["optimization"]["rollout_lr_multiplier"])
    )
    pools = s03.rollout_pools(
        context["data"],
        cfg["data"]["train_cases"],
        horizon,
        float(cfg["sampling"]["activity_threshold"]),
    )
    n60_generator = torch.Generator().manual_seed(seed + 30_000 + horizon)
    replay_generator = torch.Generator().manual_seed(seed + 40_000 + horizon)
    low_fraction = float(cfg["sampling"]["low_activity_fraction"])
    cases = list(cfg["data"]["train_cases"])
    wall_data, pp_data, replay_wall = context["replay"]
    wall_particles, pp_particles = context["replay_particles"]
    wall_scales, pp_scales = context["replay_scales"]
    threshold = float(cfg["evaluation"]["active_acceleration_threshold"])
    wall_starts = {
        case: replay_rollout_starts(trajectory, horizon, threshold)
        for case, trajectory in wall_data.items()
    }
    pp_starts = {
        case: replay_rollout_starts(trajectory, horizon, threshold)
        for case, trajectory in pp_data.items()
    }
    wall_cases = [str(value) for value in cfg["data"]["wall_cases"]]
    pp_sequence = [str(value) for value in cfg["sampling"]["particle_case_sequence"]]
    scores = []
    model.train()
    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        case = cases[iteration % len(cases)]
        use_low = float(torch.rand((), generator=n60_generator)) < low_fraction
        starts = pools[case]["low" if use_low else "active"]
        start = random_choice(starts, n60_generator)
        n60_value = s03.rollout_backward(
            model,
            context["data"][case],
            context["particles"],
            context["wall"],
            context["gravity"],
            start,
            horizon,
            context["sigmas"],
            cfg,
            n60_generator,
            float(cfg["sampling"]["n60_fraction"]),
        )
        particle_case = pp_sequence[iteration % len(pp_sequence)]
        particle_start = random_choice(pp_starts[particle_case], replay_generator)
        particle_loss = replay_backward_rollout(
            model,
            pp_data[particle_case],
            pp_particles,
            None,
            particle_start,
            horizon,
            pp_scales,
            float(cfg["noise"]["replay_sigma_q"]),
            int(cfg["curriculum"]["tbptt_chunk"]),
            float(cfg["sampling"]["replay_particle_fraction"]),
            float(cfg["noise"]["sigma_q"]),
            float(cfg["noise"]["sigma_v"]),
            replay_generator,
        )
        wall_case = wall_cases[iteration % len(wall_cases)]
        wall_start = random_choice(wall_starts[wall_case], replay_generator)
        wall_loss = replay_backward_rollout(
            model,
            wall_data[wall_case],
            wall_particles,
            replay_wall,
            wall_start,
            horizon,
            wall_scales,
            float(cfg["noise"]["replay_sigma_q"]),
            int(cfg["curriculum"]["tbptt_chunk"]),
            float(cfg["sampling"]["replay_wall_fraction"]),
            float(cfg["noise"]["sigma_q"]),
            float(cfg["noise"]["sigma_v"]),
            replay_generator,
        )
        anchor_loss = s03.l2sp_loss(model, anchor)
        extra = float(cfg["loss"]["lambda_l2sp"]) * anchor_loss
        if not torch.isfinite(extra):
            raise FloatingPointError(f"Replay no finito H={horizon} iter={iteration}")
        extra.backward()
        grad = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(cfg["optimization"]["grad_clip"])
        )
        optimizer.step()
        validation_score = ""
        if (
            iteration % int(cfg["evaluation"]["validation_every"]) == 0
            or iteration == iterations - 1
        ):
            model.eval()
            validation = s03.evaluate_n60(
                model,
                context["data"][cfg["data"]["validation_case"]],
                context["particles"],
                context["wall"],
                context["gravity"],
                [100],
                float(cfg["evaluation"]["divergence_penetration"]),
            )
            validation_score = s03.n60_score(validation, context["sigmas"])
            scores.append(validation_score)
            if (
                len(scores) >= 3
                and scores[-1] > scores[-2] > scores[-3]
            ):
                for group in optimizer.param_groups:
                    group["lr"] *= float(cfg["optimization"]["oscillation_decay"])
            model.train()
            print(
                f"    H={horizon:2d} {iteration:3d}/{iterations}: "
                f"train={n60_value:.4g} val={validation_score:.4g}",
                flush=True,
            )
        history.append(
            {
                "stage": f"H{horizon}",
                "iteration": iteration,
                "n60_loss": n60_value,
                "particle_loss": particle_loss,
                "wall_loss": wall_loss,
                "l2sp_loss": float(anchor_loss.detach()),
                "total_loss": (
                    n60_value
                    + particle_loss
                    + wall_loss
                    + float(cfg["loss"]["lambda_l2sp"])
                    * float(anchor_loss.detach())
                ),
                "grad_norm": float(grad),
                "validation_score": validation_score,
            }
        )
    return optimizer


def stage_record(model, cfg, context, seed, initialization, stage, smoke=False):
    horizons = [5, 10] if smoke else list(cfg["evaluation"]["horizons"])
    n60 = s03.evaluate_n60(
        model,
        context["data"][cfg["data"]["validation_case"]],
        context["particles"],
        context["wall"],
        context["gravity"],
        horizons,
        float(cfg["evaluation"]["divergence_penetration"]),
    )
    retention = s03.retention_audit(model, context["replay"], context["specialist_rows"], cfg)
    n60_value = s03.n60_score(n60, context["sigmas"])
    h500 = n60["horizons"].get("500", {})
    stable = bool(
        h500.get("reached", False)
        and h500.get("max_penetration", float("inf"))
        <= float(cfg["evaluation"]["h500_max_penetration"])
    )
    joint = (
        n60_value
        + math.log1p(retention["gate_score"])
        + 10.0 * max(0.0, retention["gate_score"] - 1.0)
        + (0.0 if stable or smoke else 10.0)
    )
    return {
        "seed": seed,
        "initialization": initialization,
        "stage": stage,
        "n60": n60,
        "n60_score": n60_value,
        "retention": retention,
        "stable": stable,
        "joint_score": joint,
    }


def save_checkpoint(path, model, optimizer, cfg, seed, initialization, stage, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": asdict(model.cfg),
            "optimizer": optimizer.state_dict(),
            "optimizer_reused": False,
            "config": cfg,
            "campaign_id": cfg["campaign_id"],
            "lineage": cfg["lineage"],
            "seed": seed,
            "initialization": initialization,
            "stage": stage,
            "retention_anchor_checkpoint": cfg["sources"]["retention_checkpoint"],
            "retention_anchor_sha256": cfg["sources"]["retention_sha256"],
            "n60_source_checkpoint": cfg["sources"]["n60_checkpoint"],
            "n60_source_sha256": cfg["sources"]["n60_sha256"],
            "evaluation": record,
        },
        path,
    )


def verify_saved_checkpoint(path, cfg):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = model_from_checkpoint(checkpoint, cfg)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


def train_seed(cfg, context, source_info, initial_state, initialization, seed, roots, smoke):
    checkpoint_root, result_root = roots
    run_name = f"seed-{seed}__init-{initialization}"
    run_checkpoint = checkpoint_root / run_name
    run_result = result_root / run_name
    run_checkpoint.mkdir(parents=True, exist_ok=False)
    run_result.mkdir(parents=True, exist_ok=False)
    model = s03.fresh_model(cfg, seed)
    model.load_state_dict(clone_state(initial_state), strict=True)
    anchor = clone_state(source_info["checkpoints"]["retention"]["model"])
    history = []
    records = []
    started = time.time()
    warmup_iterations = 2 if smoke else int(cfg["curriculum"]["warmup_iterations"])
    optimizer = one_step_recovery(
        model, anchor, context, cfg, seed, warmup_iterations, history
    )

    def audit(stage, current_optimizer):
        model.eval()
        record = stage_record(
            model, cfg, context, seed, initialization, stage, smoke=smoke
        )
        path = run_checkpoint / f"stage-{stage}.pt"
        save_checkpoint(
            path, model, current_optimizer, cfg, seed, initialization, stage, record
        )
        verify_saved_checkpoint(path, cfg)
        record["checkpoint"] = str(path)
        record["checkpoint_sha256"] = s03.sha256(path)
        records.append(record)
        write_json(run_result / "phase_metrics.json", records)
        return record

    audit("one-step", optimizer)
    horizons = [4] if smoke else list(cfg["curriculum"]["horizons"])
    iterations = [1] if smoke else list(cfg["curriculum"]["iterations_per_horizon"])
    for horizon, count in zip(horizons, iterations):
        optimizer = rollout_recovery(
            model,
            anchor,
            context,
            cfg,
            seed,
            int(horizon),
            int(count),
            history,
        )
        audit(f"H{horizon}", optimizer)
    final_path = run_checkpoint / "final.pt"
    save_checkpoint(
        final_path,
        model,
        optimizer,
        cfg,
        seed,
        initialization,
        records[-1]["stage"],
        records[-1],
    )
    verify_saved_checkpoint(final_path, cfg)
    with (run_result / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    write_json(
        run_result / "run_metadata.json",
        {
            "seed": seed,
            "initialization": initialization,
            "elapsed_seconds": time.time() - started,
            "final_checkpoint": str(final_path),
            "case07_used": False,
        },
    )
    return {
        "seed": seed,
        "initialization": initialization,
        "records": records,
        "history": history,
        "elapsed_seconds": time.time() - started,
    }


def copy_selected(record, destination):
    shutil.copy2(record["checkpoint"], destination)
    return {
        "source": record["checkpoint"],
        "destination": str(destination),
        "sha256": s03.sha256(destination),
        "record": record,
    }


def write_report(path, cfg, selected, full_metrics, retention, promoted, source_unchanged):
    horizons = full_metrics["horizons"]
    full = horizons.get(str(cfg["evaluation"]["full_rollout"]))
    h500 = horizons.get("500")
    smoke = full is None or h500 is None
    if full is None:
        full = horizons[sorted(horizons, key=int)[-1]]
    if h500 is None:
        h500 = full
    pp4 = next(
        row["rollout_v_rmse_post"]
        for row in retention["pp_rows"]
        if row["case"] == "4x"
    )
    lines = [
        "# S03-R1-RETENTION — resultados",
        "",
        f"- Ejecución: **{'SMOKE' if smoke else 'OFICIAL'}**.",
        f"- Promoción a S04-N60-G-RP: **{'APROBADA' if promoted else 'NO APROBADA'}**.",
        f"- Inicialización seleccionada: `{selected['best_joint']['record']['initialization']}`.",
        f"- Semilla seleccionada: `{selected['best_joint']['record']['seed']}`.",
        "- CASE07 permaneció sellado y no intervino en diagnóstico, entrenamiento ni selección.",
        f"- Checkpoints fuente inalterados: **{'sí' if source_unchanged else 'no'}**.",
        "",
        "## Gates del checkpoint conjunto",
        "",
        "| gate | resultado | requisito |",
        "|---|---:|---:|",
        f"| PW mediana RMSE v | {retention['summary']['wall_v_median']:.6g} | ≤ {cfg['evaluation']['gate_wall_median']} |",
        f"| peor PP/especialista | {max(retention['pp_multipliers'].values()):.6g}× | ≤ {cfg['evaluation']['gate_pp_case_multiplier']}× |",
        f"| PP-4x absoluto | {pp4:.6g} | ≤ {cfg['evaluation']['gate_pp4x_absolute']} |",
        f"| CASE06 {'smoke' if smoke else 'H500'} alcanzado | {h500['reached']} | True |",
        f"| penetración máxima {'smoke' if smoke else 'H500'} | {h500['max_penetration']:.6g} d_p | ≤ {cfg['evaluation']['h500_max_penetration']} d_p |",
        f"| CASE06 {'smoke' if smoke else 'H1500'} alcanzado | {full['reached']} | True |",
        "",
        "## Checkpoints seleccionados",
        "",
        f"- `best-N60-ZG.pt`: semilla {selected['best_n60']['record']['seed']}, fase {selected['best_n60']['record']['stage']}.",
        f"- `best-retention.pt`: semilla {selected['best_retention']['record']['seed']}, fase {selected['best_retention']['record']['stage']}.",
        f"- `best-joint.pt`: semilla {selected['best_joint']['record']['seed']}, fase {selected['best_joint']['record']['stage']}.",
    ]
    if promoted:
        lines += [
            "",
            "`promoted-to-S04-N60-G-RP.pt` es una copia verificada de `best-joint.pt` y carga con `strict=True`.",
        ]
    else:
        lines += [
            "",
            "No se creó `promoted-to-S04-N60-G-RP.pt`; por contrato, S04 no debe comenzar.",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/slgnn_v2/curriculum/S03_R1_retention.yaml")
    parser.add_argument("--preflight", "--preflight-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.preflight and args.smoke:
        raise RuntimeError("--preflight y --smoke son mutuamente excluyentes")

    cfg = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    cfg = runtime_config(cfg)
    suffix = "__smoke" if args.smoke else ""
    result_root = resolve(cfg["output"]["result_root"]) / f"{cfg['campaign_id']}{suffix}"
    checkpoint_root = resolve(cfg["output"]["checkpoint_root"]) / f"{cfg['campaign_id']}{suffix}"
    result_root.mkdir(parents=True, exist_ok=True)

    print("== VERIFICACIÓN DE FUENTES ==", flush=True)
    source_info = load_and_verify_sources(cfg)
    hashes_before = dict(source_info["hashes"])
    print(
        f"  S02={hashes_before['retention']} S03={hashes_before['n60']} "
        f"keys={source_info['keys']} params={source_info['parameter_count']}",
        flush=True,
    )
    print("== CARGA DE DATOS Y REPLAY ==", flush=True)
    context = load_context(cfg)
    print("== DIAGNÓSTICO SIN ENTRENAMIENTO ==", flush=True)
    diagnostic_records, selected_initializations, states = load_or_run_diagnostics(
        cfg, source_info, context, result_root, smoke=args.smoke
    )
    preflight_payload = {
        "campaign_id": cfg["campaign_id"],
        "sources": {
            "paths": source_info["paths"],
            "hashes_before": hashes_before,
            "keys": source_info["keys"],
            "shapes_equal": source_info["shapes_equal"],
            "model_config_equal": source_info["model_config_equal"],
            "strict_load": source_info["strict_load"],
            "parameter_count": source_info["parameter_count"],
        },
        "data": context["data_report"],
        "sigmas": context["sigmas"],
        "selected_initializations": selected_initializations,
        "optimizer_created": False,
        "case07_used_for_selection": False,
        "config": cfg,
    }
    write_json(result_root / "preflight.json", preflight_payload)
    if args.preflight:
        hashes_after = {
            "retention": s03.sha256(resolve(cfg["sources"]["retention_checkpoint"])),
            "n60": s03.sha256(resolve(cfg["sources"]["n60_checkpoint"])),
        }
        if hashes_after != hashes_before:
            raise RuntimeError("Un checkpoint fuente cambió durante el preflight")
        print(f"Preflight completado: {result_root}", flush=True)
        return

    checkpoint_root.mkdir(parents=True, exist_ok=True)
    seeds = [int(cfg["seeds"][0])] if args.smoke else [int(value) for value in cfg["seeds"]]
    runs = []
    for index, seed in enumerate(seeds):
        initialization = selected_initializations[index % len(selected_initializations)]
        print(
            f"== RECUPERACIÓN seed={seed} init={initialization} ==",
            flush=True,
        )
        runs.append(
            train_seed(
                cfg,
                context,
                source_info,
                states[initialization],
                initialization,
                seed,
                (checkpoint_root, result_root),
                args.smoke,
            )
        )

    candidates = [record for run in runs for record in run["records"]]
    best_n60 = min(candidates, key=lambda record: record["n60_score"])
    best_retention = min(
        candidates, key=lambda record: record["retention"]["gate_score"]
    )
    eligible = [
        record
        for record in candidates
        if record["retention"]["passed"] and (record["stable"] or args.smoke)
    ]
    best_joint = min(eligible or candidates, key=lambda record: record["joint_score"])
    selected = {
        "best_n60": copy_selected(best_n60, checkpoint_root / "best-N60-ZG.pt"),
        "best_retention": copy_selected(
            best_retention, checkpoint_root / "best-retention.pt"
        ),
        "best_joint": copy_selected(best_joint, checkpoint_root / "best-joint.pt"),
    }
    selected_model = verify_saved_checkpoint(checkpoint_root / "best-joint.pt", cfg)

    if args.smoke:
        final_horizons = [5, 10]
    else:
        final_horizons = sorted(
            set(
                list(cfg["evaluation"]["horizons"])
                + [int(cfg["evaluation"]["full_rollout"])]
            )
        )
    full_metrics = s03.evaluate_n60(
        selected_model,
        context["data"][cfg["data"]["validation_case"]],
        context["particles"],
        context["wall"],
        context["gravity"],
        final_horizons,
        float(cfg["evaluation"]["divergence_penetration"]),
    )
    retention = s03.retention_audit(
        selected_model, context["replay"], context["specialist_rows"], cfg
    )
    write_json(result_root / "N60_ZG_CASE06_metrics.json", full_metrics)
    write_json(result_root / "retention_PP2O_PW1.json", retention)

    if args.smoke:
        promoted = False
    else:
        required = ["100", "250", "500", str(cfg["evaluation"]["full_rollout"])]
        horizons_pass = all(
            full_metrics["horizons"][horizon]["reached"] for horizon in required
        )
        h500 = full_metrics["horizons"]["500"]
        promoted = bool(
            retention["passed"]
            and horizons_pass
            and h500["max_penetration"]
            <= float(cfg["evaluation"]["h500_max_penetration"])
        )
    if promoted:
        selected["promoted"] = copy_selected(
            best_joint,
            checkpoint_root / "promoted-to-S04-N60-G-RP.pt",
        )
        verify_saved_checkpoint(
            checkpoint_root / "promoted-to-S04-N60-G-RP.pt", cfg
        )

    hashes_after = {
        "retention": s03.sha256(resolve(cfg["sources"]["retention_checkpoint"])),
        "n60": s03.sha256(resolve(cfg["sources"]["n60_checkpoint"])),
    }
    source_unchanged = hashes_after == hashes_before
    if not source_unchanged:
        raise RuntimeError("Un checkpoint fuente cambió durante S03-R1")

    manifest = {
        "campaign_id": cfg["campaign_id"],
        "lineage": cfg["lineage"],
        "sources": {
            "paths": source_info["paths"],
            "hashes_before": hashes_before,
            "hashes_after": hashes_after,
            "unchanged": source_unchanged,
            "keys": source_info["keys"],
            "shapes_equal": source_info["shapes_equal"],
            "model_config_equal": source_info["model_config_equal"],
            "strict_load": source_info["strict_load"],
            "parameter_count": source_info["parameter_count"],
        },
        "seeds": seeds,
        "selected_initializations": selected_initializations,
        "diagnostic_candidates": [
            {
                "name": record["name"],
                "kind": record["kind"],
                "n60_score": record["n60_score"],
                "retention_gate_score": record["retention"]["gate_score"],
                "stable": record["stable"],
            }
            for record in diagnostic_records
        ],
        "selected": selected,
        "promoted": promoted,
        "case07_used_for_selection": False,
        "case07_evaluated": False,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "config": cfg,
        "elapsed_seconds_by_seed": {
            str(run["seed"]): run["elapsed_seconds"] for run in runs
        },
    }
    write_json(result_root / "manifest.json", manifest)
    write_report(
        result_root / "RESULTADOS.md",
        cfg,
        selected,
        full_metrics,
        retention,
        promoted,
        source_unchanged,
    )
    for name in ("best-N60-ZG.pt", "best-retention.pt", "best-joint.pt"):
        verify_saved_checkpoint(checkpoint_root / name, cfg)
    print(
        f"S03-R1 terminada. Promoción={'APROBADA' if promoted else 'NO APROBADA'}",
        flush=True,
    )
    print(f"Resultados: {result_root}", flush=True)


if __name__ == "__main__":
    main()
