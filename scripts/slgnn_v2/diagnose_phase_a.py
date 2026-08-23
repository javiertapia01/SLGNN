"""Execute the read-only Phase A audit defined by the experimental protocol.

This script never changes model weights, SLGNN.forward, the integrator, or the
dataset.  It writes a complete, reproducible diagnostic bundle below
``results/slgnn_v2/diagnostics/phase_a/<run_id>``.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from slgnn.diagnostics import (
    REGIME_NAMES, contact_age_bucket, integrator_position_residuals,
    max_geometric_penetration, ood_coverage, quantile_bin, robust_summary,
    select_stratified_starts, snapshot_geometry, update_contact_ages,
)
from slgnn.experiment import load_case_by_name, load_checkpoint, load_split
from slgnn.integrator import semi_implicit_step


ROOT = Path(__file__).resolve().parents[2]
CASES = [f"CASE{i:02d}" for i in range(1, 8)]
TRAIN_CASES = CASES[:5]
HORIZONS = [1, 4, 8, 16, 25, 50, 100, 200]
INTENSITY_FEATURES = [
    "compression", "pp_delta_max", "pw_delta", "pp_approach",
    "pp_tangential", "pw_approach", "pw_tangential", "contact_degree",
    "speed", "spin", "a_ref_norm", "alpha_ref_norm",
]
METRICS = ["e_a", "e_alpha", "e_dv", "e_q1"]


def _json_default(x):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    raise TypeError(type(x).__name__)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def summarize_checkpoint(ck, checkpoint: Path):
    state = ck["model"]
    return {
        "path": str(checkpoint.resolve()),
        "size_bytes": checkpoint.stat().st_size,
        "sha256": sha256(checkpoint),
        "tag": ck.get("tag"),
        "config": ck.get("config"),
        "model_config": ck.get("model_config"),
        "sigmas": ck.get("sigmas"),
        "state_dict_key_count": len(state),
        "state_dict_keys": list(state),
        "parameter_count": int(sum(v.numel() for v in state.values())),
        "contains_optimizer": "optim" in ck,
    }


def environment_text():
    lines = [
        f"python={sys.version}",
        f"executable={sys.executable}",
        f"platform={platform.platform()}",
        f"processor={platform.processor()}",
        f"cpu_count_logical={os.cpu_count()}",
        f"torch={torch.__version__}",
        f"cuda_available={torch.cuda.is_available()}",
        f"torch_cuda={torch.version.cuda}",
    ]
    if torch.cuda.is_available():
        lines.extend([
            f"gpu={torch.cuda.get_device_name(0)}",
            f"gpu_count={torch.cuda.device_count()}",
        ])
    try:
        import psutil
        lines.append(f"ram_bytes={psutil.virtual_memory().total}")
    except Exception:
        lines.append("ram_bytes=unknown")
    lines.append("\n[pip freeze]")
    lines.append(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True
    ))
    return "\n".join(lines)


def preflight(args, out_dir, ck, cfg_yaml):
    checkpoint = Path(args.checkpoint).resolve()
    summary = summarize_checkpoint(ck, checkpoint)
    embedded = ck.get("config")
    if embedded is None or ck.get("model_config") is None or ck.get("sigmas") is None:
        raise RuntimeError("Checkpoint missing embedded config/model_config/sigmas")

    yaml_text = yaml.safe_dump(cfg_yaml, sort_keys=True, allow_unicode=True).splitlines()
    embedded_text = yaml.safe_dump(
        embedded, sort_keys=True, allow_unicode=True
    ).splitlines()
    diff = "\n".join(difflib.unified_diff(
        yaml_text, embedded_text, fromfile=str(args.config),
        tofile="checkpoint:config", lineterm=""
    ))
    (out_dir / "config_diff.txt").write_text(
        diff if diff else "No differences: embedded config equals YAML.\n",
        encoding="utf-8",
    )
    write_json(out_dir / "checkpoint_summary.json", summary)
    (out_dir / "environment.txt").write_text(environment_text(), encoding="utf-8")

    dirty = git("status", "--short")
    metadata = {
        "run_id": args.run_id,
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repository": str(ROOT),
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "git_dirty": bool(dirty),
        "git_status_short": dirty.splitlines(),
        "checkpoint_parent": str(checkpoint),
        "checkpoint_sha256": summary["sha256"],
        "checkpoint_tag": summary["tag"],
        "dataset": embedded["data"]["dataset"],
        "train_cases": TRAIN_CASES,
        "val_case": "CASE06",
        "test_case": "CASE07",
        "seed": int(args.seed),
        "device": "cpu",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "active_channels": ["V", "R", "H"],
        "horizon": 200,
        "tbptt_chunk": embedded["curriculum"]["tbptt_chunk"],
        "yaml_matches_embedded_config": not bool(diff),
        "analysis_max_steps": args.max_steps,
        "analysis_code_only_read_checkpoint": True,
    }
    write_json(out_dir / "metadata.json", metadata)
    return embedded, metadata


def _alloc(shape):
    keys = [
        *METRICS, "regime", "pp_delta_max", "pp_delta_mean", "pw_delta",
        "pp_approach", "pp_tangential", "pw_approach", "pw_tangential",
        "contact_degree", "speed", "spin", "a_ref_norm", "alpha_ref_norm",
        "compression", "int_semi_implicit", "int_explicit", "int_midpoint",
        "aV_norm", "aR_norm", "aH_norm", "rhoV", "rhoR", "rhoH",
        "contact_acc_norm",
    ]
    out = {k: np.empty(shape, dtype=np.float32) for k in keys}
    out["regime"] = np.empty(shape, dtype=np.int8)
    return out


def analyze_case(case_name, tr, model, particles, wall, g_vec, sigmas,
                 max_steps=None):
    total_steps = tr.q.shape[0] - 1
    steps = min(total_steps, max_steps) if max_steps else total_steps
    n = tr.q.shape[1]
    data = _alloc((steps, n))
    contact_records = []
    snapshot_records = []
    event_records = []
    pp_ages, pw_ages = {}, {}
    dt = float(tr.dt)
    diameter = float(2 * particles.radii[0])
    mass = particles.m.to(tr.q)
    g_force = mass.unsqueeze(-1) * g_vec.to(tr.q)
    t0 = time.perf_counter()

    for k in range(steps):
        q, v, w = tr.q[k], tr.v[k], tr.omega[k]
        q1, v1, w1 = tr.q[k + 1], tr.v[k + 1], tr.omega[k + 1]
        a_ref = (v1 - v) / dt
        alpha_ref = (w1 - w) / dt
        with torch.no_grad():
            out = model(q, v, w, particles, wall=wall, t=k * dt, g_vec=g_vec)
        geo = snapshot_geometry(q, v, w, particles, wall, model.cfg, k * dt)

        v_hat = v + dt * out.a
        w_hat = w + dt * out.alpha
        q_hat = q + dt * v_hat
        data["e_a"][k] = ((out.a - a_ref).norm(dim=-1) / sigmas["sigma_a"]).numpy()
        data["e_alpha"][k] = (
            (out.alpha - alpha_ref).norm(dim=-1) / sigmas["sigma_alpha"]
        ).numpy()
        data["e_dv"][k] = (
            ((v_hat - v1).norm(dim=-1)) / (dt * sigmas["sigma_a"])
        ).numpy()
        data["e_q1"][k] = ((q_hat - q1).norm(dim=-1) / diameter).numpy()
        data["regime"][k] = geo.regime.numpy()
        for name in (
            "pp_delta_max", "pp_delta_mean", "pw_delta", "pp_approach",
            "pp_tangential", "pw_approach", "pw_tangential", "contact_degree",
        ):
            data[name][k] = getattr(geo, name).numpy()
        data["compression"][k] = np.maximum(
            data["pp_delta_max"][k], data["pw_delta"][k]
        )
        data["speed"][k] = v.norm(dim=-1).numpy()
        data["spin"][k] = w.norm(dim=-1).numpy()
        data["a_ref_norm"][k] = a_ref.norm(dim=-1).numpy()
        data["alpha_ref_norm"][k] = alpha_ref.norm(dim=-1).numpy()

        residuals = integrator_position_residuals(q, q1, v, v1, dt)
        for scheme, value in residuals.items():
            data[f"int_{scheme}"][k] = (value.norm(dim=-1) / diameter).numpy()

        d = out.diagnostics
        f_v = d["f_cons"] - g_force
        f_r, f_h = d["f_R"], d["f_H"]
        norms = [x.norm(dim=-1) for x in (f_v, f_r, f_h)]
        denom = norms[0] + norms[1] + norms[2] + 1e-12
        for name, value in zip(("aV_norm", "aR_norm", "aH_norm"), norms):
            data[name][k] = (value / mass).numpy()
        for name, value in zip(("rhoV", "rhoR", "rhoH"), norms):
            data[name][k] = (value / denom).numpy()
        data["contact_acc_norm"][k] = (
            (f_v + f_r + f_h).norm(dim=-1) / mass
        ).numpy()

        ray_power = float((v * f_r).sum() + (w * d["tau_R"]).sum())
        hist_power = float(d["P_hist_pp"].sum() + d["P_hist_pW"].sum())
        snapshot_records.append({
            "case": case_name, "step": k,
            "rayleigh_power": ray_power,
            "rayleigh_positive": int(ray_power > 1e-6),
            "history_power": hist_power,
            "history_positive": int(hist_power > 1e-6),
        })

        active_pp = [tuple(map(int, pair)) for pair in geo.pp_contact_pairs.tolist()]
        active_pw = [int(i) for i in geo.pw_contact_ids.tolist()]
        pp_ages = update_contact_ages(pp_ages, active_pp)
        pw_ages = update_contact_ages(pw_ages, active_pw)
        # Contact features are recomputed from coordinates for only the active
        # entities, avoiding any dependence on learned softplus compression.
        for i, j in active_pp:
            rij = q[j] - q[i]
            dist = float(rij.norm())
            normal = rij / (rij.norm() + model.cfg.eps)
            arm_i = 0.5 * dist * normal
            arm_j = -0.5 * dist * normal
            uc = (v[j] + torch.linalg.cross(w[j], arm_j)) - (
                v[i] + torch.linalg.cross(w[i], arm_i)
            )
            un = float((uc * normal).sum())
            ut = float((uc - un * normal).norm())
            contact_records.append((
                case_name, "pp", pp_ages[(i, j)],
                float((data["e_a"][k, i] + data["e_a"][k, j]) / 2),
                float((data["e_alpha"][k, i] + data["e_alpha"][k, j]) / 2),
                float((data["e_dv"][k, i] + data["e_dv"][k, j]) / 2),
                max(0.0, diameter - dist), max(0.0, -un), ut,
            ))
        for i in active_pw:
            contact_records.append((
                case_name, "pw", pw_ages[i], float(data["e_a"][k, i]),
                float(data["e_alpha"][k, i]), float(data["e_dv"][k, i]),
                float(geo.pw_delta[i]), float(geo.pw_approach[i]),
                float(geo.pw_tangential[i]),
            ))

        event_records.append({
            "case": case_name, "step": k,
            "pp_contacts": len(active_pp), "pw_contacts": len(active_pw),
            "pp_max_compression": float(geo.pp_delta_max.max()),
            "pw_max_compression": float(geo.pw_delta.max()),
            "max_penetration": max_geometric_penetration(geo),
        })
        if (k + 1) % 100 == 0 or k + 1 == steps:
            elapsed = time.perf_counter() - t0
            print(
                f"[{case_name}] {k + 1}/{steps} snapshots "
                f"({elapsed:.1f}s, {elapsed / (k + 1):.3f}s/step)", flush=True
            )

    if not all(np.isfinite(v).all() for v in data.values()):
        raise FloatingPointError(f"NaN/Inf found during one-step audit of {case_name}")
    return {
        "arrays": data,
        "contacts": contact_records,
        "snapshots": snapshot_records,
        "events": event_records,
        "steps": steps,
    }


def metric_row(base: dict, arrays: dict, mask):
    row = dict(base)
    count = int(np.count_nonzero(mask))
    row["count"] = count
    for metric in METRICS:
        stats = robust_summary(arrays[metric][mask])
        for key, value in stats.items():
            if key != "count":
                row[f"{metric}_{key}"] = value
    return row


def aggregate_tables(case_results):
    one_case, one_regime, intensity, integrator, channels = [], [], [], [], []
    train_arrays = {
        key: np.concatenate([case_results[c]["arrays"][key].ravel() for c in TRAIN_CASES])
        for key in INTENSITY_FEATURES
    }
    quantiles = {
        key: np.quantile(values, [0.5, 0.9, 0.99])
        for key, values in train_arrays.items()
    }

    for case, result in case_results.items():
        a = result["arrays"]
        all_mask = np.ones(a["regime"].shape, dtype=bool)
        one_case.append(metric_row({"case": case}, a, all_mask))
        for code, regime in enumerate(REGIME_NAMES):
            mask = a["regime"] == code
            one_regime.append(metric_row(
                {"case": case, "regime": regime}, a, mask
            ))
            for scheme in ("semi_implicit", "explicit", "midpoint"):
                s = robust_summary(a[f"int_{scheme}"][mask])
                integrator.append({
                    "case": case, "regime": regime, "scheme": scheme, **s
                })
            channel_row = {"case": case, "regime": regime, "count": int(mask.sum())}
            for name in ("aV_norm", "aR_norm", "aH_norm", "rhoV", "rhoR", "rhoH",
                         "contact_acc_norm"):
                s = robust_summary(a[name][mask])
                for stat in ("mean", "median", "p90", "p99", "max_robust"):
                    channel_row[f"{name}_{stat}"] = s[stat]
            channels.append(channel_row)

        for feature in INTENSITY_FEATURES:
            bins = quantile_bin(a[feature], *quantiles[feature])
            for b, label in enumerate(("0_Q50", "Q50_Q90", "Q90_Q99", "gt_Q99")):
                intensity.append(metric_row({
                    "case": case, "feature": feature, "bin": label,
                    "train_q50": quantiles[feature][0],
                    "train_q90": quantiles[feature][1],
                    "train_q99": quantiles[feature][2],
                }, a, bins == b))

        snaps = result["snapshots"]
        ray = np.array([x["rayleigh_power"] for x in snaps])
        hist = np.array([x["history_power"] for x in snaps])
        channels.append({
            "case": case, "regime": "snapshot_passivity", "count": len(snaps),
            "rayleigh_positive_fraction": float(np.mean(ray > 1e-6)),
            "rayleigh_max_positive": float(np.maximum(ray, 0).max(initial=0)),
            "history_positive_fraction": float(np.mean(hist > 1e-6)),
            "history_positive_mean": float(np.maximum(hist, 0).mean()),
            "history_positive_p99": float(np.quantile(np.maximum(hist, 0), 0.99)),
        })
    return one_case, one_regime, intensity, integrator, channels, quantiles


def aggregate_contact_age(case_results):
    columns = ["case", "type", "age", "e_a", "e_alpha", "e_dv",
               "compression", "approach", "tangential"]
    records = [dict(zip(columns, r)) for result in case_results.values()
               for r in result["contacts"]]
    train = [r for r in records if r["case"] in TRAIN_CASES]
    qs = {}
    for typ in ("pp", "pw"):
        subset = [r for r in train if r["type"] == typ]
        for feature in ("compression", "tangential"):
            values = [r[feature] for r in subset]
            qs[(typ, feature)] = (
                np.quantile(values, [0.5, 0.9, 0.99])
                if values else np.zeros(3)
            )
    groups = defaultdict(list)
    for r in records:
        cb = int(quantile_bin(
            [r["compression"]], *qs[(r["type"], "compression")]
        )[0])
        tb = int(quantile_bin(
            [r["tangential"]], *qs[(r["type"], "tangential")]
        )[0])
        key = (r["case"], r["type"], contact_age_bucket(r["age"]), cb, tb)
        groups[key].append(r)
    labels = ("0_Q50", "Q50_Q90", "Q90_Q99", "gt_Q99")
    rows = []
    for (case, typ, age, cb, tb), values in sorted(groups.items()):
        row = {
            "case": case, "contact_type": typ, "age_group": age,
            "compression_bin": labels[cb], "tangential_bin": labels[tb],
            "count": len(values),
        }
        for metric in ("e_a", "e_alpha", "e_dv"):
            s = robust_summary([r[metric] for r in values])
            for stat in ("mean", "median", "rmse", "p90", "p95", "p99", "max_robust"):
                row[f"{metric}_{stat}"] = s[stat]
        rows.append(row)
    return rows


def aggregate_ood(case_results):
    features = [
        "speed", "spin", "pp_delta_max", "pw_delta", "pp_approach",
        "pp_tangential", "pw_approach", "pw_tangential", "contact_degree",
        "a_ref_norm", "alpha_ref_norm",
    ]
    train = {
        x: np.concatenate([case_results[c]["arrays"][x].ravel() for c in TRAIN_CASES])
        for x in features
    }
    rows = []
    comparisons = [
        ("CASE06_vs_train", train, case_results["CASE06"]["arrays"]),
        ("CASE07_vs_train", train, case_results["CASE07"]["arrays"]),
        ("CASE07_vs_CASE06", case_results["CASE06"]["arrays"],
         case_results["CASE07"]["arrays"]),
    ]
    for label, reference, other in comparisons:
        for feature in features:
            coverage = ood_coverage(reference[feature].ravel(), other[feature].ravel())
            rows.append({
                "comparison": label, "feature": feature,
                "count": coverage["other_count"], **coverage,
            })
    return rows


def _geometry_event(q, v, w, particles, wall, model, t):
    geo = snapshot_geometry(q, v, w, particles, wall, model.cfg, t)
    return {
        "pp_contacts": int(geo.pp_contact_pairs.shape[0]),
        "pw_contacts": int(geo.pw_contact_ids.shape[0]),
        "pp_max_compression": float(geo.pp_delta_max.max()),
        "pw_max_compression": float(geo.pw_delta.max()),
        "max_penetration": max_geometric_penetration(geo),
    }


@torch.no_grad()
def recursive_rollout(model, tr, particles, wall, g_vec, sigmas, start,
                      horizon, reset_k=None):
    q, v, w = tr.q[start].clone(), tr.v[start].clone(), tr.omega[start].clone()
    q0 = q.clone()
    series = defaultdict(list)
    failures = {name: None for name in (
        "position", "velocity", "additional_penetration", "quiet_baseline",
        "nan_inf", "hard_penetration",
    )}
    for s in range(horizon):
        if reset_k and s and s % reset_k == 0:
            q, v, w = (tr.q[start + s].clone(), tr.v[start + s].clone(),
                       tr.omega[start + s].clone())
        q, v, w, _ = semi_implicit_step(
            model, q, v, w, particles, wall, (start + s) * tr.dt, tr.dt, g_vec
        )
        ref_q, ref_v, ref_w = (tr.q[start + s + 1], tr.v[start + s + 1],
                               tr.omega[start + s + 1])
        finite = bool(torch.isfinite(q).all() and torch.isfinite(v).all()
                      and torch.isfinite(w).all())
        if not finite:
            failures["nan_inf"] = s + 1
            break
        q_rmse = float((q - ref_q).pow(2).sum(-1).mean().sqrt())
        v_rmse = float((v - ref_v).pow(2).sum(-1).mean().sqrt() / sigmas["sigma_v"])
        quiet = float((q0 - ref_q).pow(2).sum(-1).mean().sqrt())
        pred_event = _geometry_event(q, v, w, particles, wall, model, (start + s + 1) * tr.dt)
        real_event = _geometry_event(
            ref_q, ref_v, ref_w, particles, wall, model, (start + s + 1) * tr.dt
        )
        add_pen = pred_event["max_penetration"] - real_event["max_penetration"]
        series["q_rmse"].append(q_rmse)
        series["v_rmse_norm"].append(v_rmse)
        series["quiet_rmse"].append(quiet)
        series["additional_penetration"].append(add_pen)
        for prefix, event in (("pred", pred_event), ("real", real_event)):
            for key, value in event.items():
                series[f"{prefix}_{key}"].append(value)
        checks = {
            "position": q_rmse >= 0.25,
            "velocity": v_rmse >= 1.0,
            "additional_penetration": add_pen >= 0.10,
            "quiet_baseline": q_rmse > quiet,
            "hard_penetration": pred_event["max_penetration"] > 1.0,
        }
        for name, failed in checks.items():
            if failed and failures[name] is None:
                failures[name] = s + 1
        if checks["hard_penetration"]:
            break
    candidates = [x for k, x in failures.items() if x is not None and k != "hard_penetration"]
    failures["operational"] = min(candidates) if candidates else None
    return {"series": dict(series), "failures": failures, "steps": len(series["q_rmse"])}


def teacher_forced_row(case_arrays, tr, sigmas, start, horizon, label):
    end = min(start + horizon, case_arrays["e_q1"].shape[0])
    eq = case_arrays["e_q1"][start:end]
    ev = case_arrays["e_dv"][start:end] * tr.dt * sigmas["sigma_a"] / sigmas["sigma_v"]
    base = (tr.q[start].unsqueeze(0) - tr.q[start + 1:end + 1]).pow(2).sum(-1).sqrt().numpy()
    q_step = np.sqrt(np.mean(eq * eq, axis=1))
    v_step = np.sqrt(np.mean(ev * ev, axis=1))
    quiet_step = np.sqrt(np.mean(base * base, axis=1))
    fail_pos = np.flatnonzero(q_step >= 0.25)
    fail_v = np.flatnonzero(v_step >= 1.0)
    fail_quiet = np.flatnonzero(q_step > quiet_step)
    failures = {
        "position": int(fail_pos[0] + 1) if fail_pos.size else None,
        "velocity": int(fail_v[0] + 1) if fail_v.size else None,
        "quiet_baseline": int(fail_quiet[0] + 1) if fail_quiet.size else None,
    }
    cand = [x for x in failures.values() if x is not None]
    return {
        "case": "CASE06", "start": start, "start_regime": label,
        "mode": "teacher_forcing", "reset_k": "", "horizon": horizon,
        "actual_steps": end - start, "reached_horizon": int(end - start >= horizon),
        "count": int((end - start) * tr.q.shape[1]),
        "rmse_q": float(np.sqrt(np.mean(eq * eq))),
        "rmse_v_normalized": float(np.sqrt(np.mean(ev * ev))),
        "quiet_rmse_q": float(np.sqrt(np.mean(base * base))),
        "operational_failure": min(cand) if cand else "",
        **{f"failure_{k}": v if v is not None else "" for k, v in failures.items()},
    }


def rollout_analysis(case_results, tr6, tr7, model, particles, wall, g_vec,
                     sigmas, max_steps=None):
    max_h = min(200, tr6.q.shape[0] - 1)
    if max_steps:
        max_h = min(max_h, max_steps)
    starts = select_stratified_starts(
        case_results["CASE06"]["arrays"]["regime"],
        max_start=case_results["CASE06"]["arrays"]["regime"].shape[0] - max_h - 1,
        per_stratum=4,
    )
    if len(starts) < 20:
        available = case_results["CASE06"]["arrays"]["regime"].shape[0] - max_h
        fill = np.linspace(0, max(0, available - 1), 20).round().astype(int)
        used = {s for s, _ in starts}
        for s in fill:
            if int(s) not in used:
                starts.append((int(s), "supplemental"))
                used.add(int(s))
            if len(starts) >= 20:
                break
    starts = sorted(starts)[:20]
    print(f"CASE06 rollout starts ({len(starts)}): {starts}", flush=True)

    rows, failures = [], []
    trace6 = None
    modes = [("open", None), ("reset_4", 4), ("reset_8", 8), ("reset_16", 16)]
    for idx, (start, label) in enumerate(starts):
        for h in [x for x in HORIZONS if x <= max_h]:
            rows.append(teacher_forced_row(
                case_results["CASE06"]["arrays"], tr6, sigmas, start, h, label
            ))
        for mode, reset_k in modes:
            result = recursive_rollout(
                model, tr6, particles, wall, g_vec, sigmas, start, max_h, reset_k
            )
            if trace6 is None and mode == "open":
                trace6 = result
            failures.append({
                "case": "CASE06", "start": start, "start_regime": label,
                "mode": mode, "reset_k": reset_k or "", "steps_run": result["steps"],
                "count": 1,
                **{f"failure_{k}": v if v is not None else ""
                   for k, v in result["failures"].items()},
            })
            series = result["series"]
            for h in [x for x in HORIZONS if x <= max_h]:
                actual = min(h, result["steps"])
                if actual == 0:
                    continue
                rows.append({
                    "case": "CASE06", "start": start, "start_regime": label,
                    "mode": mode, "reset_k": reset_k or "", "horizon": h,
                    "actual_steps": actual, "reached_horizon": int(result["steps"] >= h),
                    "count": int(actual * tr6.q.shape[1]),
                    "rmse_q": float(np.sqrt(np.mean(np.square(series["q_rmse"][:actual])))),
                    "rmse_v_normalized": float(np.sqrt(np.mean(np.square(
                        series["v_rmse_norm"][:actual]
                    )))),
                    "quiet_rmse_q": float(np.sqrt(np.mean(np.square(
                        series["quiet_rmse"][:actual]
                    )))),
                    "max_additional_penetration": float(np.max(
                        series["additional_penetration"][:actual]
                    )),
                    "operational_failure": result["failures"]["operational"] or "",
                })
        print(f"[rollouts] {idx + 1}/{len(starts)} starts", flush=True)

    tf_lookup = {
        (r["start"], r["horizon"]): r["rmse_q"]
        for r in rows if r["mode"] == "teacher_forcing"
    }
    for row in rows:
        tf = tf_lookup.get((row["start"], row["horizon"]))
        row["amplification_G"] = row["rmse_q"] / (tf + 1e-12) if tf is not None else ""

    trace7 = recursive_rollout(
        model, tr7, particles, wall, g_vec, sigmas, 0, max_h, None
    )
    failures.append({
        "case": "CASE07", "start": 0, "start_regime": "initial",
        "mode": "open", "reset_k": "", "steps_run": trace7["steps"],
        "count": 1,
        **{f"failure_{k}": v if v is not None else ""
           for k, v in trace7["failures"].items()},
    })
    return rows, failures, trace6, trace7, starts


def make_plots(out_dir, case_results, one_regime, intensity, channels,
               contact_age, ood_rows, rollout_rows, failure_rows, trace6, trace7):
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    # One-step error by regime.
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(REGIME_NAMES)); width = 0.25
    for j, case in enumerate(("CASE05", "CASE06", "CASE07")):
        vals = [next(r for r in one_regime if r["case"] == case and
                     r["regime"] == reg)["e_a_rmse"] for reg in REGIME_NAMES]
        ax.bar(x + (j - 1) * width, vals, width, label=case)
    ax.set_xticks(x, REGIME_NAMES); ax.set_yscale("log")
    ax.set_ylabel("RMSE normalizado de aceleración"); ax.legend()
    ax.set_title("Error teacher-forced a un paso por régimen")
    fig.tight_layout(); fig.savefig(plots / "one_step_regimes.png", dpi=170); plt.close(fig)

    # Contact intensity.
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = ("0_Q50", "Q50_Q90", "Q90_Q99", "gt_Q99")
    for case in ("CASE05", "CASE06", "CASE07"):
        subset = [r for r in intensity if r["case"] == case and
                  r["feature"] == "compression"]
        ax.plot(labels, [next(r for r in subset if r["bin"] == b)["e_a_rmse"]
                         for b in labels], marker="o", label=case)
    ax.set_yscale("log"); ax.set_ylabel("RMSE normalizado de aceleración")
    ax.set_xlabel("Cuantil de compresión (cortes de train)"); ax.legend()
    ax.set_title("Error a un paso frente a intensidad de contacto")
    fig.tight_layout(); fig.savefig(plots / "one_step_vs_contact_intensity.png", dpi=170); plt.close(fig)

    # Channel norms.
    fig, ax = plt.subplots(figsize=(10, 5))
    rows = [r for r in channels if r["case"] == "CASE06" and
            r["regime"] in REGIME_NAMES]
    for j, key in enumerate(("aV_norm_median", "aR_norm_median", "aH_norm_median")):
        ax.bar(x + (j - 1) * width, [next(r for r in rows if r["regime"] == reg)[key]
                                    for reg in REGIME_NAMES], width,
               label=key.split("_")[0])
    ax.set_xticks(x, REGIME_NAMES); ax.set_yscale("log")
    ax.set_ylabel("Norma mediana de aceleración"); ax.legend()
    ax.set_title("Descomposición V/R/H en CASE06")
    fig.tight_layout(); fig.savefig(plots / "channel_norms_by_regime.png", dpi=170); plt.close(fig)

    # Contact age.
    fig, ax = plt.subplots(figsize=(9, 5))
    ages = ("start", "short", "sustained", "long")
    for typ in ("pp", "pw"):
        vals = []
        for age in ages:
            subset = [r for r in contact_age if r["case"] == "CASE06" and
                      r["contact_type"] == typ and r["age_group"] == age]
            values = np.repeat([r["e_a_rmse"] for r in subset],
                               [r["count"] for r in subset]) if subset else []
            vals.append(float(np.sqrt(np.mean(np.square(values)))) if len(values) else np.nan)
        ax.plot(ages, vals, marker="o", label=typ)
    ax.set_yscale("log"); ax.set_ylabel("RMSE normalizado de aceleración")
    ax.set_title("Error según edad del contacto - CASE06"); ax.legend()
    fig.tight_layout(); fig.savefig(plots / "contact_age_error.png", dpi=170); plt.close(fig)

    # Rollout comparison at H.
    fig, ax = plt.subplots(figsize=(10, 5))
    for mode in ("teacher_forcing", "open", "reset_4", "reset_8", "reset_16"):
        vals = []
        hs = sorted({r["horizon"] for r in rollout_rows})
        for h in hs:
            subset = [r["rmse_q"] for r in rollout_rows if r["mode"] == mode and
                      r["horizon"] == h]
            vals.append(np.median(subset) if subset else np.nan)
        ax.plot(hs, vals, marker="o", label=mode)
    ax.set_yscale("log"); ax.set_xscale("log", base=2)
    ax.set_xlabel("Horizonte"); ax.set_ylabel("Mediana RMSE q")
    ax.set_title("Rollout abierto vs teacher forcing vs resets - CASE06"); ax.legend()
    fig.tight_layout(); fig.savefig(plots / "rollout_open_vs_tf.png", dpi=170); plt.close(fig)

    # Failure time distribution.
    fig, ax = plt.subplots(figsize=(9, 5))
    modes = sorted({r["mode"] for r in failure_rows if r["case"] == "CASE06"})
    values = [[float(r["failure_operational"]) for r in failure_rows
               if r["case"] == "CASE06" and r["mode"] == mode and
               r["failure_operational"] != ""] for mode in modes]
    ax.boxplot(values, tick_labels=modes, showfliers=True)
    ax.set_ylabel("Primer paso de fallo"); ax.tick_params(axis="x", rotation=25)
    ax.set_title("Distribución de tiempos de fallo - CASE06")
    fig.tight_layout(); fig.savefig(plots / "failure_time_distribution.png", dpi=170); plt.close(fig)

    # Contact-event traces.
    for case, trace in (("case06", trace6), ("case07", trace7)):
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        s = trace["series"]
        t = np.arange(1, trace["steps"] + 1)
        for key, label in (("pp_contacts", "pp"), ("pw_contacts", "pW")):
            axes[0].plot(t, s[f"real_{key}"], label=f"real {label}")
            axes[0].plot(t, s[f"pred_{key}"], "--", label=f"pred {label}")
        axes[0].set_ylabel("Número de contactos"); axes[0].legend(ncol=2)
        axes[1].plot(t, s["real_max_penetration"], label="real")
        axes[1].plot(t, s["pred_max_penetration"], "--", label="pred")
        axes[1].axhline(1.0, color="red", linestyle=":", label="parada 1 diámetro")
        axes[1].set_ylabel("Penetración máxima"); axes[1].set_xlabel("Paso")
        axes[1].legend(); fig.suptitle(f"Eventos de contacto - {case.upper()}")
        fig.tight_layout(); fig.savefig(plots / f"contact_events_{case}.png", dpi=170); plt.close(fig)

    # OOD distributions.
    features = ["speed", "spin", "compression", "pp_approach", "pp_tangential",
                "pw_approach", "pw_tangential", "a_ref_norm", "alpha_ref_norm"]
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for ax, feature in zip(axes.ravel(), features):
        for scope, cases in (("train", TRAIN_CASES), ("CASE06", ["CASE06"]),
                             ("CASE07", ["CASE07"])):
            vals = np.concatenate([case_results[c]["arrays"][feature].ravel()
                                   for c in cases])
            hi = np.quantile(vals, 0.995)
            ax.hist(vals[vals <= hi], bins=50, density=True, histtype="step", label=scope)
        ax.set_title(feature); ax.set_yscale("log")
    axes[0, 0].legend(); fig.suptitle("Distribuciones de features y soporte OOD")
    fig.tight_layout(); fig.savefig(plots / "ood_feature_distributions.png", dpi=170); plt.close(fig)


def copy_legacy_metrics(out_dir):
    for case in ("case06", "case07"):
        src = out_dir / "reproduction" / f"legacy_{case}" / "metrics.json"
        dst = out_dir / "reproduction" / f"legacy_metrics_{case}.json"
        if not src.exists():
            raise FileNotFoundError(
                f"Missing legacy reproduction {src}; run scripts/slgnn_v2/evaluate.py first"
            )
        shutil.copyfile(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", default="A-VRH-strat-H200-seed0-20260729")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="development-only cap; omit for the complete audit")
    ap.add_argument("--skip-rollouts", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    out_dir = (Path(args.out) if args.out else
               ROOT / "results" / "slgnn_v2" / "diagnostics" / "phase_a" / args.run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "tables" / "one_step_by_case.csv").exists():
        raise FileExistsError(f"Refusing to overwrite existing Phase A result: {out_dir}")

    model, ck = load_checkpoint(Path(args.checkpoint))
    cfg_yaml = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg, metadata = preflight(args, out_dir, ck, cfg_yaml)
    copy_legacy_metrics(out_dir)
    _, train, tr6, wall, g_vec, particles = load_split(cfg, ROOT)
    trajectories = {case: tr for case, tr in zip(TRAIN_CASES, train)}
    trajectories["CASE06"] = tr6
    trajectories["CASE07"] = load_case_by_name(cfg, ROOT, "CASE07")
    sigmas = ck["sigmas"]

    case_results = {}
    for case in CASES:
        case_results[case] = analyze_case(
            case, trajectories[case], model, particles, wall, g_vec, sigmas,
            max_steps=args.max_steps,
        )

    one_case, one_regime, intensity, integrator, channels, quantiles = (
        aggregate_tables(case_results)
    )
    contact_age = aggregate_contact_age(case_results)
    ood_rows = aggregate_ood(case_results)
    tables = out_dir / "tables"
    write_csv(tables / "one_step_by_case.csv", one_case)
    write_csv(tables / "one_step_by_regime.csv", one_regime)
    write_csv(tables / "one_step_by_intensity.csv", intensity)
    write_csv(tables / "integrator_residuals.csv", integrator)
    write_csv(tables / "channel_decomposition.csv", channels)
    write_csv(tables / "contact_age.csv", contact_age)
    write_csv(tables / "ood_coverage.csv", ood_rows)

    if args.skip_rollouts:
        rollout_rows, failure_rows, trace6, trace7, starts = [], [], None, None, []
    else:
        rollout_rows, failure_rows, trace6, trace7, starts = rollout_analysis(
            case_results, trajectories["CASE06"], trajectories["CASE07"], model,
            particles, wall, g_vec, sigmas, max_steps=args.max_steps,
        )
        write_csv(tables / "rollout_windows.csv", rollout_rows)
        write_csv(tables / "failure_times.csv", failure_rows)
        make_plots(
            out_dir, case_results, one_regime, intensity, channels, contact_age,
            ood_rows, rollout_rows, failure_rows, trace6, trace7,
        )

    metadata.update({
        "completed_one_step_cases": CASES,
        "snapshots_per_case": {c: case_results[c]["steps"] for c in CASES},
        "rollout_starts_case06": starts,
        "intensity_train_quantiles": quantiles,
        "nan_inf_detected": False,
    })
    write_json(out_dir / "metadata.json", metadata)
    readme = f"""# Phase A - SLGNN checkpoint audit

Run ID: `{args.run_id}`

This directory is generated by a read-only audit of `{args.checkpoint}`. Model
weights, the effective architecture, integrator, and data were not changed.

## Reproduce

```powershell
& .venv\\Scripts\\python.exe scripts\\evaluate.py --checkpoint {args.checkpoint} --config {args.config} --case CASE06 --horizon 100 --out {out_dir / 'reproduction' / 'legacy_case06'}
& .venv\\Scripts\\python.exe scripts\\evaluate.py --checkpoint {args.checkpoint} --config {args.config} --case CASE07 --horizon 100 --out {out_dir / 'reproduction' / 'legacy_case07'}
& .venv\\Scripts\\python.exe scripts\\diagnose_phase_a.py --checkpoint {args.checkpoint} --config {args.config} --run-id {args.run_id}
& .venv\\Scripts\\python.exe scripts\\profile_phase_a.py --checkpoint {args.checkpoint} --config {args.config} --out {out_dir / 'profiles'}
& .venv\\Scripts\\python.exe -m pytest -q
```

See `FASE_A_RESULTADOS.md` for interpretation and the CSV files in `tables/`
for the underlying counts and quantitative evidence.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Phase A diagnostic tables and plots written to {out_dir}")


if __name__ == "__main__":
    main()
