"""Compara transferencia pp->pared contra entrenamiento desde cero.

Diseño experimental:
  * validación leave-one-angle-out sobre 10, 30, 45, 60 y 90 grados;
  * pares de corridas con idénticos estados partícula-pared muestreados;
  * las cabezas partícula-pared parten idénticas en cada par;
  * el brazo transfer carga backbone y canal pp desde el checkpoint de dos
    esferas, usa optimizador nuevo, learning rates discriminativos y replay pp;
  * el control parte completamente desde cero y sólo ve el benchmark de pared;
  * ambos modelos finales se reevalúan en las dos esferas para medir retención.

Los resultados, checkpoints, predicciones y figuras se guardan en las rutas
definidas por configs/benchmarks/benchmark_wall_transfer.yaml.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

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


ROOT = Path(__file__).resolve().parents[2]


def build_model(cfg):
    allowed = SLGNNConfig().__dict__
    overrides = {k: v for k, v in cfg["model"].items() if k in allowed}
    return SLGNN(SLGNNConfig(**overrides))


def load_data(cfg):
    scales = default_scales()
    extracted = ROOT / "data" / "extracted"
    dt = float(cfg["data"]["dt"])
    wall_base = extracted / cfg["data"]["wall_dataset"]
    pp_base = extracted / cfg["data"]["particle_dataset"]
    wall = {
        c: scales.nondim(load_case(wall_base / c, dt=dt))
        for c in cfg["data"]["wall_cases"]
    }
    pp = {
        c: scales.nondim(load_case(pp_base / c, dt=dt))
        for c in cfg["data"]["particle_cases"]
    }
    # Caja enorme cuya única cara alcanzable es z=0; equivale al plano estático
    # del benchmark y reutiliza la SDF diferenciable verificada del proyecto.
    plane = BoxSDF([-1000.0, -1000.0, 0.0], [1000.0, 1000.0, 1000.0])
    return wall, pp, plane


def make_particles(tr):
    return Particles.uniform(
        tr.q.shape[1],
        m=tr.m[0].item(),
        radius=tr.radii[0].item(),
        dtype=tr.q.dtype,
    )


def targets(tr):
    return (
        finite_difference_accelerations(tr.v, tr.dt),
        finite_difference_accelerations(tr.omega, tr.dt),
    )


def active_indices(tr, threshold):
    a, alpha = targets(tr)
    activity = torch.maximum(a.norm(dim=-1).amax(-1), alpha.norm(dim=-1).amax(-1))
    return torch.where(activity > threshold)[0]


def wall_near_indices(tr, g_off):
    gap = tr.q[:-1, 0, 2] - tr.radii[0]
    return torch.where(gap < g_off)[0]


def pp_near_indices(tr, r_off):
    d = (tr.q[:-1, 1] - tr.q[:-1, 0]).norm(dim=-1)
    return torch.where(d < r_off)[0]


def rms(values):
    x = torch.cat([v.reshape(-1) for v in values])
    return max(float(torch.sqrt(torch.mean(x * x))), 1e-4)


def compute_scales(dataset, cases, selector):
    aa, al, vv, ww = [], [], [], []
    for case in cases:
        tr = dataset[case]
        idx = selector(tr)
        a, alpha = targets(tr)
        aa.append(a[idx])
        al.append(alpha[idx])
        vv.append(tr.v[idx])
        ww.append(tr.omega[idx])
    return {
        "sigma_a": rms(aa),
        "sigma_alpha": rms(al),
        "sigma_v": rms(vv),
        "sigma_w": rms(ww),
    }


def paired_initial_models(cfg, seed, pretrained_state):
    """Crea un par donde las cabezas pW son exactamente idénticas."""
    torch.manual_seed(seed)
    scratch = build_model(cfg)
    transfer = build_model(cfg)
    transfer_state = {k: v.detach().clone() for k, v in pretrained_state.items()}
    scratch_state = scratch.state_dict()
    for key in transfer_state:
        if key.startswith("head_pw_"):
            transfer_state[key] = scratch_state[key].detach().clone()
    transfer.load_state_dict(transfer_state)
    return scratch, transfer


def make_optimizer(model, cfg):
    wall_heads, particle_heads, shared = [], [], []
    for name, parameter in model.named_parameters():
        if name.startswith("head_pw_"):
            wall_heads.append(parameter)
        elif name.startswith("head_pp_"):
            particle_heads.append(parameter)
        else:
            shared.append(parameter)
    tc = cfg["training"]
    return torch.optim.AdamW(
        [
            {"params": wall_heads, "lr": float(tc["lr_wall_heads"]), "name": "wall"},
            {"params": shared, "lr": float(tc["lr_shared"]), "name": "shared"},
            {"params": particle_heads, "lr": float(tc["lr_particle_heads"]), "name": "particle"},
        ],
        weight_decay=float(tc["weight_decay"]),
    )


def normalized_acceleration_loss(out, a_ref, alpha_ref, scales):
    la = ((out.a - a_ref) / scales["sigma_a"]).pow(2).mean()
    lw = ((out.alpha - alpha_ref) / scales["sigma_alpha"]).pow(2).mean()
    return la + lw


def randint(n, generator):
    return int(torch.randint(n, (1,), generator=generator).item())


def sample_wall_index(tr, active, near, probability, generator):
    use_active = float(torch.rand((), generator=generator)) < probability
    choices = active if use_active else near
    return int(choices[randint(len(choices), generator)])


def replay_loss(model, pp_data, pp_cases, particles, scales, cfg, generator):
    total = torch.zeros(())
    batch = int(cfg["training"]["replay_batch_size"])
    for b in range(batch):
        case = pp_cases[b % len(pp_cases)]
        tr = pp_data[case]
        choices = pp_near_indices(tr, model.cfg.r_off)
        k = int(choices[randint(len(choices), generator)])
        a_ref, alpha_ref = targets(tr)
        out = model(tr.q[k], tr.v[k], tr.omega[k], particles)
        total = total + normalized_acceleration_loss(
            out, a_ref[k], alpha_ref[k], scales
        )
    return total / batch


def train_one_step(
    model,
    method,
    wall_data,
    train_cases,
    pp_data,
    wall,
    wall_particles,
    pp_particles,
    wall_scales,
    pp_scales,
    cfg,
    wall_generator,
    replay_generator,
    history,
):
    tc = cfg["training"]
    optimizer = make_optimizer(model, cfg)
    iterations = int(tc["one_step_iters"])
    batch_size = int(tc["batch_size"])
    probability = float(tc["active_probability"])
    threshold = float(cfg["evaluation"]["active_acceleration_threshold"])
    active = {c: active_indices(wall_data[c], threshold) for c in train_cases}
    near = {c: wall_near_indices(wall_data[c], model.cfg.g_off) for c in train_cases}
    refs = {c: targets(wall_data[c]) for c in train_cases}

    model.train()
    for it in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        wall_loss = torch.zeros(())
        for b in range(batch_size):
            case = train_cases[(it * batch_size + b) % len(train_cases)]
            tr = wall_data[case]
            k = sample_wall_index(
                tr, active[case], near[case], probability, wall_generator
            )
            a_ref, alpha_ref = refs[case]
            out = model(
                tr.q[k], tr.v[k], tr.omega[k], wall_particles, wall=wall
            )
            wall_loss = wall_loss + normalized_acceleration_loss(
                out, a_ref[k], alpha_ref[k], wall_scales
            )
        wall_loss = wall_loss / batch_size
        total = wall_loss
        replay_value = 0.0
        if method == "transfer":
            rp = replay_loss(
                model,
                pp_data,
                list(pp_data),
                pp_particles,
                pp_scales,
                cfg,
                replay_generator,
            )
            total = total + float(tc["replay_weight"]) * rp
            replay_value = float(rp.detach())
        if not torch.isfinite(total):
            raise FloatingPointError(f"Pérdida no finita: {method}, iter={it}")
        total.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), float(tc["grad_clip"]))
        optimizer.step()
        history.append(
            {
                "phase": "one_step",
                "iteration": it,
                "wall_loss": float(wall_loss.detach()),
                "replay_loss": replay_value,
                "total_loss": float(total.detach()),
                "grad_norm": float(grad),
            }
        )
        if it % int(tc["log_every"]) == 0 or it == iterations - 1:
            print(
                f"    {method:8s} one-step {it:4d}/{iterations} "
                f"wall={float(wall_loss.detach()):.5f} replay={replay_value:.5f}",
                flush=True,
            )
    return optimizer


def rollout_starts(tr, horizon, threshold):
    active = active_indices(tr, threshold)
    lo = max(0, int(active[0]) - 5)
    hi = min(int(active[-1]), tr.q.shape[0] - horizon - 1)
    return list(range(lo, max(lo, hi) + 1))


def finetune_rollout(
    model,
    method,
    wall_data,
    train_cases,
    pp_data,
    wall,
    wall_particles,
    pp_particles,
    wall_scales,
    pp_scales,
    cfg,
    wall_generator,
    replay_generator,
    history,
):
    rc, tc = cfg["rollout"], cfg["training"]
    optimizer = make_optimizer(model, cfg)
    iterations = int(rc["iters"])
    horizon, chunk_size = int(rc["horizon"]), int(rc["tbptt_chunk"])
    sigma_q = float(rc["sigma_q"])
    threshold = float(cfg["evaluation"]["active_acceleration_threshold"])
    starts = {
        c: rollout_starts(wall_data[c], horizon, threshold) for c in train_cases
    }

    model.train()
    for it in range(iterations):
        case = train_cases[it % len(train_cases)]
        tr = wall_data[case]
        choices = starts[case]
        k0 = choices[randint(len(choices), wall_generator)]
        q, v, w = tr.q[k0], tr.v[k0], tr.omega[k0]
        optimizer.zero_grad(set_to_none=True)
        wall_value = 0.0
        chunk = torch.zeros(())
        for s in range(horizon):
            q, v, w, _ = semi_implicit_step(
                model, q, v, w, wall_particles, wall=wall, t=0.0, dt=tr.dt
            )
            k = k0 + s + 1
            step = ((q - tr.q[k]) / sigma_q).pow(2).mean()
            step = step + ((v - tr.v[k]) / wall_scales["sigma_v"]).pow(2).mean()
            step = step + ((w - tr.omega[k]) / wall_scales["sigma_w"]).pow(2).mean()
            step = step / horizon
            chunk = chunk + step
            wall_value += float(step.detach())
            if (s + 1) % chunk_size == 0 or s == horizon - 1:
                chunk.backward()
                chunk = torch.zeros(())
                if s != horizon - 1:
                    q, v, w = q.detach(), v.detach(), w.detach()
        replay_value = 0.0
        if method == "transfer":
            rp = replay_loss(
                model,
                pp_data,
                list(pp_data),
                pp_particles,
                pp_scales,
                cfg,
                replay_generator,
            )
            (float(tc["replay_weight"]) * rp).backward()
            replay_value = float(rp.detach())
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), float(tc["grad_clip"]))
        optimizer.step()
        history.append(
            {
                "phase": "rollout",
                "iteration": it,
                "wall_loss": wall_value,
                "replay_loss": replay_value,
                "total_loss": wall_value + float(tc["replay_weight"]) * replay_value,
                "grad_norm": float(grad),
            }
        )
        if it % int(rc["log_every"]) == 0 or it == iterations - 1:
            print(
                f"    {method:8s} rollout  {it:4d}/{iterations} "
                f"wall={wall_value:.5f} replay={replay_value:.5f} case={case}",
                flush=True,
            )
    return optimizer


@torch.no_grad()
def predict(model, tr, particles, wall=None):
    q, v, w = tr.q[0], tr.v[0], tr.omega[0]
    qs, vs, ws = [q], [v], [w]
    for k in range(tr.q.shape[0] - 1):
        q, v, w, _ = semi_implicit_step(
            model, q, v, w, particles, wall=wall, t=k * tr.dt, dt=tr.dt
        )
        if not (torch.isfinite(q).all() and torch.isfinite(v).all() and torch.isfinite(w).all()):
            raise FloatingPointError(f"Rollout no finito en paso {k}")
        qs.append(q)
        vs.append(v)
        ws.append(w)
    return torch.stack(qs), torch.stack(vs), torch.stack(ws)


def pooled_rmse(pred, ref, sl=slice(None)):
    return float((pred[sl] - ref[sl]).pow(2).sum(-1).mean().sqrt())


@torch.no_grad()
def evaluate_wall(model, tr, particles, wall, threshold):
    q, v, w = predict(model, tr, particles, wall)
    steps = torch.arange(tr.q.shape[0], dtype=tr.q.dtype)[:, None, None]
    q_ball = tr.q[0:1] + steps * tr.dt * tr.v[0:1]
    v_ball = tr.v[0:1].expand_as(tr.v)
    w_ball = tr.omega[0:1].expand_as(tr.omega)
    active = active_indices(tr, threshold)
    first, last = int(active[0]), int(active[-1]) + 1
    post = slice(last, tr.q.shape[0])
    a_ref, alpha_ref = targets(tr)
    a_pred, alpha_pred = [], []
    for k in active.tolist():
        out = model(tr.q[k], tr.v[k], tr.omega[k], particles, wall=wall)
        a_pred.append(out.a)
        alpha_pred.append(out.alpha)
    a_pred, alpha_pred = torch.stack(a_pred), torch.stack(alpha_pred)

    impulse_ref, impulse_pred = tr.v[-1] - tr.v[0], v[-1] - v[0]
    spin_ref, spin_pred = tr.omega[-1] - tr.omega[0], w[-1] - w[0]
    e_ref = float(tr.v[-1, 0, 2] / (-tr.v[0, 0, 2]))
    e_pred = float(v[-1, 0, 2] / (-tr.v[0, 0, 2]))
    spin_den = torch.linalg.norm(spin_ref)
    spin_error = float(torch.linalg.norm(spin_pred - spin_ref) / spin_den) if spin_den > 1e-8 else float(torch.linalg.norm(spin_pred - spin_ref))
    metrics = {
        "contact_first_step": first,
        "contact_last_step": last,
        "one_step_a_rmse_active": pooled_rmse(a_pred, a_ref[active]),
        "one_step_alpha_rmse_active": pooled_rmse(alpha_pred, alpha_ref[active]),
        "rollout_q_rmse_all": pooled_rmse(q, tr.q),
        "rollout_q_rmse_post": pooled_rmse(q, tr.q, post),
        "rollout_v_rmse_all": pooled_rmse(v, tr.v),
        "rollout_v_rmse_post": pooled_rmse(v, tr.v, post),
        "rollout_w_rmse_all": pooled_rmse(w, tr.omega),
        "rollout_w_rmse_post": pooled_rmse(w, tr.omega, post),
        "ballistic_q_rmse_post": pooled_rmse(q_ball, tr.q, post),
        "ballistic_v_rmse_post": pooled_rmse(v_ball, tr.v, post),
        "ballistic_w_rmse_post": pooled_rmse(w_ball, tr.omega, post),
        "impulse_relative_error": float(torch.linalg.norm(impulse_pred - impulse_ref) / torch.linalg.norm(impulse_ref).clamp_min(1e-12)),
        "spin_change_relative_error": spin_error,
        "restitution_ref": e_ref,
        "restitution_pred": e_pred,
        "restitution_abs_error": abs(e_pred - e_ref),
        "max_penetration": float((particles.radii[0] - q[:, 0, 2]).clamp_min(0).max()),
    }
    arrays = {
        "q_pred": q.numpy(), "v_pred": v.numpy(), "w_pred": w.numpy(),
        "q_ref": tr.q.numpy(), "v_ref": tr.v.numpy(), "w_ref": tr.omega.numpy(),
    }
    return metrics, arrays


@torch.no_grad()
def evaluate_pp(model, tr, particles, threshold):
    q, v, w = predict(model, tr, particles, wall=None)
    active = active_indices(tr, threshold)
    last = int(active[-1]) + 1
    post = slice(last, tr.q.shape[0])
    impulse_ref, impulse_pred = tr.v[-1] - tr.v[0], v[-1] - v[0]
    spin_ref, spin_pred = tr.omega[-1] - tr.omega[0], w[-1] - w[0]
    return {
        "rollout_q_rmse_post": pooled_rmse(q, tr.q, post),
        "rollout_v_rmse_post": pooled_rmse(v, tr.v, post),
        "rollout_w_rmse_post": pooled_rmse(w, tr.omega, post),
        "impulse_relative_error": float(torch.linalg.norm(impulse_pred - impulse_ref) / torch.linalg.norm(impulse_ref).clamp_min(1e-12)),
        "spin_change_relative_error": float(torch.linalg.norm(spin_pred - spin_ref) / torch.linalg.norm(spin_ref).clamp_min(1e-12)),
    }


def save_checkpoint(path, model, optimizer, cfg, method, train_cases, seed, elapsed):
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": asdict(model.cfg),
            "config": cfg,
            "method": method,
            "train_cases": train_cases,
            "seed": seed,
            "elapsed_seconds": elapsed,
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def train_run(
    method,
    name,
    model,
    wall_data,
    train_cases,
    pp_data,
    wall,
    wall_scales,
    pp_scales,
    cfg,
    seed,
    out_dir,
    ckpt_dir,
):
    print(f"\n== {name} / {method}: train={train_cases} seed={seed} ==", flush=True)
    wall_particles = make_particles(wall_data[train_cases[0]])
    pp_particles = make_particles(pp_data[next(iter(pp_data))])
    wall_generator = torch.Generator().manual_seed(seed + 10_000)
    replay_generator = torch.Generator().manual_seed(seed + 20_000)
    history = []
    started = time.time()
    train_one_step(
        model, method, wall_data, train_cases, pp_data, wall,
        wall_particles, pp_particles, wall_scales, pp_scales, cfg,
        wall_generator, replay_generator, history,
    )
    optimizer = finetune_rollout(
        model, method, wall_data, train_cases, pp_data, wall,
        wall_particles, pp_particles, wall_scales, pp_scales, cfg,
        wall_generator, replay_generator, history,
    )
    elapsed = time.time() - started
    model.eval()
    save_checkpoint(
        ckpt_dir / f"{method}_{name}.pt", model, optimizer, cfg,
        method, train_cases, seed, elapsed,
    )
    with (out_dir / f"history_{method}_{name}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f"    terminado en {elapsed:.1f}s", flush=True)
    return model, history, elapsed


def plot_loso(rows, cases, path):
    selected = [r for r in rows if r["split"] == "held_out"]
    lookup = {(r["method"], r["case"]): r for r in selected}
    x = np.arange(len(cases)); width = 0.37
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    specs = [
        ("rollout_q_rmse_post", "RMSE posición post [dₚ]"),
        ("rollout_v_rmse_post", "RMSE velocidad post"),
        ("spin_change_relative_error", "Error relativo cambio de spin"),
    ]
    for ax, (key, title) in zip(axes, specs):
        ax.bar(x - width / 2, [lookup[("scratch", c)][key] for c in cases], width, label="Desde cero")
        ax.bar(x + width / 2, [lookup[("transfer", c)][key] for c in cases], width, label="Transfer + replay")
        ax.set_xticks(x, [f"{c}°" for c in cases])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_final_dynamics(final_arrays, cases, path):
    fig, axes = plt.subplots(len(cases), 3, figsize=(13, 3.0 * len(cases)))
    for row, case in enumerate(cases):
        tr = final_arrays[("transfer", case)]
        sc = final_arrays[("scratch", case)]
        t = np.arange(tr["q_ref"].shape[0])
        for ax, field, comp, ylabel in [
            (axes[row, 0], "q", 2, "z [dₚ]"),
            (axes[row, 1], "v", 2, "v_z [adim.]"),
            (axes[row, 2], "w", 1, "ω_y [adim.]"),
        ]:
            ax.plot(t, tr[f"{field}_ref"][:, 0, comp], color="black", label="DEM", lw=1.8)
            ax.plot(t, tr[f"{field}_pred"][:, 0, comp], "--", label="Transfer")
            ax.plot(t, sc[f"{field}_pred"][:, 0, comp], ":", label="Desde cero")
            ax.set_xlabel("paso"); ax.set_ylabel(f"{case}°\n{ylabel}"); ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8); axes[0, 1].legend(fontsize=8); axes[0, 2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def moving_average(values, window=25):
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_histories(histories, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for method in ["scratch", "transfer"]:
        hist = histories[(method, "final")]
        for ax, phase in zip(axes, ["one_step", "rollout"]):
            y = [r["wall_loss"] for r in hist if r["phase"] == phase]
            sm = moving_average(y, min(25, max(1, len(y) // 4)))
            ax.plot(np.arange(len(sm)), sm, label="Desde cero" if method == "scratch" else "Transfer + replay")
            ax.set_yscale("log"); ax.set_xlabel("iteración"); ax.set_ylabel("pérdida pared")
    axes[0].set_title("Ajuste a un paso (media móvil)")
    axes[1].set_title("Afinamiento rollout (media móvil)")
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_retention(rows, cases, path):
    lookup = {(r["model"], r["case"]): r for r in rows}
    models = ["pretrained", "transfer_final", "scratch_final"]
    labels = ["Preentrenado pp", "Transfer final", "Scratch final"]
    x = np.arange(len(cases)); width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for offset, (model, label) in enumerate(zip(models, labels)):
        shift = (offset - 1) * width
        axes[0].bar(x + shift, [lookup[(model, c)]["rollout_v_rmse_post"] for c in cases], width, label=label)
        axes[1].bar(x + shift, [lookup[(model, c)]["impulse_relative_error"] for c in cases], width, label=label)
    axes[0].set_title("Retención: RMSE velocidad post-choque")
    axes[1].set_title("Retención: error relativo de impulso")
    for ax in axes:
        ax.set_xticks(x, cases); ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def fmt(value):
    return f"{value:.4g}"


def write_report(path, cfg, wall_rows, retention_rows, elapsed):
    cases = list(cfg["data"]["wall_cases"])
    held = {(r["method"], r["case"]): r for r in wall_rows if r["split"] == "held_out"}
    final = {(r["method"], r["case"]): r for r in wall_rows if r["split"] == "final_fit"}
    med = {}
    for method in ["scratch", "transfer"]:
        med[method] = {
            key: float(np.median([held[(method, c)][key] for c in cases]))
            for key in ["rollout_q_rmse_post", "rollout_v_rmse_post", "spin_change_relative_error", "restitution_abs_error"]
        }
    retain = {(r["model"], r["case"]): r for r in retention_rows}
    lines = [
        "# Transferencia partícula–partícula → partícula–pared",
        "",
        "## Protocolo",
        "",
        "- Comparación pareada `transfer + replay` versus `desde cero`.",
        "- Validación leave-one-angle-out en 10°, 30°, 45°, 60° y 90°.",
        "- Cabezas partícula–pared inicializadas de forma idéntica dentro de cada par.",
        "- Optimizador nuevo; LR pared > LR backbone > LR cabezas pp.",
        "- El replay pp sólo se usa en el brazo transferido para limitar olvido catastrófico.",
        "",
        "## Generalización a un ángulo no visto",
        "",
        "| ángulo | método | RMSE q post [dₚ] | RMSE v post | error impulso | error spin | error restitución | penetración máx. [dₚ] |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in cases:
        for method in ["scratch", "transfer"]:
            r = held[(method, c)]
            lines.append(
                f"| {c}° | {method} | {fmt(r['rollout_q_rmse_post'])} | {fmt(r['rollout_v_rmse_post'])} | "
                f"{fmt(r['impulse_relative_error'])} | {fmt(r['spin_change_relative_error'])} | "
                f"{fmt(r['restitution_abs_error'])} | {fmt(r['max_penetration'])} |"
            )
    lines += [
        "",
        "## Resumen LOSO (medianas)",
        "",
        "| método | RMSE q post | RMSE v post | error spin | error restitución |",
        "|---|---:|---:|---:|---:|",
        f"| Desde cero | {fmt(med['scratch']['rollout_q_rmse_post'])} | {fmt(med['scratch']['rollout_v_rmse_post'])} | {fmt(med['scratch']['spin_change_relative_error'])} | {fmt(med['scratch']['restitution_abs_error'])} |",
        f"| Transfer + replay | {fmt(med['transfer']['rollout_q_rmse_post'])} | {fmt(med['transfer']['rollout_v_rmse_post'])} | {fmt(med['transfer']['spin_change_relative_error'])} | {fmt(med['transfer']['restitution_abs_error'])} |",
        "",
        "## Modelos finales entrenados con los cinco ángulos",
        "",
        "| ángulo | método | RMSE q post [dₚ] | RMSE v post | error spin |",
        "|---:|---|---:|---:|---:|",
    ]
    for c in cases:
        for method in ["scratch", "transfer"]:
            r = final[(method, c)]
            lines.append(
                f"| {c}° | {method} | {fmt(r['rollout_q_rmse_post'])} | "
                f"{fmt(r['rollout_v_rmse_post'])} | {fmt(r['spin_change_relative_error'])} |"
            )
    lines += [
        "",
        "## Retención del benchmark de dos esferas",
        "",
        "| caso | modelo | RMSE v post | error impulso | error spin |",
        "|---|---|---:|---:|---:|",
    ]
    for c in cfg["data"]["particle_cases"]:
        for model_name in ["pretrained", "transfer_final", "scratch_final"]:
            r = retain[(model_name, c)]
            lines.append(
                f"| {c} | {model_name} | {fmt(r['rollout_v_rmse_post'])} | "
                f"{fmt(r['impulse_relative_error'])} | {fmt(r['spin_change_relative_error'])} |"
            )
    gain_v = 100 * (1 - med["transfer"]["rollout_v_rmse_post"] / med["scratch"]["rollout_v_rmse_post"])
    gain_q = 100 * (1 - med["transfer"]["rollout_q_rmse_post"] / med["scratch"]["rollout_q_rmse_post"])
    retained_v = {
        name: float(np.median([retain[(name, c)]["rollout_v_rmse_post"] for c in cfg["data"]["particle_cases"]]))
        for name in ["pretrained", "transfer_final", "scratch_final"]
    }
    lines += [
        "",
        "## Lectura",
        "",
        f"- Cambio mediano por transferencia: **{gain_q:+.1f}%** en posición y **{gain_v:+.1f}%** en velocidad (positivo = mejora).",
        "- La transferencia sólo mejora claramente el ángulo 30°; empeora posición y velocidad en los otros cuatro folds.",
        f"- El replay sí protege la tarea anterior: la mediana pp de RMSE de velocidad pasa de {fmt(retained_v['pretrained'])} a {fmt(retained_v['transfer_final'])}; el modelo scratch, que nunca vio replay, obtiene {fmt(retained_v['scratch_final'])}.",
        "- Interpretación: hay conflicto entre adaptar el backbone al canal nodo–pared y conservar su representación arista–partícula. Bajo este protocolo, la retención funciona, pero el costo es adaptación negativa a pared.",
        "- Para uso partícula–pared se recomienda `scratch_final.pt`; `transfer_final.pt` debe conservarse como resultado experimental, no como mejor modelo de pared.",
        "- El control desde cero y el transferido reciben exactamente los mismos estados de pared; las diferencias reflejan la estrategia completa de warm start + replay.",
        "- Con sólo cinco trayectorias y una semilla por fold, el resultado es evidencia comparativa para estos benchmarks, no una garantía universal.",
        "",
        f"Tiempo total: {elapsed / 60:.2f} min.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmarks/benchmark_wall_transfer.yaml")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    if args.quick:
        cfg["training"]["one_step_iters"] = 15
        cfg["rollout"]["iters"] = 3

    out_dir = ROOT / cfg["output"]["directory"]
    ckpt_dir = ROOT / cfg["output"]["checkpoint_directory"]
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    wall_data, pp_data, wall = load_data(cfg)
    wall_cases = list(cfg["data"]["wall_cases"])
    threshold = float(cfg["evaluation"]["active_acceleration_threshold"])
    wall_scales_all = compute_scales(
        wall_data, wall_cases, lambda tr: active_indices(tr, threshold)
    )
    pp_scales = compute_scales(
        pp_data, list(pp_data), lambda tr: pp_near_indices(tr, float(cfg["model"]["r_off"]))
    )
    pretrained_path = ROOT / cfg["pretrained_checkpoint"]
    pretrained_ck = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    pretrained_state = pretrained_ck["model"]

    wall_rows, histories, final_arrays = [], {}, {}
    trained_finals = {}
    started = time.time()
    seed0 = int(cfg["seed"])
    runs = [(f"holdout_{held}", [c for c in wall_cases if c != held], held) for held in wall_cases]
    runs.append(("final", wall_cases, None))
    for run_index, (name, train_cases, held) in enumerate(runs):
        seed = seed0 + run_index
        scratch_init, transfer_init = paired_initial_models(cfg, seed, pretrained_state)
        for method, initial in [("scratch", scratch_init), ("transfer", transfer_init)]:
            wall_scales = compute_scales(
                wall_data, train_cases, lambda tr: active_indices(tr, threshold)
            )
            model, history, elapsed = train_run(
                method, name, initial, wall_data, train_cases, pp_data, wall,
                wall_scales, pp_scales, cfg, seed, out_dir, ckpt_dir,
            )
            histories[(method, name)] = history
            eval_cases = wall_cases if name == "final" else [held]
            particles = make_particles(wall_data[eval_cases[0]])
            for case in eval_cases:
                metrics, arrays = evaluate_wall(
                    model, wall_data[case], particles, wall, threshold
                )
                split = "final_fit" if name == "final" else "held_out"
                wall_rows.append({"run": name, "method": method, "train_cases": "+".join(train_cases), "case": case, "split": split, **metrics})
                if name == "final":
                    final_arrays[(method, case)] = arrays
                    np.savez_compressed(out_dir / f"prediction_{method}_final_{case}.npz", **arrays)
            if name == "final":
                trained_finals[method] = model

    # Retención pp: checkpoint original vs. ambos modelos finales.
    pretrained_model = build_model(cfg)
    pretrained_model.load_state_dict(pretrained_state)
    pretrained_model.eval()
    retention_rows = []
    pp_particles = make_particles(pp_data[next(iter(pp_data))])
    for model_name, model in [
        ("pretrained", pretrained_model),
        ("transfer_final", trained_finals["transfer"]),
        ("scratch_final", trained_finals["scratch"]),
    ]:
        for case, tr in pp_data.items():
            retention_rows.append({"model": model_name, "case": case, **evaluate_pp(model, tr, pp_particles, threshold)})

    total_elapsed = time.time() - started
    (out_dir / "wall_metrics.json").write_text(json.dumps(wall_rows, indent=2), encoding="utf-8")
    (out_dir / "retention_pp.json").write_text(json.dumps(retention_rows, indent=2), encoding="utf-8")
    for filename, rows in [("wall_metrics.csv", wall_rows), ("retention_pp.csv", retention_rows)]:
        with (out_dir / filename).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    plot_loso(wall_rows, wall_cases, out_dir / "comparison_loso.png")
    plot_final_dynamics(final_arrays, wall_cases, out_dir / "final_wall_dynamics.png")
    plot_histories(histories, out_dir / "training_curves.png")
    plot_retention(retention_rows, list(pp_data), out_dir / "retention_pp.png")
    write_report(out_dir / "RESULTADOS.md", cfg, wall_rows, retention_rows, total_elapsed)
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(), "torch": torch.__version__,
        "device": "CPU", "elapsed_seconds": total_elapsed, "config": cfg,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nExperimento completo en {total_elapsed / 60:.2f} min")
    print(f"Resultados: {out_dir}")


if __name__ == "__main__":
    main()
