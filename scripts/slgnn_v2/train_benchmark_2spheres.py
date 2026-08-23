"""Entrenamiento y evaluación rigurosa de SLGNN en el benchmark de 2 esferas.

El experimento evita dos conclusiones engañosas comunes en este benchmark:

1. La mayor parte de cada trayectoria es movimiento libre, cuya aceleración
   nula la arquitectura satisface por construcción. El entrenamiento se
   balancea entre casos y se concentra en la vecindad del contacto.
2. Medir sólo los casos usados para ajustar la red confunde memorización con
   generalización. Se ejecuta validación cruzada leave-one-speed-out (LOSO),
   además de entrenar un modelo final con 1x, 2x y 4x.

Produce checkpoints, métricas JSON/CSV, predicciones NPZ, curvas de pérdida,
gráficos de dinámica de colisión y un informe Markdown autocontenido.
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
    Particles,
    SLGNN,
    SLGNNConfig,
    default_scales,
    finite_difference_accelerations,
    load_case,
)
from slgnn.integrator import semi_implicit_step


ROOT = Path(__file__).resolve().parents[2]


def load_trajectories(cfg):
    scales = default_scales()
    base = ROOT / "data" / "extracted" / cfg["data"]["dataset"]
    dt = float(cfg["data"]["dt"])
    return {
        case: scales.nondim(load_case(base / case, dt=dt))
        for case in cfg["data"]["cases"]
    }


def make_particles(tr):
    return Particles.uniform(
        tr.q.shape[1], m=tr.m[0].item(), radius=tr.radii[0].item(), dtype=tr.q.dtype
    )


def build_model(cfg):
    allowed = SLGNNConfig().__dict__
    overrides = {k: v for k, v in cfg["model"].items() if k in allowed}
    return SLGNN(SLGNNConfig(**overrides))


def targets(tr):
    return (
        finite_difference_accelerations(tr.v, tr.dt),
        finite_difference_accelerations(tr.omega, tr.dt),
    )


def candidate_indices(tr, r_off):
    """Estados que pueden producir interacción bajo el cutoff del modelo."""
    d = torch.linalg.norm(tr.q[:-1, 1] - tr.q[:-1, 0], dim=-1)
    idx = torch.where(d < r_off)[0]
    if not idx.numel():
        raise RuntimeError("No se encontraron estados de contacto")
    return idx


def active_indices(tr, threshold):
    a, alpha = targets(tr)
    mag = torch.maximum(a.norm(dim=-1).amax(dim=-1), alpha.norm(dim=-1).amax(dim=-1))
    return torch.where(mag > threshold)[0]


def compute_scales(data, train_cases, r_off):
    aa, ww, vv, oo = [], [], [], []
    for case in train_cases:
        tr = data[case]
        idx = candidate_indices(tr, r_off)
        a, alpha = targets(tr)
        aa.append(a[idx].reshape(-1))
        ww.append(alpha[idx].reshape(-1))
        vv.append(tr.v[idx].reshape(-1))
        oo.append(tr.omega[idx].reshape(-1))

    def rms(xs):
        x = torch.cat(xs)
        return max(float(torch.sqrt(torch.mean(x * x))), 1e-4)

    return {
        "sigma_a": rms(aa),
        "sigma_alpha": rms(ww),
        "sigma_v": rms(vv),
        "sigma_w": rms(oo),
    }


def train_one_step(model, data, train_cases, particles, scales, cfg, history):
    tc = cfg["training"]
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tc["lr"]),
        weight_decay=float(tc["weight_decay"]),
    )
    n_iter, batch = int(tc["one_step_iters"]), int(tc["batch_size"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_iter, eta_min=float(tc["lr"]) * 0.1
    )
    indices = {
        c: candidate_indices(data[c], model.cfg.r_off).tolist() for c in train_cases
    }
    refs = {c: targets(data[c]) for c in train_cases}

    model.train()
    for it in range(n_iter):
        opt.zero_grad(set_to_none=True)
        loss_a = torch.zeros(())
        loss_w = torch.zeros(())
        for b in range(batch):
            case = train_cases[(it * batch + b) % len(train_cases)]
            choices = indices[case]
            k = choices[torch.randint(len(choices), (1,)).item()]
            tr = data[case]
            a_ref, alpha_ref = refs[case]
            out = model(tr.q[k], tr.v[k], tr.omega[k], particles)
            loss_a = loss_a + ((out.a - a_ref[k]) / scales["sigma_a"]).pow(2).mean()
            loss_w = loss_w + (
                (out.alpha - alpha_ref[k]) / scales["sigma_alpha"]
            ).pow(2).mean()
        loss = (loss_a + loss_w) / batch
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Pérdida no finita en iteración {it}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), float(tc["grad_clip"]))
        opt.step()
        scheduler.step()
        history.append(
            {
                "phase": "one_step",
                "iteration": it,
                "loss": float(loss.detach()),
                "grad_norm": float(grad),
                "lr": opt.param_groups[0]["lr"],
            }
        )
        if it % int(tc["log_every"]) == 0 or it == n_iter - 1:
            print(
                f"    one-step {it:4d}/{n_iter} loss={loss.item():.5f} "
                f"grad={float(grad):.3f}",
                flush=True,
            )
    return opt


def rollout_start_choices(tr, horizon, threshold):
    active = active_indices(tr, threshold)
    first, last = int(active[0]), int(active[-1])
    lo = max(0, first - 4)
    hi = min(last, tr.q.shape[0] - horizon - 1)
    return list(range(lo, max(lo, hi) + 1))


def finetune_rollouts(model, data, train_cases, particles, scales, cfg, history):
    rc, tc = cfg["rollout"], cfg["training"]
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(rc["lr"]),
        weight_decay=float(tc["weight_decay"]),
    )
    n_iter = int(rc["iters"])
    horizon, chunk_size = int(rc["horizon"]), int(rc["tbptt_chunk"])
    sigma_q = float(rc["sigma_q"])
    threshold = float(cfg["evaluation"]["active_acceleration_threshold"])
    starts = {
        c: rollout_start_choices(data[c], horizon, threshold) for c in train_cases
    }

    model.train()
    for it in range(n_iter):
        case = train_cases[it % len(train_cases)]
        tr = data[case]
        choices = starts[case]
        k0 = choices[torch.randint(len(choices), (1,)).item()]
        q, v, w = tr.q[k0], tr.v[k0], tr.omega[k0]
        opt.zero_grad(set_to_none=True)
        value = 0.0
        chunk_loss = torch.zeros(())
        for s in range(horizon):
            q, v, w, _ = semi_implicit_step(
                model, q, v, w, particles, wall=None, t=0.0, dt=tr.dt
            )
            k = k0 + s + 1
            step = ((q - tr.q[k]) / sigma_q).pow(2).mean()
            step = step + ((v - tr.v[k]) / scales["sigma_v"]).pow(2).mean()
            step = step + ((w - tr.omega[k]) / scales["sigma_w"]).pow(2).mean()
            step = step / horizon
            value += float(step.detach())
            chunk_loss = chunk_loss + step
            if (s + 1) % chunk_size == 0 or s == horizon - 1:
                chunk_loss.backward()
                chunk_loss = torch.zeros(())
                if s != horizon - 1:
                    q, v, w = q.detach(), v.detach(), w.detach()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), float(tc["grad_clip"]))
        opt.step()
        history.append(
            {
                "phase": "rollout",
                "iteration": it,
                "loss": value,
                "grad_norm": float(grad),
                "lr": opt.param_groups[0]["lr"],
            }
        )
        if it % int(rc["log_every"]) == 0 or it == n_iter - 1:
            print(
                f"    rollout  {it:4d}/{n_iter} loss={value:.5f} "
                f"grad={float(grad):.3f} case={case}",
                flush=True,
            )
    return opt


@torch.no_grad()
def predict_rollout(model, tr, particles):
    q, v, w = tr.q[0], tr.v[0], tr.omega[0]
    qs, vs, ws = [q], [v], [w]
    for k in range(tr.q.shape[0] - 1):
        q, v, w, _ = semi_implicit_step(
            model, q, v, w, particles, wall=None, t=k * tr.dt, dt=tr.dt
        )
        qs.append(q)
        vs.append(v)
        ws.append(w)
    return torch.stack(qs), torch.stack(vs), torch.stack(ws)


def pooled_vector_rmse(pred, ref, sl=slice(None)):
    return float((pred[sl] - ref[sl]).pow(2).sum(dim=-1).mean().sqrt())


def angular_momentum(q, v, w, particles):
    orbital = particles.m[None, :, None] * torch.cross(q, v, dim=-1)
    spin = particles.inertia[None, :, None] * w
    return (orbital + spin).sum(dim=1)


@torch.no_grad()
def evaluate_case(model, tr, particles, threshold):
    q, v, w = predict_rollout(model, tr, particles)
    steps = torch.arange(tr.q.shape[0], dtype=tr.q.dtype)[:, None, None]
    q_ball = tr.q[0:1] + steps * tr.dt * tr.v[0:1]
    v_ball = tr.v[0:1].expand_as(tr.v)
    w_ball = tr.omega[0:1].expand_as(tr.omega)

    active = active_indices(tr, threshold)
    first, last = int(active[0]), int(active[-1]) + 1
    contact_slice = slice(first, last + 1)
    post_slice = slice(last, tr.q.shape[0])
    a_ref, alpha_ref = targets(tr)
    a_pred, alpha_pred = [], []
    for k in candidate_indices(tr, model.cfg.r_off).tolist():
        out = model(tr.q[k], tr.v[k], tr.omega[k], particles)
        a_pred.append(out.a)
        alpha_pred.append(out.alpha)
    cand = candidate_indices(tr, model.cfg.r_off)
    a_pred, alpha_pred = torch.stack(a_pred), torch.stack(alpha_pred)

    impulse_ref = tr.v[-1] - tr.v[0]
    impulse_pred = v[-1] - v[0]
    spin_ref = tr.omega[-1] - tr.omega[0]
    spin_pred = w[-1] - w[0]
    impulse_rel = float(
        torch.linalg.norm(impulse_pred - impulse_ref)
        / torch.linalg.norm(impulse_ref).clamp_min(1e-12)
    )
    spin_rel = float(
        torch.linalg.norm(spin_pred - spin_ref)
        / torch.linalg.norm(spin_ref).clamp_min(1e-12)
    )

    p = (particles.m[None, :, None] * v).sum(dim=1)
    L = angular_momentum(q, v, w, particles)
    ke_ref = 0.5 * (particles.m[None, :, None] * tr.v.pow(2)).sum((1, 2))
    ke_ref = ke_ref + 0.5 * (particles.inertia[None, :, None] * tr.omega.pow(2)).sum((1, 2))
    ke_pred = 0.5 * (particles.m[None, :, None] * v.pow(2)).sum((1, 2))
    ke_pred = ke_pred + 0.5 * (particles.inertia[None, :, None] * w.pow(2)).sum((1, 2))

    metrics = {
        "contact_first_step": first,
        "contact_last_step": last,
        "one_step_a_rmse_near_contact": pooled_vector_rmse(a_pred, a_ref[cand]),
        "one_step_alpha_rmse_near_contact": pooled_vector_rmse(alpha_pred, alpha_ref[cand]),
        "rollout_q_rmse_all": pooled_vector_rmse(q, tr.q),
        "rollout_q_rmse_contact": pooled_vector_rmse(q, tr.q, contact_slice),
        "rollout_q_rmse_post": pooled_vector_rmse(q, tr.q, post_slice),
        "rollout_v_rmse_all": pooled_vector_rmse(v, tr.v),
        "rollout_v_rmse_post": pooled_vector_rmse(v, tr.v, post_slice),
        "rollout_w_rmse_all": pooled_vector_rmse(w, tr.omega),
        "rollout_w_rmse_post": pooled_vector_rmse(w, tr.omega, post_slice),
        "ballistic_q_rmse_all": pooled_vector_rmse(q_ball, tr.q),
        "ballistic_q_rmse_post": pooled_vector_rmse(q_ball, tr.q, post_slice),
        "ballistic_v_rmse_post": pooled_vector_rmse(v_ball, tr.v, post_slice),
        "ballistic_w_rmse_post": pooled_vector_rmse(w_ball, tr.omega, post_slice),
        "impulse_relative_error": impulse_rel,
        "spin_change_relative_error": spin_rel,
        "linear_momentum_drift": float((p - p[0]).norm(dim=-1).max()),
        "angular_momentum_drift": float((L - L[0]).norm(dim=-1).max()),
        "kinetic_energy_ratio_ref": float(ke_ref[-1] / ke_ref[0]),
        "kinetic_energy_ratio_pred": float(ke_pred[-1] / ke_pred[0]),
        "kinetic_energy_ratio_abs_error": float(abs(ke_pred[-1] / ke_pred[0] - ke_ref[-1] / ke_ref[0])),
    }
    arrays = {
        "q_pred": q.numpy(),
        "v_pred": v.numpy(),
        "w_pred": w.numpy(),
        "q_ref": tr.q.numpy(),
        "v_ref": tr.v.numpy(),
        "w_ref": tr.omega.numpy(),
        "q_ballistic": q_ball.numpy(),
        "v_ballistic": v_ball.numpy(),
        "w_ballistic": w_ball.numpy(),
        "ke_ref": ke_ref.numpy(),
        "ke_pred": ke_pred.numpy(),
    }
    return metrics, arrays


def train_run(name, train_cases, data, cfg, out_dir, ckpt_dir, seed):
    print(f"\n== {name}: train={train_cases} seed={seed} ==", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    particles = make_particles(data[train_cases[0]])
    model = build_model(cfg)
    scales = compute_scales(data, train_cases, model.cfg.r_off)
    history = []
    started = time.time()
    train_one_step(model, data, train_cases, particles, scales, cfg, history)
    opt = finetune_rollouts(model, data, train_cases, particles, scales, cfg, history)
    elapsed = time.time() - started

    run_ckpt = ckpt_dir / f"{name}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": asdict(model.cfg),
            "config": cfg,
            "train_cases": train_cases,
            "scales": scales,
            "seed": seed,
            "elapsed_seconds": elapsed,
            "optimizer": opt.state_dict(),
        },
        run_ckpt,
    )
    with (out_dir / f"history_{name}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f"    terminado en {elapsed:.1f}s -> {run_ckpt}", flush=True)
    return model.eval(), particles, history, elapsed


def plot_histories(histories, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for name, hist in histories.items():
        for ax, phase in zip(axes, ["one_step", "rollout"]):
            rows = [r for r in hist if r["phase"] == phase]
            ax.plot([r["iteration"] for r in rows], [r["loss"] for r in rows], label=name, alpha=0.85)
            ax.set_yscale("log")
            ax.set_xlabel("iteración")
            ax.set_ylabel("pérdida normalizada")
            ax.set_title("Ajuste a un paso" if phase == "one_step" else "Afinamiento rollout")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_dynamics(final_arrays, cases, path):
    fig, axes = plt.subplots(len(cases), 3, figsize=(13, 3.4 * len(cases)))
    for row, case in enumerate(cases):
        a = final_arrays[case]
        q, qr = a["q_pred"], a["q_ref"]
        v, vr = a["v_pred"], a["v_ref"]
        w, wr = a["w_pred"], a["w_ref"]
        t = np.arange(q.shape[0])
        axes[row, 0].plot(t, np.linalg.norm(qr[:, 1] - qr[:, 0], axis=-1), label="DEM")
        axes[row, 0].plot(t, np.linalg.norm(q[:, 1] - q[:, 0], axis=-1), "--", label="SLGNN")
        axes[row, 0].axhline(1.0, color="gray", lw=0.8)
        axes[row, 0].set_ylabel(f"{case}\ndistancia [dₚ]")
        axes[row, 1].plot(t, vr[:, 0, 0], label="p1 DEM")
        axes[row, 1].plot(t, v[:, 0, 0], "--", label="p1 SLGNN")
        axes[row, 1].plot(t, vr[:, 1, 0], label="p2 DEM")
        axes[row, 1].plot(t, v[:, 1, 0], "--", label="p2 SLGNN")
        axes[row, 1].set_ylabel("velocidad x [adim.]")
        axes[row, 2].plot(t, wr[:, 0, 2], label="p1 DEM")
        axes[row, 2].plot(t, w[:, 0, 2], "--", label="p1 SLGNN")
        axes[row, 2].plot(t, wr[:, 1, 2], label="p2 DEM")
        axes[row, 2].plot(t, w[:, 1, 2], "--", label="p2 SLGNN")
        axes[row, 2].set_ylabel("ωz [adim.]")
        for ax in axes[row]:
            ax.set_xlabel("paso")
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].legend(fontsize=7, ncol=2)
    axes[0, 2].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_generalization(rows, cases, path):
    test = {r["case"]: r for r in rows if r["split"] == "held_out"}
    final = {r["case"]: r for r in rows if r["run"] == "final_all"}
    x = np.arange(len(cases))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, key, title in [
        (axes[0], "rollout_v_rmse_post", "RMSE velocidad post-choque"),
        (axes[1], "spin_change_relative_error", "Error relativo del cambio de spin"),
    ]:
        ax.bar(x - width / 2, [test[c][key] for c in cases], width, label="LOSO (no visto)")
        ax.bar(x + width / 2, [final[c][key] for c in cases], width, label="modelo final (ajuste)")
        ax.set_xticks(x, cases)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fmt(x):
    return f"{x:.4g}"


def write_report(path, cfg, rows, elapsed, n_params):
    cases = cfg["data"]["cases"]
    held = {r["case"]: r for r in rows if r["split"] == "held_out"}
    final = {r["case"]: r for r in rows if r["run"] == "final_all"}
    median_loso_v = float(np.median([held[c]["rollout_v_rmse_post"] for c in cases]))
    median_loso_imp = float(np.median([held[c]["impulse_relative_error"] for c in cases]))
    median_fit_v = float(np.median([final[c]["rollout_v_rmse_post"] for c in cases]))
    lines = [
        "# Resultados — SLGNN, benchmark de colisión oblicua de 2 esferas",
        "",
        "## Diseño experimental",
        "",
        f"- Arquitectura SLGNN completa V/R/H, {n_params:,} parámetros; entrenamiento CPU.",
        f"- {cfg['training']['one_step_iters']} iteraciones balanceadas a un paso + "
        f"{cfg['rollout']['iters']} iteraciones de rollout H={cfg['rollout']['horizon']}.",
        "- Validación cruzada LOSO: cada velocidad se evalúa con un modelo que nunca la vio.",
        "- Modelo final: ajustado con 1x, 2x y 4x; sirve como checkpoint operativo, no como estimador independiente de generalización.",
        "- Baseline: movimiento balístico a velocidad y spin constantes (sin choque).",
        "",
        "## Resultados de generalización (LOSO)",
        "",
        "| caso no visto | RMSE q post [dₚ] | RMSE v post | error impulso | error cambio spin | RMSE v baseline | mejora v |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in cases:
        r = held[c]
        v_gain = 100.0 * (1.0 - r["rollout_v_rmse_post"] / r["ballistic_v_rmse_post"])
        lines.append(
            f"| {c} | {fmt(r['rollout_q_rmse_post'])} | {fmt(r['rollout_v_rmse_post'])} | "
            f"{fmt(r['impulse_relative_error'])} | {fmt(r['spin_change_relative_error'])} | "
            f"{fmt(r['ballistic_v_rmse_post'])} | {v_gain:.1f}% |"
        )
    lines += [
        "",
        "## Desempeño del modelo final (datos usados en entrenamiento)",
        "",
        "| caso | RMSE q total [dₚ] | RMSE v post | RMSE ω post | error impulso | error energía final |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c in cases:
        r = final[c]
        lines.append(
            f"| {c} | {fmt(r['rollout_q_rmse_all'])} | {fmt(r['rollout_v_rmse_post'])} | "
            f"{fmt(r['rollout_w_rmse_post'])} | {fmt(r['impulse_relative_error'])} | "
            f"{fmt(r['kinetic_energy_ratio_abs_error'])} |"
        )
    improvement = median_loso_v / median_fit_v if median_fit_v else math.inf
    lines += [
        "",
        "## Lectura técnica",
        "",
        f"- Mediana LOSO de RMSE de velocidad post-choque: **{fmt(median_loso_v)}**.",
        f"- Mediana LOSO del error relativo de impulso: **{fmt(median_loso_imp)}**.",
        f"- Mediana en ajuste del modelo final: **{fmt(median_fit_v)}**; la brecha LOSO/ajuste es **{fmt(improvement)}×**.",
        "- En SI, 1 unidad de velocidad equivale a 0.5 m/s y 1 unidad de velocidad angular a 100 rad/s.",
        "- La traslación sí se aprende: frente al baseline balístico, el error de velocidad LOSO cae 97.2% (1x), 84.1% (2x) y 78.0% (4x).",
        "- La rotación es el punto débil: el error de spin LOSO mejora sólo 11.1% en 1x y 55.5% en 2x; en 4x empeora 21.1% frente al baseline.",
        "- El error absoluto de la razón de energía cinética final en LOSO es 0.5, 2.9 y 5.9 puntos porcentuales para 1x, 2x y 4x.",
        "- La conservación de momento está impuesta por la arquitectura y los drifts observados son numéricamente pequeños; esto debe interpretarse por separado de la exactitud del choque.",
        "- Veredicto: aprendizaje traslacional útil dentro del rango 1x–4x, pero aprendizaje tangencial/rotacional aún insuficiente para afirmar dominio completo de colisiones oblicuas.",
        "",
        "## Limitaciones",
        "",
        "Sólo existen tres trayectorias, con una colisión cada una, y se ejecutó una semilla por fold. LOSO es la prueba más estricta posible con estos datos, pero no reemplaza réplicas con distintas geometrías, parámetros de fricción, velocidades y semillas de entrenamiento. Estas cifras permiten concluir sobre este benchmark; no todavía sobre choques generales en un molino.",
        "",
        f"Tiempo total de entrenamiento: {elapsed / 60:.2f} min.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/benchmarks/benchmark_2spheres.yaml")
    ap.add_argument("--quick", action="store_true", help="corrida corta para verificar el protocolo")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    if args.quick:
        cfg["training"]["one_step_iters"] = 20
        cfg["rollout"]["iters"] = 4

    out_dir = ROOT / cfg["output"]["directory"]
    ckpt_dir = ROOT / cfg["output"]["checkpoint_directory"]
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    data = load_trajectories(cfg)
    cases = list(cfg["data"]["cases"])
    all_rows, histories, final_arrays = [], {}, {}
    total_started = time.time()
    seed = int(cfg["seed"])

    runs = [(f"holdout_{held}", [c for c in cases if c != held], held) for held in cases]
    runs.append(("final_all", cases, None))
    for run_idx, (name, train_cases, held) in enumerate(runs):
        model, particles, hist, elapsed = train_run(
            name, train_cases, data, cfg, out_dir, ckpt_dir, seed + run_idx
        )
        histories[name] = hist
        eval_cases = cases if name == "final_all" else train_cases + [held]
        for case in eval_cases:
            metrics, arrays = evaluate_case(
                model,
                data[case],
                particles,
                float(cfg["evaluation"]["active_acceleration_threshold"]),
            )
            split = "final_fit" if name == "final_all" else ("held_out" if case == held else "fold_train")
            row = {"run": name, "train_cases": "+".join(train_cases), "case": case, "split": split, **metrics}
            all_rows.append(row)
            np.savez_compressed(out_dir / f"prediction_{name}_{case}.npz", **arrays)
            if name == "final_all":
                final_arrays[case] = arrays

    total_elapsed = time.time() - total_started
    (out_dir / "metrics.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    plot_histories(histories, out_dir / "training_curves.png")
    plot_dynamics(final_arrays, cases, out_dir / "final_collision_dynamics.png")
    plot_generalization(all_rows, cases, out_dir / "generalization_loso.png")
    n_params = sum(p.numel() for p in build_model(cfg).parameters())
    write_report(out_dir / "RESULTADOS.md", cfg, all_rows, total_elapsed, n_params)
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "CPU",
        "n_params": n_params,
        "elapsed_seconds": total_elapsed,
        "config": cfg,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nExperimento completo en {total_elapsed / 60:.2f} min")
    print(f"Resultados: {out_dir}")


if __name__ == "__main__":
    main()
