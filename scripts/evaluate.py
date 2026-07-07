"""Evalúa un checkpoint de SLGNN-v2 entrenado: rollout de inferencia contra la
trayectoria DEM real, métricas, y gráficos comparativos.

Genera en `--out`:
  - rmse_vs_time.png     RMSE de posición/velocidad por paso vs. baseline estático
  - energy.png           balance mecánico (T+V) y disipación acumulada por paso
  - snapshots.png        vistas 2D (xy) del sistema real vs. predicho en varios instantes
  - trajectories.png      trayectoria (posición vs. tiempo) de unas pocas partículas
  - metrics.json          métricas resumidas
  - rollout.gif           (opcional, --gif) animación real vs. predicho

Uso:
    python scripts/evaluate.py --checkpoint checkpoints/gravity_rollout/best.pt \
        --config configs/gravity_rollout.yaml --case CASE06 --horizon 200

    # sobre el caso de extrapolación energética (held-out, nunca visto)
    python scripts/evaluate.py --checkpoint checkpoints/gravity_rollout/best.pt \
        --config configs/gravity_rollout.yaml --case CASE07 --out results/case07
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # sin display: solo guardar PNG/GIF
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from slgnn.experiment import load_case_by_name, load_checkpoint, load_split
from slgnn.integrator import semi_implicit_step

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Rollout de inferencia con diagnósticos por paso (energía, penetración)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def rollout_with_diagnostics(model, tr, particles, wall, g_vec, dt, horizon):
    q, v, w = tr.q[0], tr.v[0], tr.omega[0]
    qs, vs, ws = [q], [v], [w]
    energy, diss_pp, diss_pw, max_pen = [], [], [], []
    for _ in range(horizon):
        q, v, w, out = semi_implicit_step(
            model, q, v, w, particles, wall, t=0.0, dt=dt, g_vec=g_vec
        )
        d = out.diagnostics
        T = 0.5 * (particles.m * v.pow(2).sum(-1)).sum() + 0.5 * (
            particles.inertia * w.pow(2).sum(-1)
        ).sum()
        energy.append((T + d["V_g"] + d["V_pp"] + d["V_pW"]).item())
        diss_pp.append(d["R_pp"].item())
        diss_pw.append(d["R_pW"].item())
        pen = (particles.radii - d["phi"]).clamp(min=0).max().item()
        max_pen.append(pen)
        qs.append(q)
        vs.append(v)
        ws.append(w)
    return (
        torch.stack(qs), torch.stack(vs), torch.stack(ws),
        {"energy": energy, "diss_pp": diss_pp, "diss_pw": diss_pw, "max_pen": max_pen},
    )


def rmse_over_time(pred, ref):
    return (pred - ref).pow(2).sum(-1).mean(-1).sqrt().numpy()


# --------------------------------------------------------------------------- #
# Gráficos
# --------------------------------------------------------------------------- #
def plot_rmse(t, rmse_q, rmse_v, base_q, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(t, rmse_q, label="SLGNN (predicho)")
    axes[0].plot(t, base_q, "--", color="gray", label="baseline: partícula quieta")
    axes[0].set_xlabel("paso"); axes[0].set_ylabel("RMSE posición (diámetros)")
    axes[0].set_title("Error de posición vs. tiempo"); axes[0].legend()
    axes[1].plot(t, rmse_v, color="tab:orange")
    axes[1].set_xlabel("paso"); axes[1].set_ylabel("RMSE velocidad (adim.)")
    axes[1].set_title("Error de velocidad vs. tiempo")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_energy(diag, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(diag["energy"])
    axes[0].set_xlabel("paso"); axes[0].set_ylabel("T + V (adim.)")
    axes[0].set_title("Energía mecánica del rollout predicho")
    axes[1].plot(diag["diss_pp"], label="R_pp (pares)")
    axes[1].plot(diag["diss_pw"], label="R_pW (pared)")
    axes[1].plot(diag["max_pen"], "--", color="red", label="penetración máx.")
    axes[1].set_xlabel("paso"); axes[1].set_ylabel("valor (adim.)")
    axes[1].set_title("Disipación de Rayleigh y penetración")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_snapshots(q_pred, q_ref, steps, out_path):
    n = len(steps)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4))
    if n == 1:
        axes = [axes]
    for ax, s in zip(axes, steps):
        ax.scatter(q_ref[s, :, 0], q_ref[s, :, 1], s=25, c="tab:blue",
                  label="DEM real", alpha=0.8)
        ax.scatter(q_pred[s, :, 0], q_pred[s, :, 1], s=25, marker="x",
                  c="tab:red", label="SLGNN predicho", alpha=0.9)
        ax.set_title(f"paso {s}")
        ax.set_xlabel("x"); ax.set_ylabel("y (eje de gravedad)")
        ax.set_aspect("equal")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_trajectories(q_pred, q_ref, particle_idx, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    t = np.arange(q_pred.shape[0])
    for i in particle_idx:
        line, = axes[0].plot(t, q_ref[:, i, 0], label=f"partícula {i} (real)")
        axes[0].plot(t, q_pred[:, i, 0], "--", color=line.get_color())
        line, = axes[1].plot(t, q_ref[:, i, 1], label=f"partícula {i} (real)")
        axes[1].plot(t, q_pred[:, i, 1], "--", color=line.get_color())
    axes[0].set_xlabel("paso"); axes[0].set_ylabel("x"); axes[0].set_title("Posición x (línea=real, discontinua=predicho)")
    axes[1].set_xlabel("paso"); axes[1].set_ylabel("y"); axes[1].set_title("Posición y (eje de gravedad)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def save_gif(q_pred, q_ref, out_path, stride=1, fps=15):
    from matplotlib.animation import FuncAnimation, PillowWriter

    steps = list(range(0, q_pred.shape[0], stride))
    fig, ax = plt.subplots(figsize=(5, 5))
    sc_ref = ax.scatter([], [], s=25, c="tab:blue", label="DEM real")
    sc_pred = ax.scatter([], [], s=25, marker="x", c="tab:red", label="SLGNN predicho")
    ax.set_xlim(q_ref[..., 0].min().item(), q_ref[..., 0].max().item())
    ax.set_ylim(q_ref[..., 1].min().item(), q_ref[..., 1].max().item())
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    title = ax.set_title("")

    def update(k):
        s = steps[k]
        sc_ref.set_offsets(q_ref[s, :, :2].numpy())
        sc_pred.set_offsets(q_pred[s, :, :2].numpy())
        title.set_text(f"paso {s}")
        return sc_ref, sc_pred, title

    anim = FuncAnimation(fig, update, frames=len(steps), blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None,
                    help="por defecto usa el config guardado dentro del checkpoint")
    ap.add_argument("--case", default=None,
                    help="CASE a evaluar; por defecto el val_case del config")
    ap.add_argument("--horizon", type=int, default=None,
                    help="pasos de rollout; por defecto toda la trayectoria disponible")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gif", action="store_true", help="guardar animación (más pesado)")
    ap.add_argument("--gif-stride", type=int, default=2)
    args = ap.parse_args()

    ck_path = Path(args.checkpoint)
    model, ck = load_checkpoint(ck_path)
    cfg = yaml.safe_load(Path(args.config).read_text()) if args.config else ck["config"]

    _, _, val_default, wall, g_vec, particles = load_split(cfg, ROOT)
    case_name = args.case or cfg["data"]["val_case"]
    tr = val_default if case_name == cfg["data"]["val_case"] else load_case_by_name(cfg, ROOT, case_name)

    horizon = args.horizon or (tr.q.shape[0] - 1)
    horizon = min(horizon, tr.q.shape[0] - 1)

    out_dir = Path(args.out) if args.out else ROOT / "results" / f"{ck_path.stem}_{case_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"checkpoint: {ck_path}  (tag={ck['tag']})")
    print(f"caso: {case_name}  horizonte: {horizon} pasos  N={tr.q.shape[1]}")

    q_pred, v_pred, w_pred, diag = rollout_with_diagnostics(
        model, tr, particles, wall, g_vec, tr.dt, horizon
    )
    q_ref, v_ref = tr.q[: horizon + 1], tr.v[: horizon + 1]

    rmse_q = rmse_over_time(q_pred, q_ref)      # una cifra por paso (incluye t=0)
    rmse_v = rmse_over_time(v_pred, v_ref)
    base_q = rmse_over_time(tr.q[0].unsqueeze(0).expand_as(q_ref), q_ref)

    # "pooled": una sola cifra que agrupa todos los pasos 1..horizon a la vez
    # (mismo criterio que usa train.py.validate durante el entrenamiento) —
    # NO es el promedio de rmse_q por paso: sqrt(mean(x)) != mean(sqrt(x)).
    def pooled_rmse(pred, ref):
        return float((pred[1:] - ref[1:]).pow(2).sum(-1).mean().sqrt())

    metrics = {
        "checkpoint_tag": ck["tag"],
        "case": case_name,
        "horizon": horizon,
        "rmse_q_pooled": pooled_rmse(q_pred, q_ref),        # comparable al log de entrenamiento
        "rmse_q_last_step": float(rmse_q[-1]),               # error acumulado en el último paso
        "rmse_v_pooled": pooled_rmse(v_pred, v_ref),
        "rmse_v_last_step": float(rmse_v[-1]),
        "baseline_q_pooled": pooled_rmse(
            tr.q[0].unsqueeze(0).expand_as(q_ref), q_ref
        ),
        "baseline_q_last_step": float(base_q[-1]),
        "max_penetration": float(max(diag["max_pen"])),
        "energy_initial": diag["energy"][0],
        "energy_final": diag["energy"][-1],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    # rmse_q/rmse_v/base_q ya incluyen t=0 (error nulo, mismo estado inicial)
    t = np.arange(horizon + 1)
    plot_rmse(t, rmse_q, rmse_v, base_q, out_dir / "rmse_vs_time.png")
    plot_energy(diag, out_dir / "energy.png")
    snap_steps = sorted(set(int(x) for x in np.linspace(0, horizon, 5)))
    plot_snapshots(q_pred, q_ref, snap_steps, out_dir / "snapshots.png")
    n_particles_to_show = min(5, tr.q.shape[1])
    plot_trajectories(q_pred, q_ref, range(n_particles_to_show), out_dir / "trajectories.png")

    if args.gif:
        save_gif(q_pred, q_ref, out_dir / "rollout.gif", stride=args.gif_stride)

    print(f"\ngráficos y métricas guardados en: {out_dir}")


if __name__ == "__main__":
    main()
