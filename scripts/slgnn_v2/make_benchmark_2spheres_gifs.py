"""Genera GIFs DEM vs. SLGNN para el benchmark de dos esferas.

La cámara usa el marco del centro de masa y una ventana temporal alrededor del
choque. De ese modo se ve la interacción en vez de perderla por la traslación
global del sistema. Cada panel conserva exactamente los mismos límites y la
misma escala; el radio dibujado es el radio físico adimensional de la esfera.

Uso:
    python scripts/slgnn_v2/make_benchmark_2spheres_gifs.py
    python scripts/slgnn_v2/make_benchmark_2spheres_gifs.py --cases 1x 4x --fps 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
import numpy as np
import torch

from slgnn import Particles, SLGNN, SLGNNConfig, default_scales, load_case
from slgnn.integrator import semi_implicit_step


ROOT = Path(__file__).resolve().parents[2]
DATASET = "Benchmark_2Spheres_Oblique_Collision"


def load_model(checkpoint: Path):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = SLGNN(SLGNNConfig(**ck["model_config"]))
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def load_trajectory(case: str, dt: float):
    case_dir = ROOT / "data" / "extracted" / DATASET / case
    return default_scales().nondim(load_case(case_dir, dt=dt))


@torch.no_grad()
def predict(model, tr, particles):
    q, v, w = tr.q[0], tr.v[0], tr.omega[0]
    qs, vs, ws = [q], [v], [w]
    for k in range(tr.q.shape[0] - 1):
        q, v, w, _ = semi_implicit_step(
            model,
            q,
            v,
            w,
            particles,
            wall=None,
            t=k * tr.dt,
            dt=tr.dt,
        )
        qs.append(q)
        vs.append(v)
        ws.append(w)
    return torch.stack(qs), torch.stack(vs), torch.stack(ws)


def collision_window(tr, pre: int, post: int):
    acceleration = (tr.v[1:] - tr.v[:-1]) / tr.dt
    active = torch.where(acceleration.norm(dim=-1).amax(dim=-1) > 1e-3)[0]
    if not active.numel():
        return 0, tr.q.shape[0] - 1
    start = max(0, int(active[0]) - pre)
    end = min(tr.q.shape[0] - 1, int(active[-1]) + 1 + post)
    return start, end


def square_limits(q_ref, q_pred, radius):
    points = np.concatenate([q_ref[..., :2].reshape(-1, 2), q_pred[..., :2].reshape(-1, 2)])
    lo = points.min(axis=0) - radius
    hi = points.max(axis=0) + radius
    center = 0.5 * (lo + hi)
    half = max(float((hi - lo).max()) * 0.58, 1.25 * radius)
    return (center[0] - half, center[0] + half), (center[1] - half, center[1] + half)


def make_gif(case, tr, q_pred, v_pred, out_path, fps, pre, post, trail):
    start, end = collision_window(tr, pre, post)
    frame_steps = np.arange(start, end + 1)

    q_ref = tr.q.numpy()
    v_ref = tr.v.numpy()
    q_hat = q_pred.numpy()
    v_hat = v_pred.numpy()

    # Una cámara común en el marco del centro de masa permite comparar la
    # geometría relativa sin ocultar diferencias de conservación del COM.
    com_ref = q_ref.mean(axis=1, keepdims=True)
    q_ref_cam = q_ref - com_ref
    q_hat_cam = q_hat - com_ref
    radius = float(tr.radii[0])
    xlim, ylim = square_limits(
        q_ref_cam[start : end + 1], q_hat_cam[start : end + 1], radius
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True)
    titles = ["DEM (referencia)", "SLGNN (predicción)"]
    colors = ["#2474B5", "#F28E2B"]
    circles = []
    trails = []
    distance_labels = []
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontweight="bold")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x relativo al centro de masa [dₚ]")
        ax.grid(alpha=0.18)
        ax.axhline(0.0, color="0.75", linewidth=0.8)
        ax.axvline(0.0, color="0.75", linewidth=0.8)
        panel_circles = []
        panel_trails = []
        for particle, color in enumerate(colors):
            circle = Circle((0, 0), radius, facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.92)
            ax.add_patch(circle)
            panel_circles.append(circle)
            (line,) = ax.plot([], [], color=color, linewidth=1.8, alpha=0.65)
            panel_trails.append(line)
        distance_labels.append(ax.text(0.03, 0.96, "", transform=ax.transAxes, va="top"))
        circles.append(panel_circles)
        trails.append(panel_trails)
    axes[0].set_ylabel("y relativo al centro de masa [dₚ]")

    fig.suptitle(f"Colisión oblicua de dos esferas — caso {case}", fontsize=14, fontweight="bold")
    status = fig.text(0.5, 0.025, "", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))

    def update(frame_index):
        k = int(frame_steps[frame_index])
        datasets = [(q_ref_cam, v_ref), (q_hat_cam, v_hat)]
        for panel, (q_data, _) in enumerate(datasets):
            first = max(start, k - trail + 1)
            for particle in range(2):
                circles[panel][particle].center = tuple(q_data[k, particle, :2])
                trails[panel][particle].set_data(
                    q_data[first : k + 1, particle, 0],
                    q_data[first : k + 1, particle, 1],
                )
            distance = np.linalg.norm(q_data[k, 1] - q_data[k, 0])
            distance_labels[panel].set_text(f"distancia centros: {distance:.3f} dₚ")

        q_error = np.sqrt(np.mean(np.sum((q_hat[k] - q_ref[k]) ** 2, axis=-1)))
        v_error = np.sqrt(np.mean(np.sum((v_hat[k] - v_ref[k]) ** 2, axis=-1)))
        time_ms = k * tr.dt * 10.0  # dt adim * T0=0.01 s; convertir a ms
        status.set_text(
            f"paso {k:03d}  ·  t = {time_ms:.2f} ms  ·  "
            f"RMSE posición = {q_error:.3f} dₚ  ·  RMSE velocidad = {v_error:.3f}"
        )
        artists = [status]
        for panel in range(2):
            artists.extend(circles[panel])
            artists.extend(trails[panel])
            artists.append(distance_labels[panel])
        return artists

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_steps),
        interval=1000 / fps,
        blit=False,
        repeat=True,
    )
    animation.save(out_path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
    return start, end


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/slgnn_v2/benchmarks/benchmark_2spheres/final_all.pt",
    )
    parser.add_argument("--cases", nargs="+", default=["1x", "2x", "4x"])
    parser.add_argument("--out", default="results/slgnn_v2/benchmarks/benchmark_2spheres/gifs")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--pre-contact", type=int, default=10)
    parser.add_argument("--post-contact", type=int, default=25)
    parser.add_argument("--trail", type=int, default=10)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model, ck = load_model(checkpoint)
    dt = float(ck["config"]["data"]["dt"])
    for case in args.cases:
        if case not in {"1x", "2x", "4x"}:
            raise ValueError(f"Caso no reconocido: {case}")
        tr = load_trajectory(case, dt)
        particles = Particles.uniform(
            tr.q.shape[1],
            m=tr.m[0].item(),
            radius=tr.radii[0].item(),
            dtype=tr.q.dtype,
        )
        q_pred, v_pred, _ = predict(model, tr, particles)
        out_path = out_dir / f"slgnn_vs_dem_{case}.gif"
        start, end = make_gif(
            case,
            tr,
            q_pred,
            v_pred,
            out_path,
            fps=args.fps,
            pre=args.pre_contact,
            post=args.post_contact,
            trail=args.trail,
        )
        print(f"{case}: pasos {start}..{end} -> {out_path}")


if __name__ == "__main__":
    main()
