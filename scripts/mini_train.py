"""Mini-entrenamiento de SLGNN-v2 (smoke de M5): overfit del benchmark de
2 esferas en colisión oblicua, en CPU.

Entrena sobre targets de aceleración por diferencias finitas (translacional y
angular) y evalúa con un rollout corto comparado contra la trayectoria DEM.

Uso:
    python scripts/mini_train.py [--iters 300] [--batch 8] [--lr 3e-3]
"""

import argparse
import time
from pathlib import Path

import torch

from slgnn import (Particles, SLGNN, SLGNNConfig, default_scales,
                   finite_difference_accelerations, load_case, rollout)

ROOT = Path(__file__).resolve().parent.parent
CASE = ROOT / "data" / "extracted" / "Benchmark_2Spheres_Oblique_Collision" / "1x"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    scales = default_scales()
    tr = scales.nondim(load_case(CASE, dt=1e-4))
    t_steps, n, _ = tr.q.shape
    print(f"caso: {CASE.name}  T={t_steps}  N={n}  dt~={tr.dt:.4f} (adimensional)")

    a_ref = finite_difference_accelerations(tr.v, tr.dt)      # [T-1,N,3]
    alpha_ref = finite_difference_accelerations(tr.omega, tr.dt)

    particles = Particles.uniform(n, m=tr.m[0].item(), radius=tr.radii[0].item())
    model = SLGNN(SLGNNConfig())
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parámetros: {n_params}")

    def full_loss():
        """Pérdida sobre los T-1 pasos completos (métrica estable; la de
        minibatch oscila porque la colisión ocupa ~5 de 100 pasos)."""
        with torch.no_grad():
            total = 0.0
            for k in range(t_steps - 1):
                out = model(tr.q[k], tr.v[k], tr.omega[k], particles)
                total += (out.a - a_ref[k]).pow(2).mean().item()
                total += (out.alpha - alpha_ref[k]).pow(2).mean().item()
            return total / (t_steps - 1)

    loss_before = full_loss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    t0 = time.time()
    for it in range(args.iters):
        ks = torch.randint(0, t_steps - 1, (args.batch,))
        opt.zero_grad()
        loss = torch.zeros(())
        for k in ks.tolist():
            out = model(tr.q[k], tr.v[k], tr.omega[k], particles)
            loss = loss + (out.a - a_ref[k]).pow(2).mean()
            loss = loss + (out.alpha - alpha_ref[k]).pow(2).mean()
        loss = loss / args.batch
        loss.backward()
        opt.step()
        if it % 50 == 0 or it == args.iters - 1:
            print(f"iter {it:4d}  loss(minibatch) {loss.item():.4f}")
    print(f"tiempo entrenamiento: {time.time() - t0:.1f}s")
    loss_after = full_loss()
    print(f"pérdida trayectoria completa: {loss_before:.4f} -> {loss_after:.4f} "
          f"(reducción {loss_before / max(loss_after, 1e-12):.0f}x)")

    # rollout completo desde el estado inicial vs. trayectoria DEM
    h = t_steps - 1
    qs, vs, ws = rollout(
        model, tr.q[0], tr.v[0], tr.omega[0], particles, wall=None,
        dt=tr.dt, n_steps=h,
    )
    rmse_q = (qs - tr.q[: h + 1]).pow(2).sum(-1).mean().sqrt().item()
    rmse_baseline = (
        (tr.q[0].unsqueeze(0) - tr.q[: h + 1]).pow(2).sum(-1).mean().sqrt().item()
    )
    print(f"rollout {h} pasos  RMSE posición: {rmse_q:.4f} d_p "
          f"(baseline 'quieto': {rmse_baseline:.4f} d_p)")


if __name__ == "__main__":
    main()
