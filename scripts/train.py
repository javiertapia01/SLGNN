"""Entrenamiento serio de SLGNN-v2 sobre 60Spheres_Gravity, con rollout largo.

Usa todas las herramientas de la arquitectura simultáneamente:
  - canal conservativo (gravedad conocida + V_pp y V_pW aprendidos) con
    fuerzas por Euler-Lagrange (autograd, doble backward);
  - disipación de Rayleigh (R_pp, R_pW) con fuerzas y torques por autograd;
  - canal histórico/residual (f_H, tau_H), regularizado y con pasividad;
  - rotación propia (omega, alpha, torques);
  - pared por SDF (caja cúbica) con su geometría de contacto;
  - integrador de Euler semiimplícito en rollout;
  - las seis pérdidas de §34: L_a, L_alpha, L_roll, L_res, L_pass, L_pen.

Estrategia (ver Informe_Estrategia_Entrenamiento_SLGNN.md):
  1. Warmup a un paso (L_a + L_alpha) para asentar V y R.
  2. Ventanas de rollout de horizonte creciente (4 -> 64), con backprop
     truncado (TBPTT) para acotar la memoria del doble backward, ruido de
     entrada en el estado inicial, y las pérdidas auxiliares activas.

Normalización derivada de los datos (sigmas), gravedad en el eje verificado
empíricamente (−y), y checkpointing por fase con config y sigmas guardados.

Uso:
    python scripts/train.py --config configs/gravity_rollout.yaml
    python scripts/train.py --config configs/gravity_rollout.yaml --smoke
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from slgnn import (BoxSDF, Particles, SLGNN, SLGNNConfig, default_scales,
                   finite_difference_accelerations, load_case)
from slgnn.integrator import semi_implicit_step
from slgnn.losses import (acceleration_loss, angular_acceleration_loss,
                          passivity_loss, penetration_loss,
                          residual_regularization)

ROOT = Path(__file__).resolve().parent.parent
_AXIS = {"x": 0, "y": 1, "z": 2}


# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #
def load_split(cfg):
    """Carga y adimensionaliza train/val, y arma la pared y el vector gravedad."""
    scales = default_scales()
    base = ROOT / "data" / "extracted" / cfg["data"]["dataset"]
    dt = float(cfg["data"]["dt"])

    def _load(case):
        return scales.nondim(load_case(base / case, dt=dt))

    train = [_load(c) for c in cfg["data"]["train_cases"]]
    val = _load(cfg["data"]["val_case"])

    # pared: caja en unidades adimensionales (0.03 m / 0.005 m = 6 diámetros)
    box_min = [scales.length(x) for x in cfg["data"]["box_min"]]
    box_max = [scales.length(x) for x in cfg["data"]["box_max"]]
    wall = BoxSDF(box_min, box_max)

    # gravedad en el eje verificado en los datos (sedimentación en −y)
    g_mag = scales.gravity(float(cfg["data"]["gravity"]))
    g_vec = torch.zeros(3, dtype=train[0].q.dtype)
    g_vec[_AXIS[cfg["data"]["gravity_axis"]]] = -g_mag

    particles = Particles.uniform(
        train[0].q.shape[1],
        m=train[0].m[0].item(),
        radius=train[0].radii[0].item(),
        dtype=train[0].q.dtype,
    )
    return scales, train, val, wall, g_vec, particles


def compute_sigmas(train, dt):
    """Escalas de normalización de las pérdidas, derivadas de los datos (§34, §36).

    sigma_a / sigma_alpha desde las aceleraciones por diferencias finitas;
    sigma_v / sigma_w desde velocidades y velocidades angulares; sigma_q en
    una escala natural de un diámetro.
    """
    a = torch.cat([finite_difference_accelerations(tr.v, tr.dt).reshape(-1) for tr in train])
    al = torch.cat([finite_difference_accelerations(tr.omega, tr.dt).reshape(-1) for tr in train])
    v = torch.cat([tr.v.reshape(-1) for tr in train])
    w = torch.cat([tr.omega.reshape(-1) for tr in train])
    eps = 1e-6
    return {
        "sigma_a": float(a.std()) + eps,
        "sigma_alpha": float(al.std()) + eps,
        "sigma_q": 1.0,
        "sigma_v": float(v.std()) + eps,
        "sigma_w": float(w.std()) + eps,
    }


# --------------------------------------------------------------------------- #
# Pérdidas de una ventana de rollout (accede a diagnósticos por paso)
# --------------------------------------------------------------------------- #
def rollout_window_loss(model, tr, k0, horizon, particles, wall, g_vec, dt,
                        sig, cfg, opt):
    """Rueda `horizon` pasos desde k0 (+ruido) y acumula todas las pérdidas.

    Hace backprop truncado (TBPTT): backward por bloques de `tbptt_chunk`
    pasos, desconectando el estado entre bloques para acotar la memoria del
    grafo de autograd del doble backward. Devuelve el valor escalar total.
    """
    lcfg = cfg["loss"]
    K = int(cfg["curriculum"]["tbptt_chunk"])
    decay = float(lcfg["rollout_time_decay"])
    radii = particles.radii.to(tr.q.dtype)

    # estado inicial con ruido coherente (§34: el modelo verá sus propios errores)
    q = tr.q[k0] + cfg["noise"]["sigma_q"] * torch.randn_like(tr.q[k0])
    v = tr.v[k0] + cfg["noise"]["sigma_v"] * torch.randn_like(tr.v[k0])
    w = tr.omega[k0].clone()

    total = 0.0
    chunk = torch.zeros((), dtype=tr.q.dtype)
    for s in range(horizon):
        q, v, w, out = semi_implicit_step(
            model, q, v, w, particles, wall, t=0.0, dt=dt, g_vec=g_vec
        )
        d = out.diagnostics
        weight = decay ** s

        qr, vr, wr = tr.q[k0 + s + 1], tr.v[k0 + s + 1], tr.omega[k0 + s + 1]
        step = weight * (
            ((q - qr) / sig["sigma_q"]).pow(2).sum(-1).mean()
            + lcfg["lambda_v"] * ((v - vr) / sig["sigma_v"]).pow(2).sum(-1).mean()
            + lcfg["lambda_w"] * ((w - wr) / sig["sigma_w"]).pow(2).sum(-1).mean()
        )
        # pérdidas auxiliares por paso (todas las herramientas)
        step = step + lcfg["lambda_res"] * residual_regularization(d["f_H"], d["tau_H"])
        step = step + lcfg["lambda_pass"] * passivity_loss(d["P_hist_pp"], d["P_hist_pW"])
        step = step + lcfg["lambda_pen"] * penetration_loss(d["phi"], radii, model.cfg.beta)
        # aceleración a un paso solo en el primer paso (estado ≈ dato)
        if s == 0:
            a_ref = (tr.v[k0 + 1] - tr.v[k0]) / dt
            al_ref = (tr.omega[k0 + 1] - tr.omega[k0]) / dt
            step = step + acceleration_loss(out.a, a_ref, sig["sigma_a"])
            step = step + angular_acceleration_loss(out.alpha, al_ref, sig["sigma_alpha"])

        chunk = chunk + step
        if (s + 1) % K == 0 or s == horizon - 1:
            chunk.backward()
            total += float(chunk.detach())
            q, v, w = q.detach(), v.detach(), w.detach()
            chunk = torch.zeros((), dtype=tr.q.dtype)
    return total


# --------------------------------------------------------------------------- #
# Validación (inferencia, sin grafo)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def validate(model, tr, horizon, particles, wall, g_vec, dt, sig):
    q, v, w = tr.q[0], tr.v[0], tr.omega[0]
    radii = particles.radii.to(tr.q.dtype)
    se_q = se_v = 0.0
    max_pen = 0.0
    for s in range(horizon):
        q, v, w, out = semi_implicit_step(
            model, q, v, w, particles, wall, t=0.0, dt=dt, g_vec=g_vec
        )
        se_q += (q - tr.q[s + 1]).pow(2).sum(-1).mean().item()
        se_v += (v - tr.v[s + 1]).pow(2).sum(-1).mean().item()
        pen = (radii - out.diagnostics["phi"]).clamp(min=0).max().item()
        max_pen = max(max_pen, pen)
    rmse_q = (se_q / horizon) ** 0.5
    rmse_v = (se_v / horizon) ** 0.5
    # baseline "quieto": error de no moverse
    base_q = (tr.q[0].unsqueeze(0) - tr.q[1 : horizon + 1]).pow(2).sum(-1).mean().sqrt().item()
    return {"rmse_q": rmse_q, "rmse_v": rmse_v, "max_pen": max_pen, "base_q": base_q}


# --------------------------------------------------------------------------- #
# Bucle principal
# --------------------------------------------------------------------------- #
def build_model(cfg):
    # float32: el doble backward funciona igual y es ~2x más rápido/liviano que
    # float64 en CPU, lo importante para rollouts largos. Las garantías físicas
    # que exigen float64 se verifican aparte en los tests.
    fields = SLGNNConfig().__dict__
    overrides = {k: v for k, v in (cfg.get("model") or {}).items() if k in fields}
    return SLGNN(SLGNNConfig(**overrides))


def save_checkpoint(path, model, opt, cfg, sig, tag):
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optim": opt.state_dict(),
            "config": cfg,
            "model_config": asdict_config(model.cfg),
            "sigmas": sig,
            "tag": tag,
        },
        path / f"{tag}.pt",
    )


def asdict_config(c: SLGNNConfig):
    return dict(c.__dict__)


def set_lr(opt, lr):
    for g in opt.param_groups:
        g["lr"] = lr


def train(cfg, smoke=False):
    torch.manual_seed(cfg["seed"])
    scales, train_set, val, wall, g_vec, particles = load_split(cfg)
    dt = train_set[0].dt
    sig = compute_sigmas(train_set, dt)
    n = particles.radii.shape[0]
    print(f"train cases: {len(train_set)}  N={n}  dt(adim)={dt:.4f}")
    print("sigmas:", {k: round(v, 4) for k, v in sig.items()})
    print("g_vec (adim):", g_vec.tolist())

    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parámetros: {n_params}")

    ckpt_dir = ROOT / cfg["log"]["ckpt_dir"]
    grad_clip = float(cfg["optim"]["grad_clip"])
    wd = float(cfg["optim"]["weight_decay"])
    log_every = cfg["log"]["every"]
    val_every = cfg["log"]["val_every"]
    val_h = cfg["log"]["val_horizon"]
    wpi = int(cfg["curriculum"]["windows_per_iter"])

    warmup = int(cfg["curriculum"]["warmup_iters"])
    horizons = list(cfg["curriculum"]["horizons"])
    iters_per_h = int(cfg["curriculum"]["iters_per_horizon"])
    if smoke:
        warmup, horizons, iters_per_h = 10, [4, 8], 10
        val_every, val_h = 10, 20

    best_val = float("inf")
    t_start = time.time()

    # ---- Fase de warmup: aceleración a un paso ---------------------------- #
    opt = torch.optim.Adam(
        model.parameters(), lr=float(cfg["optim"]["lr_warmup"]), weight_decay=wd
    )
    print(f"\n== warmup: {warmup} iters (L_a + L_alpha a un paso) ==")
    dtype = train_set[0].q.dtype
    for it in range(warmup):
        opt.zero_grad()
        loss = torch.zeros((), dtype=dtype)
        for _ in range(wpi):
            tr = train_set[torch.randint(len(train_set), (1,)).item()]
            k = torch.randint(tr.q.shape[0] - 1, (1,)).item()
            out = model(tr.q[k], tr.v[k], tr.omega[k], particles, wall=wall, g_vec=g_vec)
            a_ref = (tr.v[k + 1] - tr.v[k]) / dt
            al_ref = (tr.omega[k + 1] - tr.omega[k]) / dt
            loss = loss + acceleration_loss(out.a, a_ref, sig["sigma_a"])
            loss = loss + angular_acceleration_loss(out.alpha, al_ref, sig["sigma_alpha"])
        loss = loss / wpi
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        if it % log_every == 0 or it == warmup - 1:
            print(f"  warmup {it:4d}  loss {loss.item():.4f}")
    save_checkpoint(ckpt_dir, model, opt, cfg, sig, "warmup")

    # ---- Fases de rollout de horizonte creciente ------------------------- #
    set_lr(opt, float(cfg["optim"]["lr_rollout"]))
    for horizon in horizons:
        print(f"\n== rollout horizon={horizon}: {iters_per_h} iters ==")
        for it in range(iters_per_h):
            opt.zero_grad()
            total = 0.0
            for _ in range(wpi):
                tr = train_set[torch.randint(len(train_set), (1,)).item()]
                k0 = torch.randint(tr.q.shape[0] - horizon - 1, (1,)).item()
                total += rollout_window_loss(
                    model, tr, k0, horizon, particles, wall, g_vec, dt, sig, cfg, opt
                )
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            if it % log_every == 0 or it == iters_per_h - 1:
                print(f"  H={horizon} it {it:4d}  loss {total / wpi:.4f}")
            if it % val_every == 0 or it == iters_per_h - 1:
                m = validate(model, val, val_h, particles, wall, g_vec, dt, sig)
                print(
                    f"    [val H={val_h}] RMSE_q {m['rmse_q']:.4f} "
                    f"(base {m['base_q']:.4f})  RMSE_v {m['rmse_v']:.4f}  "
                    f"pen_max {m['max_pen']:.4f}"
                )
                if m["rmse_q"] < best_val:
                    best_val = m["rmse_q"]
                    save_checkpoint(ckpt_dir, model, opt, cfg, sig, "best")
        save_checkpoint(ckpt_dir, model, opt, cfg, sig, f"horizon_{horizon}")

    print(f"\ntiempo total: {time.time() - t_start:.1f}s   mejor RMSE_q val: {best_val:.4f}")
    save_checkpoint(ckpt_dir, model, opt, cfg, sig, "final")
    (ckpt_dir / "sigmas.json").write_text(json.dumps(sig, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gravity_rollout.yaml")
    ap.add_argument("--smoke", action="store_true",
                    help="corrida mínima para verificar que todo encadena")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    train(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()
