"""Nivel 1: micro-overfit y auditoría de cableado (§16.3).

No mide generalización. Comprueba que la señal llega a donde debe:

1. la pérdida primaria cae de forma marcada sobre un conjunto fijo y pequeño;
2. los gradientes llegan a `V`, `Psi` e `I` **cuando corresponde**;
3. `e` y `kappa` se mueven desde su inicialización en ejemplos impulsivos;
4. ninguna cabeza recibe gradiente en el régimen donde está desactivada;
5. cada incremento es atribuible desde los diagnósticos.

Si el residual no baja del objetivo del 90-95 %, se reporta el residual y se
explica qué restricción física lo causa, en vez de aflojar la restricción.

    python scripts/v3/run_micro_overfit.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _common import REPO_ROOT, build_model, load_yaml, prepare  # noqa: E402
from slgnn_experiments.runner import _batch_loss, train  # noqa: E402

# La misma rejilla y el mismo presupuesto que en `compare.py`. El micro-overfit
# audita cableado, pero un `lr` inadecuado para una familia produce una
# "reducción negativa" que se leería como fallo de conexión cuando en realidad
# es divergencia del optimizador. Ver D-021.
LR_GRID = (3e-3, 1e-3, 3e-4, 1e-4)

HEAD_GROUPS = {
    "V": ("head_V", "proc_V"),
    "Psi": ("head_Psi", "proc_Psi"),
    "I": ("head_I", "proc_I"),
    "encoder": ("encoder",),
}


def grad_by_head(model) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, prefixes in HEAD_GROUPS.items():
        total = 0.0
        for pname, p in model.named_parameters():
            if p.grad is not None and pname.startswith(prefixes):
                total += float(p.grad.abs().sum())
        out[name] = total
    return out


def fixed_picks(data, n: int = 8) -> list[tuple[int, int]]:
    """Batch FIJO para medir antes/después.

    Comparar la pérdida de un batch aleatorio con la de otro batch aleatorio
    mide sobre todo la varianza del muestreo: con 60 partículas y unos pocos
    contactos por frame, dos batches distintos difieren más entre sí que el
    efecto del entrenamiento.
    """
    stride = max(1, len(data.index) // n)
    return data.index.items[::stride][:n]


def probe(model, scene, data, tcfg, picks) -> dict:
    """Un forward+backward sobre el batch fijo, para ver a qué cabezas llega
    el gradiente y cuánto vale la pérdida."""
    model.train()
    model.zero_grad(set_to_none=True)
    loss, parts, res = _batch_loss(model, scene, data, picks, tcfg, True)
    loss.backward()
    d = res.diagnostics
    impact = getattr(d, "impact", {})
    return {
        "loss": parts,
        "grad_by_head": grad_by_head(model),
        "router": getattr(d, "router", {}),
        "e_mean": impact.get("e_mean"),
        "kappa_mean": impact.get("kappa_mean"),
        "n_impulsive": impact.get("n_contacts", 0),
        "result": res,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", type=Path,
                    default=REPO_ROOT / "configs/experiments/micro_overfit.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "results/v3_mvp/micro_overfit.json")
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    experiment = load_yaml(args.experiment)
    variants = {
        "v3_c": ("slgnn_v3", "configs/v3/mvp_c.yaml"),
        "v3_i": ("slgnn_v3", "configs/v3/mvp_i.yaml"),
        "gns_controlled": ("gns_controlled", "configs/gns/controlled.yaml"),
    }

    report: dict = {"experiment": experiment.get("name"), "seed": args.seed,
                    "variants": {}}
    for variant, (model_key, cfg_path) in variants.items():
        print(f"\n=== {variant} ===", flush=True)
        (scales, split, loaded, scene, train_ds, val_ds, test_ds,
         sampler, tcfg) = prepare(experiment, args.seed)
        torch.manual_seed(args.seed)
        model, resolved, profile = build_model(
            model_key, load_yaml(REPO_ROOT / cfg_path).get("model")
        )
        picks = fixed_picks(train_ds)

        # Selección de lr por validación, presupuesto idéntico por familia.
        lr_scores: dict[str, float] = {}
        best_lr, best_score = LR_GRID[0], float("inf")
        for lr in LR_GRID:
            (_, _, _, sc2, tr2, va2, _, sam2, tc2) = prepare(experiment, args.seed)
            tc2.lr = lr
            torch.manual_seed(args.seed)
            probe_model, _, _ = build_model(
                model_key, load_yaml(REPO_ROOT / cfg_path).get("model")
            )
            score = train(probe_model, sc2, tr2, va2, sam2, tc2)["best_val_loss"]
            lr_scores[f"lr={lr:g}"] = score
            if score is not None and score < best_score:
                best_lr, best_score = lr, score
        tcfg.lr = best_lr
        print(f"  lr elegido: {best_lr:g}  (rejilla: {lr_scores})", flush=True)

        torch.manual_seed(args.seed)
        model, resolved, profile = build_model(
            model_key, load_yaml(REPO_ROOT / cfg_path).get("model")
        )
        before = probe(model, scene, train_ds, tcfg, picks)
        summary = train(model, scene, train_ds, val_ds, sampler, tcfg)
        after = probe(model, scene, train_ds, tcfg, picks)

        first = before["loss"]["total"]
        last = after["loss"]["total"]
        reduction = 1.0 - last / first if first else float("nan")

        entry = {
            "profile": profile,
            "selected_lr": best_lr,
            "lr_grid": lr_scores,
            "n_parameters": model.n_parameters()[0],
            "loss_first": first, "loss_last": last,
            "reduction_fraction": reduction,
            "best_val_loss": summary["best_val_loss"],
            "grad_before": before["grad_by_head"],
            "grad_after": after["grad_by_head"],
            "router_before": before["router"], "router_after": after["router"],
            "e_before": before["e_mean"], "e_after": after["e_mean"],
            "kappa_before": before["kappa_mean"], "kappa_after": after["kappa_mean"],
            "n_impulsive": after["n_impulsive"],
            "train_seconds": summary["train_seconds"],
        }
        entry["reachability"] = _reachability(scene, train_ds, picks)
        entry["wiring_checks"] = _wiring_checks(variant, entry)
        report["variants"][variant] = entry
        print(json.dumps({k: entry[k] for k in
                          ("loss_first", "loss_last", "reduction_fraction",
                           "grad_after", "e_after", "kappa_after",
                           "reachability", "wiring_checks")},
                         indent=2, default=float))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\n[micro-overfit] {args.out}")
    return 0


def _reachability(scene, data, picks) -> dict:
    """Cuánto de `Delta p` (descontada la gravedad) puede alcanzar un canal
    puramente normal. Cota independiente del modelo."""
    import torch

    from slgnn_experiments.metrics import reachable_decomposition
    from slgnn_v3.contact_kinematics import build_contacts
    from slgnn_v3.graph import build_candidate_graph
    from slgnn_v3 import V3Config

    cfg = V3Config()
    acc = {"reachable_fraction": 0.0, "unreachable_fraction": 0.0}
    n = 0
    for t_idx, k in picks:
        tr = data.trajectories[t_idx]
        st = scene.state_at(tr, k)
        pb = st.particles
        wall = scene.surfaces.query(pb.q, pb.radius, pb.batch_id,
                                    st.time_scalar(), cfg.graph.pw_gap_off)
        edges = build_candidate_graph(pb, cfg.graph.pp_gap_off, cfg.graph.skin)
        cs = build_contacts(pb, edges, wall, cfg)
        if cs.n_contacts == 0:
            continue
        target = data.targets[t_idx].delta_p[k].clone()
        if scene.gravity is not None:
            target = target - scene.dt * pb.mass.unsqueeze(-1) * scene.gravity
        # solo los contactos realmente activos pueden ejercer fuerza
        active = cs.gap <= 0
        if not bool(active.any()):
            continue
        sub = cs.subset(active)
        d = reachable_decomposition(target, sub.n[sub.inc_contact],
                                    sub.inc_node, pb.n)
        acc["reachable_fraction"] += d["reachable_fraction"]
        acc["unreachable_fraction"] += d["unreachable_fraction"]
        n += 1
    if n == 0:
        return {"note": "sin contactos activos en el batch fijo"}
    return {k: v / n for k, v in acc.items()} | {"n_batches": n}


def _wiring_checks(variant: str, e: dict) -> dict:
    g = e["grad_after"]
    checks: dict[str, bool | str] = {
        "loss_decreases": bool(e["reduction_fraction"] > 0.0),
        "loss_drops_over_90pct": bool(e["reduction_fraction"] >= 0.90),
        "encoder_receives_gradient": g.get("encoder", 0.0) > 0,
    }
    if variant == "v3_c":
        checks["V_receives_gradient"] = g.get("V", 0.0) > 0
        checks["Psi_receives_gradient"] = g.get("Psi", 0.0) > 0
        # I está desactivada en v3-C: no debe recibir gradiente
        checks["I_silent_as_expected"] = g.get("I", 0.0) == 0.0
    elif variant == "v3_i":
        checks["I_receives_gradient"] = g.get("I", 0.0) > 0
        checks["V_silent_as_expected"] = g.get("V", 0.0) == 0.0
        checks["Psi_silent_as_expected"] = g.get("Psi", 0.0) == 0.0
        if e["e_after"] is not None and e["e_before"] is not None:
            checks["e_moved_from_init"] = abs(e["e_after"] - e["e_before"]) > 1e-6
        if e["kappa_after"] is not None and e["kappa_before"] is not None:
            checks["kappa_moved_from_init"] = (
                abs(e["kappa_after"] - e["kappa_before"]) > 1e-9
            )
    return checks


if __name__ == "__main__":
    raise SystemExit(main())
