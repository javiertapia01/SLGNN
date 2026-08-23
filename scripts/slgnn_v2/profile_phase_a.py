"""CPU/GPU hot-path profile for the Phase A checkpoint audit."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from slgnn.experiment import load_checkpoint, load_split
from slgnn.graph import neighbor_pairs, wall_geometry
from slgnn.integrator import rollout

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import rollout_window_loss, validate  # noqa: E402


def summary_ms(samples):
    x = np.asarray(samples, dtype=np.float64) * 1000
    return {
        "count": int(x.size), "median_ms": float(np.median(x)),
        "p90_ms": float(np.quantile(x, 0.9)), "max_ms": float(x.max()),
    }


def measure(fn, warmups, repeats):
    for _ in range(warmups):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return summary_ms(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    model, ck = load_checkpoint(Path(args.checkpoint))
    cfg = ck["config"]
    _, train, val, wall, g_vec, particles = load_split(cfg, ROOT)
    tr = val
    q, v, w = tr.q[50], tr.v[50], tr.omega[50]
    warm_short, rep_short = (1, 2) if args.quick else (20, 50)
    warm_long, rep_long = (0, 1) if args.quick else (1, 3)
    inference_h = [2] if args.quick else [16, 64, 200]
    training_h = [2] if args.quick else [8, 16, 64]
    result = {
        "device": "cpu", "torch_version": torch.__version__,
        "cpu": platform.processor(), "cpu_count_logical": __import__("os").cpu_count(),
        "protocol": {
            "short_warmups": warm_short, "short_repeats": rep_short,
            "long_warmups": warm_long, "long_repeats": rep_long,
            "cuda_synchronization": "not_applicable",
        },
        "benchmarks": {}, "operations": {},
    }

    def forward():
        with torch.no_grad():
            model(q, v, w, particles, wall=wall, g_vec=g_vec)

    def forward_backward():
        model.zero_grad(set_to_none=True)
        out = model(q, v, w, particles, wall=wall, g_vec=g_vec)
        (out.a.square().mean() + out.alpha.square().mean()).backward()

    result["benchmarks"]["forward_one_step"] = measure(
        forward, warm_short, rep_short
    )
    result["benchmarks"]["forward_backward_one_step"] = measure(
        forward_backward, warm_short, rep_short
    )

    def cdist_nonzero():
        neighbor_pairs(q, model.cfg.r_list)

    def cdist_only():
        torch.cdist(q, q)

    def nonzero_only():
        d = torch.cdist(q, q)
        ((d < model.cfg.r_list).triu(diagonal=1)).nonzero()

    def sdf_query():
        wall_geometry(q, v, w, particles.radii, wall, 0.0, model.cfg)

    for name, fn in (
        ("cdist_plus_nonzero", cdist_nonzero), ("cdist", cdist_only),
        ("nonzero_with_precomputed_like_mask", nonzero_only), ("sdf_query", sdf_query),
    ):
        result["operations"][name] = measure(fn, warm_short, rep_short)

    hook_samples = {"processor_V": [], "processor_R": [], "processor_H": []}
    starts = {}
    handles = []
    for name, module in (("processor_V", model.proc_V), ("processor_R", model.proc_R),
                         ("processor_H", model.proc_H)):
        handles.append(module.register_forward_pre_hook(
            lambda m, inp, n=name: starts.__setitem__(n, time.perf_counter())
        ))
        handles.append(module.register_forward_hook(
            lambda m, inp, out, n=name: hook_samples[n].append(
                time.perf_counter() - starts[n]
            )
        ))
    for _ in range(warm_short + rep_short):
        forward()
    for h in handles:
        h.remove()
    for name, values in hook_samples.items():
        result["operations"][name] = summary_ms(values[-rep_short:])

    for horizon in inference_h:
        def infer_rollout(h=horizon):
            rollout(model, q, v, w, particles, wall, tr.dt, h, g_vec=g_vec)
        result["benchmarks"][f"rollout_inference_H{horizon}"] = measure(
            infer_rollout, warm_long, rep_long
        )

    for horizon in training_h:
        def training_rollout(h=horizon):
            model.zero_grad(set_to_none=True)
            rollout_window_loss(
                model, train[0], 50, h, particles, wall, g_vec, tr.dt,
                ck["sigmas"], cfg, None,
            )
        result["benchmarks"][f"rollout_training_H{horizon}"] = measure(
            training_rollout, warm_long, rep_long
        )

    val_h = 2 if args.quick else 100
    result["benchmarks"][f"validation_CASE06_H{val_h}"] = measure(
        lambda: validate(model, val, val_h, particles, wall, g_vec, tr.dt, ck["sigmas"]),
        warm_long, rep_long,
    )

    # One representative outer backward trace. PyTorch operator names expose
    # cdist, nonzero, SDF reductions, and both nested/autograd backward paths.
    activities = [torch.profiler.ProfilerActivity.CPU]
    with torch.profiler.profile(
        activities=activities, record_shapes=True, profile_memory=True,
        with_stack=True,
    ) as prof:
        with torch.profiler.record_function("phase_a_forward_backward"):
            forward_backward()
    prof.export_chrome_trace(str(out_dir / "trace.json"))
    events = []
    for event in prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=100).splitlines():
        events.append(event)
    result["profiler_table"] = events
    result["interpretation_note"] = (
        "Processor timings use module hooks. Gradient-of-V and gradient-of-Rayleigh "
        "share nested autograd operator names and are therefore reported in the "
        "Chrome trace rather than attributed by an unreliable wall-clock split."
    )
    (out_dir / "cpu_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    cuda = {
        "status": "available" if torch.cuda.is_available() else "not_available",
        "torch_cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        cuda.update({
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
            "note": "CUDA exists, but this checkpoint audit intentionally loaded CPU tensors; "
                    "a device-complete GPU port is required before valid GPU timings.",
        })
    (out_dir / "cuda_summary.json").write_text(
        json.dumps(cuda, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result["benchmarks"], indent=2))


if __name__ == "__main__":
    main()
