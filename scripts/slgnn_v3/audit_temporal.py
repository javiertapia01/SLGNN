"""Auditoría temporal del dataset, previa a cualquier entrenamiento (§14).

Responde con datos —no con supuestos heredados— a la pregunta que decide la
arquitectura: ¿el intervalo entre snapshots resuelve la duración del contacto,
o la contiene entera?

Produce `results/slgnn_v3/audit_temporal/{audit.json, audit_summary.csv}` y
`docs/slgnn_v3/DATA_AUDIT.md`.

    python scripts/slgnn_v3/audit_temporal.py --config configs/experiments/audit_temporal.yaml

Limitación declarada: los CSV de Dynami-CAL **no** contienen fuerzas de
contacto, impulsos ni los subpasos internos del solver (dt_DEM = 1e-7 s). Todo
lo que sigue se infiere de geometría y cinemática de los snapshots grabados.
La recomendación de perfil es por tanto una hipótesis fundamentada, no una
observación directa del régimen.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from slgnn_experiments.data import DATASETS, Trajectory, list_cases, load_case  # noqa: E402

# Caras de una caja axialmente alineada: (eje, signo). El gap a la cara es
# +(q[ax] - lo) para signo -1 y +(hi - q[ax]) para signo +1, menos el radio.
BOX_FACE_NAMES = ("-x", "+x", "-y", "+y", "-z", "+z")

# Rigidez normal documentada del DEM (data/DATA_NOTES.md §3 y §4). Se usa solo
# para el contraste analítico de la sección 2b del informe: si la duración de
# episodio medida geométricamente coincide con el semiperiodo del oscilador
# masa-resorte, la detección está midiendo el contacto real y no ruido.
DEM_STIFFNESS_N_PER_M = {
    "two_spheres": 1000.0,
    "one_sphere_wall": 1000.0,
    "sixty_homogeneous": 1000.0,
    "sixty_gravity": 10000.0,
    "rotating_cylinder": 10000.0,
}


def analytic_contact_snapshots(tr: Trajectory, kind: str) -> float | None:
    """Duración de contacto de un resorte lineal sin amortiguar, en snapshots.

    `t_c = pi sqrt(m_eff / k_n)`, con `m_eff = m/2` para partícula–partícula y
    `m_eff = m` para partícula–pared (la pared es infinitamente masiva).
    """
    k_n = DEM_STIFFNESS_N_PER_M.get(tr.dataset_key)
    if k_n is None:
        return None
    m = float(tr.mass.mean())
    m_eff = m / 2.0 if kind == "pp" else m
    return float(np.pi * np.sqrt(m_eff / k_n) / tr.dt)


@dataclass
class ContactEpisode:
    key: tuple
    kind: str          # "pp" | "pw"
    start: int
    end: int           # inclusive
    max_penetration: float
    argmax_offset: int  # posición del máximo dentro del episodio

    @property
    def length(self) -> int:
        return self.end - self.start + 1


# --------------------------------------------------------------------------
# Geometría por snapshot
# --------------------------------------------------------------------------

def pair_gaps(q: torch.Tensor, radius: torch.Tensor, band: float):
    """Pares (i<j) con gap <= band. Devuelve (idx [E,2], gap [E])."""
    d = torch.cdist(q, q)
    rsum = radius.unsqueeze(0) + radius.unsqueeze(1)
    g = d - rsum
    mask = torch.triu(g <= band, diagonal=1)
    idx = mask.nonzero()
    return idx, g[mask]


def wall_face_gaps(q: torch.Tensor, radius: torch.Tensor, box_min, box_max):
    """Gap a cada una de las 6 caras de la caja. Devuelve [N, 6].

    Multi-superficie desde el principio: una partícula en una esquina tiene
    tres gaps pequeños simultáneos, cosa que un `min` sobre caras esconde.
    """
    lo = torch.as_tensor(box_min, dtype=q.dtype)
    hi = torch.as_tensor(box_max, dtype=q.dtype)
    # orden: -x, +x, -y, +y, -z, +z
    per_axis_lo = q - lo            # distancia a la cara inferior de cada eje
    per_axis_hi = hi - q
    phi = torch.stack(
        [per_axis_lo[:, 0], per_axis_hi[:, 0],
         per_axis_lo[:, 1], per_axis_hi[:, 1],
         per_axis_lo[:, 2], per_axis_hi[:, 2]], dim=-1
    )
    return phi - radius.unsqueeze(-1)


# --------------------------------------------------------------------------
# Verificaciones empíricas de dt y gravedad
# --------------------------------------------------------------------------

def estimate_dt(tr: Trajectory, free_mask: torch.Tensor | None = None) -> dict:
    """Estima `dt` desde la propia trayectoria: q_{k+1}-q_k ~ dt*(v_k+v_{k+1})/2.

    Mínimos cuadrados sobre todas las componentes; si `free_mask` marca
    partículas en vuelo libre, se restringe a ellas (la regla del punto medio
    es exacta para movimiento uniformemente acelerado y se degrada durante un
    contacto).
    """
    dq = tr.q[1:] - tr.q[:-1]
    vm = 0.5 * (tr.v[1:] + tr.v[:-1])
    if free_mask is not None:
        dq, vm = dq[free_mask], vm[free_mask]
    num = float((dq * vm).sum())
    den = float((vm * vm).sum())
    dt_hat = num / den if den > 0 else float("nan")
    resid = dq - dt_hat * vm
    rel = float(resid.norm() / (dq.norm() + 1e-30))
    return {
        "dt_documented": tr.dt,
        "dt_estimated": dt_hat,
        "dt_relative_error": abs(dt_hat - tr.dt) / tr.dt if tr.dt else float("nan"),
        "midpoint_rule_residual_relative": rel,
        "n_samples": int(dq.numel() // 3),
    }


def estimate_gravity(tr: Trajectory, free_mask: torch.Tensor) -> dict:
    """Aceleración media de las partículas en vuelo libre: debe ser `g`."""
    a = (tr.v[1:] - tr.v[:-1]) / tr.dt          # [T-1, N, 3]
    if free_mask.sum() < 10:
        return {"available": False, "reason": "menos de 10 transiciones libres"}
    a_free = a[free_mask]                        # [K, 3]
    mean = a_free.mean(dim=0)
    return {
        "available": True,
        "mean_acceleration": [float(x) for x in mean],
        "std_acceleration": [float(x) for x in a_free.std(dim=0)],
        "magnitude": float(mean.norm()),
        "dominant_axis": "xyz"[int(mean.abs().argmax())],
        "n_free_transitions": int(free_mask.sum()),
    }


# --------------------------------------------------------------------------
# Auditoría de un caso
# --------------------------------------------------------------------------

def audit_case(tr: Trajectory, spec, band_factor: float) -> dict:
    T, N = tr.n_steps, tr.n_particles
    band = band_factor * float(tr.radius.mean()) * 2.0
    has_box = spec.geometry == "box" and spec.box_min is not None

    # --- ocupación de contacto por frame ---------------------------------
    in_contact = torch.zeros(T, N, dtype=torch.bool)     # particula toca algo
    pp_active: list[set[tuple[int, int]]] = []
    pw_active: list[set[tuple[int, int]]] = []
    pp_pen: list[dict] = []
    pw_pen: list[dict] = []
    n_faces_hist = defaultdict(int)
    all_pp_gaps: list[np.ndarray] = []
    all_pw_gaps: list[np.ndarray] = []

    for k in range(T):
        q, r = tr.q[k], tr.radius
        idx, g = pair_gaps(q, r, band)
        touching = g <= 0.0
        pp_set = {(int(a), int(b)) for a, b in idx[touching].tolist()}
        pp_active.append(pp_set)
        pp_pen.append({key: float(-gv) for key, gv in
                       zip(pp_set, g[touching].tolist())} if pp_set else {})
        if idx.numel():
            all_pp_gaps.append(g.numpy())
        for a, b in pp_set:
            in_contact[k, a] = True
            in_contact[k, b] = True

        if has_box:
            gw = wall_face_gaps(q, r, spec.box_min, spec.box_max)     # [N,6]
            all_pw_gaps.append(gw.min(dim=-1).values.numpy())
            hit = gw <= 0.0
            pw_set = {(int(i), int(f)) for i, f in hit.nonzero().tolist()}
            pw_active.append(pw_set)
            pw_pen.append({key: float(-gw[key[0], key[1]]) for key in pw_set})
            per_particle = hit.sum(dim=-1)
            for c in per_particle.tolist():
                n_faces_hist[int(c)] += 1
            for i, _f in pw_set:
                in_contact[k, i] = True
        else:
            pw_active.append(set())
            pw_pen.append({})

    # --- episodios por clave estable -------------------------------------
    episodes = _episodes(pp_active, pp_pen, "pp") + _episodes(pw_active, pw_pen, "pw")
    ep_pp = [e for e in episodes if e.kind == "pp"]
    ep_pw = [e for e in episodes if e.kind == "pw"]

    # --- dt y gravedad sobre transiciones libres --------------------------
    free_trans = (~in_contact[:-1]) & (~in_contact[1:])
    dt_info = estimate_dt(tr, free_trans if int(free_trans.sum()) > 50 else None)
    grav_info = estimate_gravity(tr, free_trans)

    # --- saltos de velocidad condicionados a contacto ---------------------
    dv = (tr.v[1:] - tr.v[:-1]).norm(dim=-1)      # [T-1, N]
    contact_trans = in_contact[:-1] | in_contact[1:]
    jumps = {
        "free_mean": _safe(dv[~contact_trans].mean()),
        "free_p99": _safe(torch.quantile(dv[~contact_trans].flatten(), 0.99)
                          if int((~contact_trans).sum()) else torch.tensor(float("nan"))),
        "contact_mean": _safe(dv[contact_trans].mean()),
        "contact_p99": _safe(torch.quantile(dv[contact_trans].flatten(), 0.99)
                             if int(contact_trans.sum()) else torch.tensor(float("nan"))),
        "contact_max": _safe(dv[contact_trans].max() if int(contact_trans.sum()) else
                             torch.tensor(float("nan"))),
        "ratio_contact_over_free": None,
    }
    if jumps["free_mean"] and jumps["free_mean"] > 0 and jumps["contact_mean"]:
        jumps["ratio_contact_over_free"] = jumps["contact_mean"] / jumps["free_mean"]

    # --- nacimiento / muerte entre snapshots ------------------------------
    births = deaths = persist = 0
    for k in range(1, T):
        prev = pp_active[k - 1] | {("W",) + key for key in pw_active[k - 1]}
        cur = pp_active[k] | {("W",) + key for key in pw_active[k]}
        births += len(cur - prev)
        deaths += len(prev - cur)
        persist += len(cur & prev)

    lengths = [e.length for e in episodes]
    single = sum(1 for L in lengths if L == 1)
    # máximo de compresión resuelto: episodios de >=3 frames cuyo pico de
    # penetración cae estrictamente dentro del episodio.
    long_eps = [e for e in episodes if e.length >= 3]
    interior_peak = sum(1 for e in long_eps if 0 < e.argmax_offset < e.length - 1)

    median_len = statistics.median(lengths) if lengths else 0.0
    t_contact = median_len * tr.dt if lengths else float("nan")
    rho_t = tr.dt / t_contact if t_contact and t_contact > 0 else float("inf")

    return {
        "case": tr.name,
        "dataset": tr.dataset_key,
        "schema_variant": tr.schema_variant,
        "n_snapshots": T,
        "n_particles": N,
        "duration_seconds": (T - 1) * tr.dt,
        "radius_m": float(tr.radius.mean()),
        "mass_kg": float(tr.mass.mean()),
        "inertia_kg_m2": float(tr.inertia.mean()),
        "band_m": band,
        "timing": dt_info,
        "gravity": grav_info,
        "gaps": {
            "pp": _dist_stats(all_pp_gaps),
            "pw_min_over_faces": _dist_stats(all_pw_gaps),
        },
        "episodes": {
            "n_total": len(episodes),
            "n_pp": len(ep_pp),
            "n_pw": len(ep_pw),
            "length_snapshots": {
                "min": min(lengths) if lengths else None,
                "median": median_len,
                "mean": statistics.fmean(lengths) if lengths else None,
                "p90": _pct(lengths, 90),
                "max": max(lengths) if lengths else None,
            },
            "median_duration_s": t_contact,
            "fraction_single_snapshot": single / len(lengths) if lengths else None,
            "pp_median_length": statistics.median([e.length for e in ep_pp]) if ep_pp else None,
            "pw_median_length": statistics.median([e.length for e in ep_pw]) if ep_pw else None,
            "n_with_3plus_snapshots": len(long_eps),
            "fraction_interior_compression_peak":
                interior_peak / len(long_eps) if long_eps else None,
            "max_penetration_over_radius":
                max((e.max_penetration for e in episodes), default=0.0) / float(tr.radius.mean()),
        },
        "transitions": {
            "births": births,
            "deaths": deaths,
            "persisting": persist,
            "fraction_births_over_active": births / max(births + persist, 1),
        },
        "rho_t": rho_t,
        "analytic_contact_snapshots": {
            "pp": analytic_contact_snapshots(tr, "pp"),
            "pw": analytic_contact_snapshots(tr, "pw"),
            "k_n_documented": DEM_STIFFNESS_N_PER_M.get(tr.dataset_key),
        },
        "velocity_jumps": jumps,
        "wall_multiplicity": {str(k): v for k, v in sorted(n_faces_hist.items())},
        "contact_occupancy": {
            "fraction_particle_frames_in_contact": float(in_contact.float().mean()),
            "fraction_transitions_free": float(free_trans.float().mean()),
        },
        "micro_overfit_windows": _suggest_windows(pp_active, pw_active, T),
    }


def _episodes(active: list[set], pen: list[dict], kind: str) -> list[ContactEpisode]:
    """Runs contiguos de una misma clave estable."""
    open_ep: dict[tuple, list] = {}
    out: list[ContactEpisode] = []
    for k, keys in enumerate(active):
        for key in keys:
            depth = pen[k].get(key, 0.0)
            if key in open_ep:
                st = open_ep[key]
                st[1] = k
                if depth > st[2]:
                    st[2], st[3] = depth, k - st[0]
            else:
                open_ep[key] = [k, k, depth, 0]
        for key in list(open_ep):
            if key not in keys:
                st = open_ep.pop(key)
                out.append(ContactEpisode(key, kind, st[0], st[1], st[2], st[3]))
    for key, st in open_ep.items():
        out.append(ContactEpisode(key, kind, st[0], st[1], st[2], st[3]))
    return out


def _suggest_windows(pp_active, pw_active, T, n=6) -> dict:
    """Índices de transición representativos de cada régimen, para micro-overfit."""
    picks = {"free": [], "pw_birth": [], "pw_persistent": [], "pp_contact": [], "mixed": []}
    for k in range(T - 1):
        pp, pw = pp_active[k], pw_active[k]
        pp1, pw1 = pp_active[k + 1], pw_active[k + 1]
        if not pp and not pw and not pp1 and not pw1:
            if len(picks["free"]) < n:
                picks["free"].append(k)
        if pw1 - pw and len(picks["pw_birth"]) < n:
            picks["pw_birth"].append(k)
        if pw & pw1 and len(picks["pw_persistent"]) < n:
            picks["pw_persistent"].append(k)
        if pp and len(picks["pp_contact"]) < n:
            picks["pp_contact"].append(k)
        if pp and pw and len(picks["mixed"]) < n:
            picks["mixed"].append(k)
    return picks


def _dist_stats(arrays: list[np.ndarray]) -> dict | None:
    if not arrays:
        return None
    x = np.concatenate(arrays)
    if x.size == 0:
        return None
    return {
        "n": int(x.size),
        "min": float(x.min()),
        "p01": float(np.percentile(x, 1)),
        "median": float(np.median(x)),
        "max": float(x.max()),
        "fraction_negative": float((x < 0).mean()),
    }


def _pct(xs, p):
    if not xs:
        return None
    return float(np.percentile(np.asarray(xs, dtype=float), p))


def _safe(t) -> float | None:
    try:
        v = float(t)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(v) else v


# --------------------------------------------------------------------------
# Recomendación de perfil
# --------------------------------------------------------------------------

def recommend(case_reports: list[dict]) -> dict:
    """Hipótesis de perfil a partir de las métricas agregadas.

    No es una observación del régimen: sin fuerzas ni subpasos DEM, lo único
    observable es cuántos snapshots dura geométricamente un solapamiento.
    """
    lens, singles, interior, rho = [], [], [], []
    for r in case_reports:
        ep = r["episodes"]
        if ep["length_snapshots"]["median"] is not None:
            lens.append(ep["length_snapshots"]["median"])
        if ep["fraction_single_snapshot"] is not None:
            singles.append(ep["fraction_single_snapshot"])
        if ep["fraction_interior_compression_peak"] is not None:
            interior.append(ep["fraction_interior_compression_peak"])
        if np.isfinite(r["rho_t"]):
            rho.append(r["rho_t"])

    med_len = statistics.median(lens) if lens else 0.0
    frac_single = statistics.fmean(singles) if singles else 1.0
    frac_interior = statistics.fmean(interior) if interior else 0.0
    med_rho = statistics.median(rho) if rho else float("inf")

    if med_len >= 4 and frac_single < 0.25 and frac_interior > 0.5:
        profile, why = "v3-C", (
            "los solapamientos duran varios snapshots y el pico de compresión "
            "cae dentro del episodio: el contacto está temporalmente resuelto."
        )
    elif med_len <= 1.5 or frac_single > 0.6:
        profile, why = "v3-I", (
            "la mayoría de los contactos aparecen y desaparecen entre snapshots "
            "consecutivos: el intervalo contiene el choque completo."
        )
    else:
        profile, why = "v3-H (hipótesis)", (
            "coexisten episodios de un solo snapshot con contactos persistentes; "
            "ni v3-C ni v3-I cubren solos la distribución observada."
        )
    return {
        "profile": profile,
        "rationale": why,
        "median_episode_snapshots": med_len,
        "mean_fraction_single_snapshot": frac_single,
        "mean_fraction_interior_peak": frac_interior,
        "median_rho_t": med_rho,
        "caveat": (
            "El dataset no contiene fuerzas de contacto, impulsos ni los subpasos "
            "internos del solver (dt_DEM = 1e-7 s). La duración de contacto se "
            "infiere de solapamiento geométrico entre snapshots grabados, que es "
            "una cota inferior: un choque puede empezar y terminar dentro de un "
            "intervalo sin dejar ningún frame con g < 0."
        ),
    }


# --------------------------------------------------------------------------
# Informe
# --------------------------------------------------------------------------

def write_markdown(path: Path, report: dict) -> None:
    r = report
    L: list[str] = []
    A = L.append
    A("# DATA_AUDIT — auditoría temporal del dataset Dynami-CAL")
    A("")
    A("Generado por `scripts/slgnn_v3/audit_temporal.py`. **No editar a mano**: vuelve a")
    A("correr el script si cambia el dataset o la banda de detección.")
    A("")
    A(f"- Fecha de la corrida: `{r['generated_at']}`")
    A(f"- Commit: `{r['git_sha']}` (dirty: {r['git_dirty']})")
    A(f"- Banda de candidatos: `{r['band_factor']} x diámetro`")
    A("")
    A("## 0. Qué no contiene el dataset")
    A("")
    A("Los CSV traen únicamente `q`, `v`, `omega`, `Density` y (a veces) `Diameter`.")
    A("**No hay** fuerzas de contacto, impulsos por contacto, ni los subpasos internos")
    A("del solver MFiX (`dt_DEM = 1e-7 s` según la documentación del dataset). Por eso")
    A("toda la clasificación de régimen que sigue es geométrica y cinemática.")
    A("")
    A("## 1. Verificación de `dt` y de la gravedad contra los datos")
    A("")
    A("`dt` se re-estima por mínimos cuadrados con la regla del punto medio")
    A("`q_{k+1} - q_k = dt (v_k + v_{k+1})/2` sobre transiciones en vuelo libre.")
    A("La gravedad es la aceleración media de esas mismas transiciones.")
    A("")
    A("| Caso | dt doc [s] | dt estimado [s] | error rel. | g medido [m/s²] | eje |")
    A("|---|---|---|---|---|---|")
    for c in r["cases"]:
        t, g = c["timing"], c["gravity"]
        gm = f"{g['magnitude']:.3f}" if g.get("available") else "—"
        ax = g.get("dominant_axis", "—") if g.get("available") else "—"
        A(f"| `{c['case']}` | {t['dt_documented']:.1e} | {t['dt_estimated']:.4e} | "
          f"{t['dt_relative_error']:.2e} | {gm} | {ax} |")
    A("")
    A("## 2. Episodios de contacto por clave estable")
    A("")
    A("Un episodio es un run contiguo de snapshots con `g <= 0` para la misma clave")
    A("`(i,j)` o `(i, cara)`. La mediana en snapshots es la magnitud que decide el")
    A("perfil: si vale 1, el intervalo de grabación contiene el choque entero.")
    A("")
    A("| Caso | episodios | mediana [snap] | % de 1 snap | pico interior | ρ_t | pen. máx / R |")
    A("|---|---|---|---|---|---|---|")
    for c in r["cases"]:
        ep = c["episodes"]
        fs = ep["fraction_single_snapshot"]
        fi = ep["fraction_interior_compression_peak"]
        A(f"| `{c['case']}` | {ep['n_total']} | {ep['length_snapshots']['median']} | "
          f"{'—' if fs is None else f'{100*fs:.0f}%'} | "
          f"{'—' if fi is None else f'{100*fi:.0f}%'} | "
          f"{c['rho_t']:.2f} | {ep['max_penetration_over_radius']:.3f} |")
    A("")
    A("`ρ_t = dt_snapshot / t_contacto`. `ρ_t << 1` es contacto resuelto (v3-C);")
    A("`ρ_t >= 1` es contacto submuestreado (v3-I).")
    A("")
    A("### 2b. Contraste con la duración analítica del contacto DEM")
    A("")
    A("Si la detección geométrica está midiendo el contacto real y no ruido de")
    A("discretización, la duración de episodio debe coincidir con el semiperiodo")
    A("del oscilador masa–resorte del DEM, `t_c = π sqrt(m_eff/k_n)`, con la")
    A("rigidez `k_n` documentada en `data/DATA_NOTES.md` y `m_eff = m/2` para")
    A("partícula–partícula, `m_eff = m` para partícula–pared.")
    A("")
    A("| Caso | k_n [N/m] | t_c pp [snap] | t_c pw [snap] | mediana pp medida | mediana pw medida |")
    A("|---|---|---|---|---|---|")
    for c in r["cases"]:
        a, ep = c["analytic_contact_snapshots"], c["episodes"]
        f = lambda x: "—" if x is None else f"{x:.1f}"
        A(f"| `{c['case']}` | {a['k_n_documented'] or '—'} | {f(a['pp'])} | {f(a['pw'])} | "
          f"{f(ep['pp_median_length'])} | {f(ep['pw_median_length'])} |")
    A("")
    A("La coincidencia entre columnas analíticas y medidas es la validación de")
    A("que esta auditoría mide física y no artefactos. Una discrepancia grande")
    A("indicaría o un `k_n` mal documentado o un detector de contacto roto.")
    A("")
    A("## 3. Nacimiento, persistencia y ruptura")
    A("")
    A("| Caso | nacimientos | muertes | persistencias | % nacimientos sobre activos |")
    A("|---|---|---|---|---|")
    for c in r["cases"]:
        t = c["transitions"]
        A(f"| `{c['case']}` | {t['births']} | {t['deaths']} | {t['persisting']} | "
          f"{100*t['fraction_births_over_active']:.1f}% |")
    A("")
    A("## 4. Saltos de velocidad condicionados a contacto")
    A("")
    A("| Caso | ‖Δv‖ libre (media) | ‖Δv‖ contacto (media) | razón | ‖Δv‖ contacto (máx) |")
    A("|---|---|---|---|---|")
    for c in r["cases"]:
        j = c["velocity_jumps"]
        ratio = j["ratio_contact_over_free"]
        A(f"| `{c['case']}` | {_fmt(j['free_mean'])} | {_fmt(j['contact_mean'])} | "
          f"{'—' if ratio is None else f'{ratio:.1f}x'} | {_fmt(j['contact_max'])} |")
    A("")
    A("## 5. Distribución de gaps y separación pp / pw")
    A("")
    A("| Caso | pp: mediana | pp: mín | % pp<0 | pw: mediana | pw: mín | % pw<0 |")
    A("|---|---|---|---|---|---|---|")
    for c in r["cases"]:
        pp, pw = c["gaps"]["pp"], c["gaps"]["pw_min_over_faces"]
        f = lambda d, k: "—" if d is None else f"{d[k]:.2e}"
        fp = lambda d: "—" if d is None else f"{100*d['fraction_negative']:.2f}%"
        A(f"| `{c['case']}` | {f(pp,'median')} | {f(pp,'min')} | {fp(pp)} | "
          f"{f(pw,'median')} | {f(pw,'min')} | {fp(pw)} |")
    A("")
    A("## 6. Contactos múltiples con la pared (aristas y esquinas)")
    A("")
    A("Número de caras de la caja simultáneamente en contacto con una misma")
    A("partícula, contado sobre todos los pares (frame, partícula):")
    A("")
    A("| Caso | 0 caras | 1 cara | 2 caras | 3 caras |")
    A("|---|---|---|---|---|")
    for c in r["cases"]:
        w = c["wall_multiplicity"]
        A(f"| `{c['case']}` | {w.get('0', 0)} | {w.get('1', 0)} | {w.get('2', 0)} | "
          f"{w.get('3', 0)} |")
    A("")
    A("Cualquier entrada distinta de cero en las columnas de 2 y 3 caras refuta")
    A("`min` sobre caras como interfaz única de contacto (§4.2 de las instrucciones).")
    A("")
    A("## 7. Ventanas recomendadas para micro-overfit")
    A("")
    A("Índices de transición `k` por régimen, listos para fijar un conjunto")
    A("versionado pequeño:")
    A("")
    A("```json")
    A(json.dumps({c["case"]: c["micro_overfit_windows"] for c in r["cases"]}, indent=2))
    A("```")
    A("")
    A("## 8. Recomendación provisional de perfil")
    A("")
    rec = r["recommendation"]
    A(f"**{rec['profile']}** — {rec['rationale']}")
    A("")
    A(f"- mediana de duración de episodio: `{rec['median_episode_snapshots']}` snapshots")
    A(f"- fracción media de episodios de un solo snapshot: "
      f"`{100*rec['mean_fraction_single_snapshot']:.1f}%`")
    A(f"- fracción media con pico de compresión interior: "
      f"`{100*rec['mean_fraction_interior_peak']:.1f}%`")
    A(f"- mediana de `ρ_t`: `{rec['median_rho_t']:.2f}`")
    A("")
    A(f"> {rec['caveat']}")
    A("")
    A("## 9. Comparación con las estimaciones previas del repositorio")
    A("")
    A("`data/DATA_NOTES.md` documenta `dt = 1e-4 s` para caja y benchmarks y")
    A("`1e-3 s` para el cilindro, con `dt` interno del solver `1e-7 s`. La")
    A("estimación por punto medio de la sección 1 confirma o refuta ese valor")
    A("caso por caso; cualquier discrepancia mayor que `1e-3` relativa aparece")
    A("en la columna de error de esa tabla y debe resolverse antes de entrenar.")
    A("")
    A("El informe previo `docs/slgnn_v2/training/Informe_Estrategia_Entrenamiento_SLGNN.md` asumía un")
    A("régimen compliant implícito al usar aceleración como target. Esta")
    A("auditoría reemplaza ese supuesto: la decisión ahora se toma sobre la")
    A("mediana de duración de episodio medida arriba, no sobre la conveniencia")
    A("de la parametrización.")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def _fmt(x) -> str:
    return "—" if x is None else f"{x:.3e}"


def write_csv(path: Path, cases: list[dict]) -> None:
    cols = [
        "case", "n_snapshots", "n_particles", "dt_documented", "dt_estimated",
        "dt_rel_error", "gravity_magnitude", "gravity_axis", "n_episodes",
        "median_episode_snapshots", "fraction_single_snapshot",
        "fraction_interior_peak", "rho_t", "births", "persisting",
        "dv_free_mean", "dv_contact_mean", "wall_2faces", "wall_3faces",
        "max_penetration_over_R",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for c in cases:
            ep, t, g, j = c["episodes"], c["timing"], c["gravity"], c["velocity_jumps"]
            w.writerow([
                c["case"], c["n_snapshots"], c["n_particles"],
                t["dt_documented"], f"{t['dt_estimated']:.6e}",
                f"{t['dt_relative_error']:.3e}",
                f"{g['magnitude']:.4f}" if g.get("available") else "",
                g.get("dominant_axis", "") if g.get("available") else "",
                ep["n_total"], ep["length_snapshots"]["median"],
                ep["fraction_single_snapshot"], ep["fraction_interior_compression_peak"],
                f"{c['rho_t']:.4f}", c["transitions"]["births"],
                c["transitions"]["persisting"],
                j["free_mean"], j["contact_mean"],
                c["wall_multiplicity"].get("2", 0), c["wall_multiplicity"].get("3", 0),
                f"{ep['max_penetration_over_radius']:.4f}",
            ])


def git_state() -> tuple[str, bool]:
    import subprocess
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip())
        return sha, dirty
    except Exception:
        return "unknown", False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path,
                    default=REPO_ROOT / "configs/experiments/audit_temporal.yaml")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results/slgnn_v3/audit_temporal")
    ap.add_argument("--doc", type=Path, default=REPO_ROOT / "docs/slgnn_v3/DATA_AUDIT.md")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    band_factor = float(cfg.get("band_factor", 0.10))
    args.out.mkdir(parents=True, exist_ok=True)

    reports: list[dict] = []
    for entry in cfg["targets"]:
        key = entry["dataset"]
        spec = DATASETS[key]
        cases = entry.get("cases") or list_cases(key, REPO_ROOT)
        max_steps = entry.get("max_steps")
        for case in cases:
            print(f"[audit] {key}/{case} ...", flush=True)
            tr = load_case(key, case, REPO_ROOT, max_steps=max_steps)
            reports.append(audit_case(tr, spec, band_factor))

    from datetime import datetime, timezone
    sha, dirty = git_state()
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": sha,
        "git_dirty": dirty,
        "band_factor": band_factor,
        "cases": reports,
        "recommendation": recommend(reports),
    }
    (args.out / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(args.out / "audit_summary.csv", reports)
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(args.doc, report)
    print(f"\n[audit] recomendación: {report['recommendation']['profile']}")
    print(f"[audit] {args.out / 'audit.json'}")
    print(f"[audit] {args.doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
