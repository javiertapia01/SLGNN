"""Detector de eventos de colisión sobre trayectorias (§7.1).

El espectro de colisiones es un **funcional de trayectorias**: no hace falta
instrumentar el solver. Ese es el contrato que hace intercambiable la fuente
(MFiX o rollout de SLGNN) sin que `C_phi` lo sepa.

Todo en este módulo es **adimensional**: la conversión a SI ocurre una sola vez,
en `twin.coarse` (§6).

Desviación deliberada respecto de la especificación
---------------------------------------------------
Las instrucciones anticipan *aliasing* (contacto más corto que el intervalo
entre snapshots) y por eso proponen leer la velocidad `k_pre` snapshots antes
del mínimo, con `k_pre = 1` si el muestreo es grueso. En este dataset ocurre lo
contrario: medido sobre `Benchmark_2Spheres_Oblique_Collision`, el contacto
**abarca 11 snapshots** y el solapamiento es profundo. Con `k_pre = 1` la
velocidad normal ya está frenada por el resorte de contacto (-0.375 frente a
-1.450 reales en el subcaso `1x`, un factor 15 en energía).

El detector usa por tanto el **onset de contacto**: se camina hacia atrás desde
el mínimo local hasta el último snapshot de vuelo libre (`gap >= delta`), y ahí
se lee la velocidad relativa. Verificado exacto en los 8 subcasos de los dos
micro-benchmarks: la velocidad recuperada coincide con la de vuelo libre a
precisión de máquina (ver `tests/test_events.py`).

`max_lookback` acota el retroceso. Si no se encuentra vuelo libre dentro de esa
ventana, el evento se descarta y se contabiliza en `n_unresolved` — contactos
sostenidos (carga apoyada) no son impactos y no deben entrar en un espectro de
energías de colisión.
"""

from dataclasses import dataclass, field, replace

import numpy as np
import torch

PP = 0  # partícula-partícula
PW = 1  # partícula-pared

_KIND_NAME = {PP: "pp", PW: "pw"}


@dataclass
class EventTable:
    """Tabla de eventos en unidades adimensionales.

    `v_n` es la rapidez de aproximación (positiva al acercarse). `E_impact` es
    la energía de impacto ½·m_eff·v_n², que es la que consumen las leyes de
    rotura. La energía *disipada* no es recuperable de trayectorias sin la ley
    de contacto: se deriva aparte en `twin.coarse`, rotulada como derivada.
    """

    step: np.ndarray      # [K] índice del mínimo local
    onset: np.ndarray     # [K] índice del snapshot de vuelo libre usado
    i: np.ndarray         # [K]
    j: np.ndarray         # [K] -1 para pared
    kind: np.ndarray      # [K] PP o PW
    v_n: np.ndarray       # [K] rapidez normal de aproximación
    v_t: np.ndarray       # [K] rapidez tangencial en el onset
    m_eff: np.ndarray     # [K]
    E_impact: np.ndarray  # [K] ½ m_eff v_n²
    gap_min: np.ndarray   # [K] penetración máxima (gap en el mínimo)
    n_steps: int = 0
    dt: float = 0.0
    n_unresolved: int = 0
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.step.shape[0])

    def select(self, mask: np.ndarray) -> "EventTable":
        mask = np.asarray(mask)
        return replace(
            self,
            step=self.step[mask], onset=self.onset[mask], i=self.i[mask],
            j=self.j[mask], kind=self.kind[mask], v_n=self.v_n[mask],
            v_t=self.v_t[mask], m_eff=self.m_eff[mask],
            E_impact=self.E_impact[mask], gap_min=self.gap_min[mask],
        )

    def window(self, lo: int, hi: int) -> "EventTable":
        """Eventos con mínimo local en [lo, hi). La ventana se define sobre
        `step`, no sobre `onset`, para que cada evento pertenezca a exactamente
        una ventana aunque su onset caiga en la anterior."""
        return self.select((self.step >= lo) & (self.step < hi))

    def of_kind(self, kind: int) -> "EventTable":
        return self.select(self.kind == kind)

    def summary(self) -> str:
        parts = [f"{len(self)} eventos"]
        for k in (PP, PW):
            parts.append(f"{_KIND_NAME[k]}={int((self.kind == k).sum())}")
        parts.append(f"no resueltos={self.n_unresolved}")
        return ", ".join(parts)


def _empty(n_steps: int, dt: float, n_unresolved: int = 0, meta=None) -> EventTable:
    z = np.zeros(0)
    zi = np.zeros(0, dtype=np.int64)
    return EventTable(
        step=zi, onset=zi.copy(), i=zi.copy(), j=zi.copy(), kind=zi.copy(),
        v_n=z, v_t=z.copy(), m_eff=z.copy(), E_impact=z.copy(), gap_min=z.copy(),
        n_steps=n_steps, dt=dt, n_unresolved=n_unresolved, meta=meta or {},
    )


def concat(tables: list[EventTable]) -> EventTable:
    tables = [t for t in tables]
    if not tables:
        raise ValueError("concat de lista vacía")
    cat = lambda name: np.concatenate([getattr(t, name) for t in tables])
    order = np.argsort(cat("step"), kind="stable")
    out = EventTable(
        step=cat("step")[order], onset=cat("onset")[order], i=cat("i")[order],
        j=cat("j")[order], kind=cat("kind")[order], v_n=cat("v_n")[order],
        v_t=cat("v_t")[order], m_eff=cat("m_eff")[order],
        E_impact=cat("E_impact")[order], gap_min=cat("gap_min")[order],
        n_steps=max(t.n_steps for t in tables), dt=tables[0].dt,
        n_unresolved=sum(t.n_unresolved for t in tables),
    )
    for t in tables:
        out.meta.update(t.meta)
    return out


# --------------------------------------------------------------------------
# Núcleo: mínimos locales de una serie de gaps + retroceso al vuelo libre
# --------------------------------------------------------------------------

def _minima_with_onset(gap: np.ndarray, delta: float, max_lookback: int):
    """Mínimos locales de `gap` [T, C] por columna, con su snapshot de onset.

    Devuelve (t_idx, col_idx, onset_idx, n_unresolved). Un evento requiere:
      - mínimo local estricto por la izquierda, no estricto por la derecha;
      - gap en el mínimo por debajo de `delta` (banda de detección);
      - un snapshot de vuelo libre (gap >= delta) a lo sumo `max_lookback`
        atrás; si no lo hay, el evento se descarta como contacto sostenido.

    Los mínimos que comparten onset (rebotes internos de un mismo contacto) se
    colapsan al primero: un impacto por aproximación.
    """
    T = gap.shape[0]
    if T < 3:
        return (np.zeros(0, np.int64),) * 3 + (0,)

    interior = gap[1:-1]
    is_min = (interior < gap[:-2]) & (interior <= gap[2:]) & (interior < delta)
    t_idx, col_idx = np.nonzero(is_min)
    t_idx = t_idx + 1  # el índice 0 de `interior` es el snapshot 1
    if t_idx.size == 0:
        return t_idx, col_idx, t_idx.copy(), 0

    # último snapshot de vuelo libre en o antes de cada t, por columna
    free = gap >= delta
    steps = np.arange(T, dtype=np.int64)[:, None]
    last_free = np.maximum.accumulate(np.where(free, steps, -1), axis=0)
    onset = last_free[t_idx, col_idx]

    ok = (onset >= 0) & ((t_idx - onset) <= max_lookback)
    n_unresolved = int((~ok).sum())
    t_idx, col_idx, onset = t_idx[ok], col_idx[ok], onset[ok]
    if t_idx.size == 0:
        return t_idx, col_idx, onset, n_unresolved

    # colapsar mínimos que comparten (columna, onset) al más temprano
    order = np.lexsort((t_idx, onset, col_idx))
    t_idx, col_idx, onset = t_idx[order], col_idx[order], onset[order]
    keep = np.ones(t_idx.size, dtype=bool)
    keep[1:] = (col_idx[1:] != col_idx[:-1]) | (onset[1:] != onset[:-1])
    return t_idx[keep], col_idx[keep], onset[keep], n_unresolved


def _as_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


# --------------------------------------------------------------------------
# Partícula-partícula
# --------------------------------------------------------------------------

_CELL_LIST_MIN_N = 256  # por debajo de esto, cdist denso es más rápido


def close_pairs_frame(pos: np.ndarray, radii: np.ndarray, band: float) -> np.ndarray:
    """Pares (i<j) con gap < band en un snapshot. [P, 2].

    Con pocas partículas el `cdist` denso gana; a partir de unos cientos, la
    lista de celdas es la diferencia entre minutos y media hora sobre las 2073
    esferas del cilindro (O(N) frente a O(N²) por snapshot).
    """
    n = pos.shape[0]
    cutoff = 2.0 * float(radii.max()) + max(band, 0.0)
    if n < _CELL_LIST_MIN_N:
        d = torch.cdist(torch.from_numpy(pos), torch.from_numpy(pos)).numpy()
        contact = radii[:, None] + radii[None, :]
        ii, jj = np.nonzero(np.triu(d - contact < band, k=1))
        return np.stack([ii, jj], axis=1)

    lo = pos.min(axis=0)
    cell = np.floor((pos - lo) / max(cutoff, 1e-30)).astype(np.int64)
    dims = cell.max(axis=0) + 3
    flat = (cell[:, 0] * dims[1] + cell[:, 1]) * dims[2] + cell[:, 2]
    order = np.argsort(flat, kind="stable")
    sorted_flat = flat[order]
    idx = np.arange(n, dtype=np.int64)

    chunks = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                key = (((cell[:, 0] + dx) * dims[1] + cell[:, 1] + dy) * dims[2]
                       + cell[:, 2] + dz)
                left = np.searchsorted(sorted_flat, key, side="left")
                right = np.searchsorted(sorted_flat, key, side="right")
                counts = right - left
                total = int(counts.sum())
                if total == 0:
                    continue
                src = np.repeat(idx, counts)
                starts = np.repeat(left, counts)
                offs = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
                dst = order[starts + offs]
                keep = src < dst
                if keep.any():
                    chunks.append(np.stack([src[keep], dst[keep]], axis=1))
    if not chunks:
        return np.zeros((0, 2), dtype=np.int64)
    cand = np.concatenate(chunks)
    rel = pos[cand[:, 1]] - pos[cand[:, 0]]
    gap = np.linalg.norm(rel, axis=-1) - (radii[cand[:, 0]] + radii[cand[:, 1]])
    return cand[gap < band]


def _close_pairs(pos: np.ndarray, radii: np.ndarray, band: float) -> np.ndarray:
    """Pares (i<j) que en algún snapshot del bloque quedan a gap < band."""
    n = pos.shape[1]
    codes = []
    for t in range(pos.shape[0]):
        p = close_pairs_frame(pos[t], radii, band)
        if p.shape[0]:
            codes.append(p[:, 0] * n + p[:, 1])
    if not codes:
        return np.zeros((0, 2), dtype=np.int64)
    uniq = np.unique(np.concatenate(codes))
    return np.stack([uniq // n, uniq % n], axis=1)


def detect_pp_events(pos, vel, radii, m, *, delta=0.0, max_lookback=32,
                     v_n_min=0.0, dt=1.0, block=256) -> EventTable:
    """Colisiones partícula-partícula sobre `pos`, `vel` [T, N, 3] adimensionales.

    `delta` es la banda de detección sobre el gap superficial d - (R_i + R_j):
    delta = 0 detecta solapamiento real; delta > 0 amplía la banda.

    El costo es O(T · N²) por el `cdist` de cada snapshot, dominante pero
    tolerable en CPU hasta ~2000 partículas. Se procesa por bloques con un
    solape de `max_lookback + 1` snapshots para no perder eventos ni onsets en
    las fronteras.
    """
    pos, vel, radii, m = (_as_numpy(x) for x in (pos, vel, radii, m))
    T = pos.shape[0]
    pad = max_lookback + 1
    tables, n_unres = [], 0

    for a in range(0, T, block):
        b = min(a + block, T)
        lo, hi = max(a - pad, 0), min(b + 1, T)
        chunk = pos[lo:hi]
        pairs = _close_pairs(chunk, radii, band=max(delta, 0.0) + 1e-12)
        if pairs.shape[0] == 0:
            continue
        pi, pj = pairs[:, 0], pairs[:, 1]
        # acumular por componente evita materializar [Tc, P, 3], que con miles de
        # partículas y bloques largos es el pico de memoria de todo el pipeline
        sq = np.zeros((chunk.shape[0], pairs.shape[0]))
        for d in range(3):
            diff = chunk[:, pj, d] - chunk[:, pi, d]
            sq += diff * diff
        gap = np.sqrt(sq) - (radii[pi] + radii[pj])
        del sq

        t_loc, col, onset_loc, nu = _minima_with_onset(gap, delta, max_lookback)
        n_unres += nu
        # quedarse solo con los mínimos cuyo snapshot cae en el bloque [a, b)
        t_abs = t_loc + lo
        sel = (t_abs >= a) & (t_abs < b)
        t_loc, col, onset_loc, t_abs = t_loc[sel], col[sel], onset_loc[sel], t_abs[sel]
        if t_abs.size == 0:
            continue

        ii, jj = pi[col], pj[col]
        o_abs = onset_loc + lo
        r0 = pos[o_abs, jj] - pos[o_abs, ii]
        e = r0 / np.linalg.norm(r0, axis=-1, keepdims=True)
        v_rel = vel[o_abs, jj] - vel[o_abs, ii]
        v_n = -np.einsum("kd,kd->k", v_rel, e)                 # >0 al acercarse
        v_t = np.linalg.norm(v_rel + v_n[:, None] * e, axis=-1)
        m_eff = m[ii] * m[jj] / (m[ii] + m[jj])

        keep = v_n > v_n_min
        tables.append(EventTable(
            step=t_abs[keep], onset=o_abs[keep], i=ii[keep], j=jj[keep],
            kind=np.full(int(keep.sum()), PP, dtype=np.int64),
            v_n=v_n[keep], v_t=v_t[keep], m_eff=m_eff[keep],
            E_impact=0.5 * m_eff[keep] * v_n[keep] ** 2,
            gap_min=gap[t_loc, col][keep], n_steps=T, dt=dt,
        ))

    if not tables:
        return _empty(T, dt, n_unres)
    out = concat(tables)
    out.n_unresolved = n_unres
    return out


# --------------------------------------------------------------------------
# Partícula-pared
# --------------------------------------------------------------------------

def detect_pw_events(pos, vel, radii, m, sdf, *, delta=0.0, max_lookback=32,
                     v_n_min=0.0, dt=1.0, t0=0.0, eps=1e-8) -> EventTable:
    """Colisiones partícula-pared usando `sdf.phi`, `sdf.grad_phi` y
    `sdf.wall_velocity` — la cinemática de pared explícita que la v2 ya provee.

    El tiempo adimensional del snapshot k es `t0 + k·dt`, y se pasa a la SDF
    para que `wall_velocity` refleje la rampa de omega del cilindro.
    """
    pos_t = pos if isinstance(pos, torch.Tensor) else torch.as_tensor(pos)
    pos_t = pos_t.to(torch.float64)
    pos_np, vel_np, radii_np, m_np = (_as_numpy(x) for x in (pos, vel, radii, m))
    T, N = pos_np.shape[0], pos_np.shape[1]

    radii_t = torch.as_tensor(radii_np, dtype=torch.float64)
    with torch.no_grad():
        phi = np.stack([
            sdf.phi(pos_t[k], t0 + k * dt).numpy() for k in range(T)
        ])
    gap = phi - radii_np[None, :]

    t_idx, col, onset, n_unres = _minima_with_onset(gap, delta, max_lookback)
    if t_idx.size == 0:
        return _empty(T, dt, n_unres)

    v_n = np.zeros(t_idx.size)
    v_t = np.zeros(t_idx.size)
    for k in np.unique(onset):
        sel = onset == k
        idx = col[sel]
        q = pos_t[k][idx].clone()
        _, nu, _, v_w = sdf.query(q, t0 + float(k) * dt, eps)
        nu = nu.detach().numpy()
        # `sdf.query` normaliza con un eps de estabilidad para autograd, que deja
        # |nu| = 1 - O(eps) y filtra ~1e-8·v_n a la componente tangencial. Aquí
        # estamos fuera del grafo, así que la normalización puede ser exacta y la
        # descomposición normal/tangencial queda limpia.
        nu = nu / np.linalg.norm(nu, axis=-1, keepdims=True)
        v_w = v_w.detach().numpy()
        w_rel = vel_np[k][idx] - v_w
        vn = -np.einsum("kd,kd->k", w_rel, nu)  # nu entrante: >0 al acercarse
        v_n[sel] = vn
        v_t[sel] = np.linalg.norm(w_rel + vn[:, None] * nu, axis=-1)

    m_eff = m_np[col]  # la pared tiene masa infinita
    keep = v_n > v_n_min
    return EventTable(
        step=t_idx[keep], onset=onset[keep], i=col[keep],
        j=np.full(int(keep.sum()), -1, dtype=np.int64),
        kind=np.full(int(keep.sum()), PW, dtype=np.int64),
        v_n=v_n[keep], v_t=v_t[keep], m_eff=m_eff[keep],
        E_impact=0.5 * m_eff[keep] * v_n[keep] ** 2,
        gap_min=gap[t_idx, col][keep], n_steps=T, dt=dt, n_unresolved=n_unres,
    )


def detect_events(pos, vel, radii, m, sdf=None, *, delta=0.0, max_lookback=32,
                  v_n_min=0.0, dt=1.0, t0=0.0, block=256) -> EventTable:
    """Ambos canales en una sola tabla."""
    tables = [detect_pp_events(pos, vel, radii, m, delta=delta,
                               max_lookback=max_lookback, v_n_min=v_n_min,
                               dt=dt, block=block)]
    if sdf is not None:
        tables.append(detect_pw_events(pos, vel, radii, m, sdf, delta=delta,
                                       max_lookback=max_lookback,
                                       v_n_min=v_n_min, dt=dt, t0=t0))
    return concat(tables)
