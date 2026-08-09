"""Sampler estratificado por régimen de contacto (§13.3).

Muestrear uniformemente los frames del dataset de 60 esferas es un error
medible: la auditoría temporal muestra que la mayoría de los pares
(frame, partícula) están en vuelo libre, de modo que un batch uniforme casi no
contiene la física que el modelo debe aprender.

El sampler asigna **cuotas por estrato** y registra la composición real de
cada época, que se guarda en el manifiesto. El mismo objeto lo usan v3 y GNS.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import torch

from .contact_labels import STRATA, RegimeLabels, stratum_of
from .data import Trajectory


@dataclass
class TransitionIndex:
    """Índice global de transiciones `(trayectoria, k)` con su estrato."""

    items: list[tuple[int, int]]           # (traj_idx, k)
    strata: list[str]
    by_stratum: dict[str, list[int]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.items)

    def composition(self) -> dict[str, int]:
        return {s: len(v) for s, v in sorted(self.by_stratum.items())}


def build_index(
    trajectories: list[Trajectory],
    labels: list[RegimeLabels],
    high_compression: float = 0.05,
) -> TransitionIndex:
    items: list[tuple[int, int]] = []
    strata: list[str] = []
    by: dict[str, list[int]] = defaultdict(list)
    for t_idx, (tr, lab) in enumerate(zip(trajectories, labels)):
        for k in range(tr.n_steps - 1):
            s = stratum_of(lab, k, high_compression)
            by[s].append(len(items))
            items.append((t_idx, k))
            strata.append(s)
    return TransitionIndex(items=items, strata=strata, by_stratum=dict(by))


@dataclass
class StratifiedSampler:
    """Muestrea con cuotas por estrato, con reemplazo dentro del estrato.

    Los estratos vacíos se redistribuyen proporcionalmente entre los que sí
    tienen datos, y esa redistribución se registra: una cuota pedida que no
    puede cumplirse no debe convertirse silenciosamente en otra cosa.
    """

    index: TransitionIndex
    quotas: dict[str, float]
    seed: int = 0
    _gen: torch.Generator = field(init=False, repr=False)

    def __post_init__(self):
        self._gen = torch.Generator().manual_seed(self.seed)
        total = sum(self.quotas.values())
        if total <= 0:
            raise ValueError("Las cuotas deben sumar un valor positivo")
        self.effective_quotas = self._resolve_quotas()

    def _resolve_quotas(self) -> dict[str, float]:
        available = {s: q for s, q in self.quotas.items()
                     if self.index.by_stratum.get(s)}
        self.dropped_strata = sorted(set(self.quotas) - set(available))
        if not available:
            raise ValueError(
                f"Ningún estrato pedido tiene datos. Pedidos: {sorted(self.quotas)}; "
                f"disponibles: {sorted(self.index.by_stratum)}"
            )
        total = sum(available.values())
        return {s: q / total for s, q in available.items()}

    def sample(self, batch_size: int) -> list[tuple[int, int]]:
        picks: list[int] = []
        names = list(self.effective_quotas)
        counts = [int(round(self.effective_quotas[s] * batch_size)) for s in names]
        # ajusta el redondeo sobre el estrato mayoritario
        while sum(counts) < batch_size:
            counts[int(torch.argmax(torch.tensor(counts, dtype=torch.float64)))] += 1
        while sum(counts) > batch_size:
            counts[int(torch.argmax(torch.tensor(counts, dtype=torch.float64)))] -= 1
        for name, c in zip(names, counts):
            pool = self.index.by_stratum[name]
            if c <= 0 or not pool:
                continue
            sel = torch.randint(0, len(pool), (c,), generator=self._gen)
            picks += [pool[int(i)] for i in sel]
        return [self.index.items[i] for i in picks]

    def epoch_composition(self, batch_size: int, n_batches: int) -> dict[str, int]:
        """Composición esperada de una época, para el manifiesto."""
        out = defaultdict(int)
        for name, frac in self.effective_quotas.items():
            out[name] = int(round(frac * batch_size * n_batches))
        return dict(out)


def uniform_index(index: TransitionIndex) -> dict[str, float]:
    """Cuotas que reproducen el muestreo uniforme, para la ablation."""
    n = max(len(index), 1)
    return {s: len(v) / n for s, v in index.by_stratum.items()}


DEFAULT_QUOTAS: dict[str, float] = {
    "free": 0.15,
    "pp": 0.20,
    "pw": 0.20,
    "mixed": 0.20,
    "contact_birth": 0.20,
    "high_compression": 0.05,
}
