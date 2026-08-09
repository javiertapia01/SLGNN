"""`GNSControlled`: baseline con decoder directo por nodo.

Interfaz **idéntica** a `SLGNNv3.step`, para que el runner, el sampler y las
métricas sean literalmente los mismos. Lo único que cambia es lo que ocurre
entre las entradas y `(Delta p, Delta L)`:

    v3   : energías y disipación -> fuerzas -> impulsos por solver -> integrador
    GNS  : latente por nodo -> MLP -> (Delta p, Delta L)

Sin potencial, sin convexidad, sin cono, sin complementariedad, sin
conservación impuesta. Si el sesgo inductivo de v3 sirve, tiene que notarse
aquí; si no sirve, este baseline lo mostrará y ese también es un resultado.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from slgnn_experiments.scene import shared_graph

from .config import GNSConfig
from .encoder import GNSEncoder, mlp, to_directed, wall_features
from .processor import GNSProcessor


class _Result:
    """Mismo contrato de salida que `slgnn_v3.StepResult`, con los campos que
    un decoder directo puede llenar honestamente. Los que no existen en este
    modelo se marcan como desactivados, no se inventan."""

    def __init__(self, next_state, delta_p, delta_L, contacts=None, diagnostics=None):
        self.next_state = next_state
        self.delta_p = delta_p
        self.delta_L = delta_L
        self.delta_p_regular = delta_p
        self.delta_p_impulse = torch.zeros_like(delta_p)
        self.delta_L_regular = delta_L
        self.delta_L_impulse = torch.zeros_like(delta_L)
        self.impulses = delta_p.new_zeros(0)
        self.contacts = contacts
        self.mode = None
        self.diagnostics = diagnostics


class GNSControlled(nn.Module):
    """Baseline controlado: mismos datos y targets, decoder vectorial directo."""

    def __init__(self, cfg: GNSConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = GNSEncoder(cfg)
        self.processor = GNSProcessor(cfg)
        self.decoder = mlp([cfg.hidden, cfg.hidden, cfg.hidden, 6],
                           cfg.activation, layer_norm=False)
        nn.init.normal_(self.decoder[-1].weight, std=1e-2)
        nn.init.zeros_(self.decoder[-1].bias)

    # -- compatibilidad con el runner --------------------------------------
    def reset_lifecycle(self) -> None:
        """GNS no tiene estado de contacto. Existe para que el runner sea uno."""

    def n_parameters(self) -> tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        return total, sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def profile(self) -> str:
        return "gns-controlled"

    # -- paso ---------------------------------------------------------------
    def step(self, state, dt: float, surfaces, gravity=None, eval_mode: bool = False,
             create_graph=None):
        p = state.particles
        p.validate()
        t = state.time_scalar()

        wq = surfaces.query(p.q, p.radius, p.batch_id, t, self.cfg.pw_gap_off)
        edges = shared_graph(p, self.cfg.pp_gap_off, self.cfg.skin)
        directed = to_directed(edges)

        h_node = self.encoder.node_features(p, wall_features(p, wq, p.n))
        h_edge = self.encoder.edge_features(p, directed)
        h = self.processor(h_node, h_edge, directed)
        out = self.cfg.output_scale * self.decoder(h)
        delta_p, delta_L = out[:, :3], out[:, 3:]

        # La gravedad es una entrada conocida también aquí: se suma su impulso
        # analítico para que el baseline no gaste capacidad en reaprenderla.
        # Es la misma información que recibe v3, no una ventaja.
        if gravity is not None:
            g = torch.as_tensor(gravity, dtype=p.dtype, device=p.device)
            delta_p = delta_p + dt * p.mass.unsqueeze(-1) * g

        v_next = p.v + delta_p / p.mass.unsqueeze(-1)
        w_next = p.omega + delta_L / p.inertia.unsqueeze(-1)
        q_next = p.q + dt * v_next
        next_state = type(state)(
            p.replace(q=q_next, v=v_next, omega=w_next), time=t + dt,
            memory=getattr(state, "memory", None),
        )
        return _Result(next_state, delta_p, delta_L, contacts=None,
                       diagnostics=_diagnostics(edges, wq))


DISABLED = "disabled"


class GNSDiagnostics:
    """Diagnósticos del baseline, con la misma disciplina que los de v3: lo que
    este modelo no tiene se marca `disabled` en vez de rellenarse con ceros.

    Contenedor propio y no el de v3: `gns_baseline` no importa `slgnn_v3`.
    """

    def __init__(self, sections: dict):
        self._sections = sections
        for name, value in sections.items():
            setattr(self, name, value)

    def sections(self) -> dict:
        return self._sections

    def disabled_fields(self) -> list[str]:
        return [f"{n}.{k}" for n, sec in self._sections.items()
                for k, v in sec.items() if v == DISABLED]

    def scalars(self) -> dict[str, float]:
        return {
            f"{n}.{k}": float(v) for n, sec in self._sections.items()
            for k, v in sec.items() if isinstance(v, (int, float))
        }


def _diagnostics(edges, wq):
    unavailable = {"available": DISABLED}
    return GNSDiagnostics({
        "energies": dict(unavailable), "dissipation": dict(unavailable),
        "impact": dict(unavailable), "router": dict(unavailable),
        "solver": dict(unavailable),
        "regular": {"decoder": "direct", "force_V": DISABLED, "force_Psi": DISABLED},
        "balance": {"conservation": DISABLED},
        "geometry": {
            "n_candidates": int(edges.shape[0] + wq.particle.numel()),
            "n_pp": int(edges.shape[0]), "n_pW": int(wq.particle.numel()),
            "gap_min": (float(wq.gap.detach().min()) if wq.gap.numel()
                        else float("nan")),
        },
        "wall": {"wall_velocity_max": (
            float(wq.wall_velocity.detach().norm(dim=-1).max())
            if wq.particle.numel() else 0.0)},
    })
