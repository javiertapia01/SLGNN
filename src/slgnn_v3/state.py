"""Estado de partículas y validación de contratos (§3.1 de las instrucciones).

`ParticleBatch` es un batch **concatenado**: varios sistemas físicos
independientes viven en los mismos tensores, separados por `batch_id`. Ninguna
arista puede cruzar ese límite; `graph.py` lo garantiza y
`tests/comparison/test_no_batch_leakage.py` lo verifica.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch


@dataclass
class ParticleBatch:
    """Partículas esféricas con masa e inercia conocidas."""

    q: torch.Tensor           # [N, 3]
    v: torch.Tensor           # [N, 3]
    omega: torch.Tensor       # [N, 3]
    mass: torch.Tensor        # [N]
    radius: torch.Tensor      # [N]
    inertia: torch.Tensor     # [N]
    type_id: torch.Tensor     # [N] long
    batch_id: torch.Tensor    # [N] long
    properties: torch.Tensor | None = None   # [N, P] escalares extra

    # -- constructores -----------------------------------------------------
    @classmethod
    def from_arrays(
        cls,
        q: torch.Tensor,
        v: torch.Tensor,
        omega: torch.Tensor,
        mass: torch.Tensor,
        radius: torch.Tensor,
        inertia: torch.Tensor | None = None,
        type_id: torch.Tensor | None = None,
        batch_id: torch.Tensor | None = None,
        properties: torch.Tensor | None = None,
    ) -> "ParticleBatch":
        n = q.shape[0]
        if inertia is None:
            # esfera maciza homogénea: I = 2/5 m R^2
            inertia = 0.4 * mass * radius**2
        if type_id is None:
            type_id = torch.zeros(n, dtype=torch.long, device=q.device)
        if batch_id is None:
            batch_id = torch.zeros(n, dtype=torch.long, device=q.device)
        pb = cls(q, v, omega, mass, radius, inertia, type_id, batch_id, properties)
        pb.validate()
        return pb

    @staticmethod
    def concat(batches: list["ParticleBatch"]) -> "ParticleBatch":
        """Concatena sistemas independientes reindexando `batch_id`."""
        offset = 0
        bids = []
        for b in batches:
            bids.append(b.batch_id + offset)
            offset += int(b.batch_id.max()) + 1
        cat = lambda name: torch.cat([getattr(b, name) for b in batches], dim=0)
        props = None
        if all(b.properties is not None for b in batches):
            props = cat("properties")
        out = ParticleBatch(
            q=cat("q"), v=cat("v"), omega=cat("omega"), mass=cat("mass"),
            radius=cat("radius"), inertia=cat("inertia"), type_id=cat("type_id"),
            batch_id=torch.cat(bids), properties=props,
        )
        out.validate()
        return out

    # -- propiedades -------------------------------------------------------
    @property
    def n(self) -> int:
        return int(self.q.shape[0])

    @property
    def n_systems(self) -> int:
        return int(self.batch_id.max()) + 1 if self.n else 0

    @property
    def dtype(self) -> torch.dtype:
        return self.q.dtype

    @property
    def device(self) -> torch.device:
        return self.q.device

    def replace(self, **kw) -> "ParticleBatch":
        return replace(self, **kw)

    def detach(self) -> "ParticleBatch":
        return replace(
            self, q=self.q.detach(), v=self.v.detach(), omega=self.omega.detach()
        )

    def requires_grad_q(self) -> "ParticleBatch":
        """Copia con `q` como hoja diferenciable: la fuerza conservativa se
        obtiene por autograd respecto de esta hoja."""
        return replace(self, q=self.q.detach().clone().requires_grad_(True))

    def to(self, dtype=None, device=None) -> "ParticleBatch":
        f = lambda x: None if x is None else x.to(
            dtype=dtype if (dtype and x.is_floating_point()) else x.dtype,
            device=device or x.device,
        )
        return ParticleBatch(
            q=f(self.q), v=f(self.v), omega=f(self.omega), mass=f(self.mass),
            radius=f(self.radius), inertia=f(self.inertia), type_id=f(self.type_id),
            batch_id=f(self.batch_id), properties=f(self.properties),
        )

    # -- validación --------------------------------------------------------
    def validate(self) -> None:
        n = self.n
        for name, x, shape in (
            ("q", self.q, (n, 3)), ("v", self.v, (n, 3)), ("omega", self.omega, (n, 3)),
            ("mass", self.mass, (n,)), ("radius", self.radius, (n,)),
            ("inertia", self.inertia, (n,)), ("type_id", self.type_id, (n,)),
            ("batch_id", self.batch_id, (n,)),
        ):
            if tuple(x.shape) != shape:
                raise ValueError(f"ParticleBatch.{name}: shape {tuple(x.shape)} != {shape}")
        floats = [self.q, self.v, self.omega, self.mass, self.radius, self.inertia]
        dtypes = {x.dtype for x in floats}
        if len(dtypes) > 1:
            raise ValueError(f"ParticleBatch: dtypes mezclados {dtypes}")
        devices = {x.device for x in floats + [self.type_id, self.batch_id]}
        if len(devices) > 1:
            raise ValueError(f"ParticleBatch: devices mezclados {devices}")
        for name, x in (("mass", self.mass), ("radius", self.radius),
                        ("inertia", self.inertia)):
            if not bool((x > 0).all()):
                raise ValueError(f"ParticleBatch.{name} debe ser estrictamente positivo")
        for name, x in (("q", self.q), ("v", self.v), ("omega", self.omega)):
            if not bool(torch.isfinite(x).all()):
                raise ValueError(f"ParticleBatch.{name} contiene NaN/Inf")
        if int(self.batch_id.min()) < 0:
            raise ValueError("ParticleBatch.batch_id negativo")


@dataclass
class ContactMemoryState:
    """Memoria tangencial persistente `xi` (cabeza `M`).

    **Contrato declarado, no implementado.** Existe para que la firma del paso
    y los diagnósticos no cambien cuando se implemente la fase 10, y para que
    ninguna parte del MVP pueda simular memoria por accidente: `xi` se
    mantiene vacío y `enabled` es siempre `False` en esta etapa.
    """

    keys: torch.Tensor | None = None   # [K, 3] long: (batch, i, j|surface)
    xi: torch.Tensor | None = None     # [K, 3] desplazamiento tangencial
    age: torch.Tensor | None = None    # [K] long
    enabled: bool = False

    @staticmethod
    def empty() -> "ContactMemoryState":
        return ContactMemoryState()

    def __post_init__(self) -> None:
        if self.enabled:
            raise NotImplementedError(
                "La memoria tangencial M no está implementada en el MVP normal. "
                "Ver docs/slgnn_v3/IMPLEMENTATION_STATUS.md, fase 10."
            )


@dataclass
class V3State:
    """Estado completo: partículas, tiempo real y memoria de contacto."""

    particles: ParticleBatch
    time: float | torch.Tensor = 0.0
    memory: ContactMemoryState | None = None

    def time_scalar(self) -> float:
        """Tiempo real como escalar. Nunca se sustituye por 0 durante rollout."""
        t = self.time
        return float(t) if not isinstance(t, torch.Tensor) else float(t.reshape(-1)[0])
