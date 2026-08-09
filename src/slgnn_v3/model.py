"""SLGNN-v3: modelo completo y orquestación de un paso temporal.

La API principal representa **un paso temporal completo**, no una predicción
de aceleración. `step` implementa el orden exacto de §11 de las instrucciones:

 1. validar estado y tiempo
 2. consultar superficies y velocidad de pared
 3. construir grafo candidato
 4. construir cinemática y operadores
 5. ejecutar encoder y processors
 6. obtener router
 7. evaluar V y Psi solo sobre COMPLIANT
 8. sumar fuerzas externas conocidas
 9. obtener nu_free
10. detectar candidatos impulsivos usando el estado libre
11. predecir (e, kappa) y resolver globalmente por componente conexa
12. aplicar J^T Lambda
13. actualizar posición con la velocidad post-impulso
14. reconstruir diagnósticos

Este módulo **no importa `slgnn.model`** ni su integrador: v3 es un paquete
independiente (§2.2).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import memory as memory_mod
from .closure import ClosureHead
from .config import RouterProfile, V3Config
from .contact_kinematics import ContactSet, build_contacts, empty_contacts
from .contact_operator import (
    JT_times_normal_scalars,
    J_times_velocity,
    assemble_normal_delassus,
    connected_components,
    pack_by_component,
    unpack_by_component,
)
from .diagnostics import (
    DISABLED,
    StepDiagnostics,
    internal_momentum_error,
    wall_transfer,
)
from .dissipation import DissipationHead
from .encoder import GeometricEncoder, N_EDGE_KIN, kinematic_features
from .graph import assert_no_cross_batch, build_candidate_graph
from .impact import ImpactHead
from .integrator import (
    StepResult,
    advance_position,
    apply_impulse,
    free_velocity,
    gravity_force,
)
from .potential import PotentialHead, conservative_force
from .processor import ContactProcessor
from .router import (
    ContactLifecycle,
    ContactMode,
    assert_no_double_counting,
    build_router,
    impulsive_mask,
    regular_weight,
)
from .solver import solve_normal
from .state import ParticleBatch, V3State
from .surfaces import SurfaceSet


class SLGNNv3(nn.Module):
    """Modelo SLGNN-v3 con perfiles normales `v3-C` y `v3-I`."""

    def __init__(self, cfg: V3Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = GeometricEncoder(cfg.encoder)
        # V no recibe cinemática: `extra_dim = 0`. Esa asimetría es la
        # prohibición de fuga entre canales hecha código.
        self.proc_V = ContactProcessor(cfg.encoder, extra_dim=0)
        self.proc_Psi = ContactProcessor(cfg.encoder, extra_dim=N_EDGE_KIN)
        self.proc_I = ContactProcessor(cfg.encoder, extra_dim=N_EDGE_KIN)
        self.head_V = PotentialHead(cfg.potential, cfg.encoder)
        self.head_Psi = DissipationHead(cfg.dissipation, cfg.encoder)
        self.head_I = ImpactHead(cfg.impact, cfg.encoder)
        self.closure = ClosureHead(enabled=cfg.closure_enabled)
        if cfg.memory_enabled:
            raise NotImplementedError(
                "memory_enabled=True: la cabeza M no está implementada. "
                "Ver src/slgnn_v3/memory.py."
            )
        self._route = build_router(cfg.router)
        self.lifecycle = ContactLifecycle(cfg.router)

    # -- utilidades --------------------------------------------------------
    def reset_lifecycle(self) -> None:
        """Obligatorio al empezar una trayectoria o un rollout nuevo."""
        self.lifecycle.reset()

    def n_parameters(self) -> tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    @property
    def profile(self) -> RouterProfile:
        return self.cfg.router.profile

    # -- paso --------------------------------------------------------------
    def step(
        self,
        state: V3State,
        dt: float,
        surfaces: SurfaceSet,
        gravity: torch.Tensor | None = None,
        eval_mode: bool = False,
        create_graph: bool | None = None,
    ) -> StepResult:
        cfg = self.cfg
        particles = state.particles
        particles.validate()
        t = state.time_scalar()
        create_graph = self.training if create_graph is None else create_graph

        # `q` debe estar en el grafo de autograd para que F_V = -grad_q V sea
        # una fuerza real. Si viene de un paso anterior ya lo está y no se
        # clona, para que el gradiente atraviese el rollout.
        q = particles.q
        if not q.requires_grad:
            q = q.clone().requires_grad_(True)
            particles = particles.replace(q=q)

        # (2)(3)(4) geometría --------------------------------------------
        wall = surfaces.query(
            q, particles.radius, particles.batch_id, t, cfg.graph.pw_gap_off
        )
        edges = build_candidate_graph(particles, cfg.graph.pp_gap_off, cfg.graph.skin)
        assert_no_cross_batch(edges, particles.batch_id)
        contacts = build_contacts(particles, edges, wall, cfg)

        active, birth, age = self.lifecycle.step(contacts)

        # (5) encoder y processors ---------------------------------------
        h_node, h_edge, _ = self.encoder(contacts, particles)
        kin = kinematic_features(contacts, particles, age)
        h_V = self.proc_V(contacts, h_node, h_edge)
        h_Psi = self.proc_Psi(contacts, h_node, h_edge, kin)

        # (6)(7)(8)(9) rama regular con el peso preliminar del perfil ------
        weight = self._preliminary_weight(contacts)
        reg = self._regular_branch(particles, contacts, h_V, h_Psi, weight, gravity,
                                   create_graph)
        v_free, w_free = free_velocity(particles, reg["force"], reg["torque"], dt)

        # (10) detección con el estado libre --------------------------------
        gap_free = self._free_gaps(particles, contacts, v_free, dt, surfaces, t)
        u_free = J_times_velocity(contacts, v_free, w_free)
        u_n_free = (u_free * contacts.n).sum(dim=-1)
        mode = self._route(contacts, cfg.router, active, gap_free, u_n_free)
        assert_no_double_counting(mode)

        # v3-C con protección impulsiva puede reclasificar un contacto que ya
        # recibió V+Psi: en ese caso se recalcula la rama regular con el peso
        # correcto en vez de sumar las dos respuestas normales.
        new_weight = regular_weight(mode, particles.dtype)
        if bool((new_weight != weight).any()):
            weight = new_weight
            reg = self._regular_branch(particles, contacts, h_V, h_Psi, weight,
                                       gravity, create_graph)
            v_free, w_free = free_velocity(particles, reg["force"], reg["torque"], dt)
            u_free = J_times_velocity(contacts, v_free, w_free)
            u_n_free = (u_free * contacts.n).sum(dim=-1)

        # (11) cabeza I y solver acoplado ----------------------------------
        imp = self._impulsive_branch(
            particles, contacts, mode, h_node, h_edge, kin, u_free, u_n_free,
            birth, age, dt, eval_mode,
        )

        # (12)(13) integración ---------------------------------------------
        v_next, w_next = apply_impulse(
            particles, v_free, w_free, imp["force"], imp["torque"]
        )
        q_next = advance_position(q, v_next, dt)

        next_particles = particles.replace(q=q_next, v=v_next, omega=w_next)
        next_state = V3State(next_particles, time=t + dt, memory=state.memory)

        # (14) diagnósticos --------------------------------------------------
        dp_reg = dt * reg["force"]
        dL_reg = dt * reg["torque"]
        diag = self._diagnostics(
            particles, contacts, mode, reg, imp, dp_reg, dL_reg, gravity, dt, birth, age
        )
        return StepResult(
            next_state=next_state,
            delta_p_regular=dp_reg, delta_p_impulse=imp["force"],
            delta_L_regular=dL_reg, delta_L_impulse=imp["torque"],
            forces=reg["force"], torques=reg["torque"],
            impulses=imp["lambda_n"], diagnostics=diag,
            contacts=contacts, mode=mode,
        )

    # -- piezas ------------------------------------------------------------
    def _preliminary_weight(self, contacts: ContactSet) -> torch.Tensor:
        """Peso `(1 - gamma)` antes de conocer el estado libre.

        En `v3-I` es cero: ningún contacto es compliant, así que las fuerzas
        regulares son solo las externas y `nu*` puede calcularse sin circular.
        En `v3-C` es la ventana, y solo la protección impulsiva puede
        corregirlo después.
        """
        if self.cfg.router.profile is RouterProfile.IMPULSIVE:
            return torch.zeros_like(contacts.gap)
        return (contacts.window > 0).to(contacts.gap.dtype)

    def _regular_branch(
        self, particles, contacts, h_V, h_Psi, weight, gravity, create_graph
    ) -> dict:
        n = particles.n
        dtype, dev = particles.dtype, particles.device

        V, v_parts = self.head_V.total_potential(contacts, h_V, weight)
        f_V = (
            conservative_force(V, particles.q, create_graph)
            if contacts.n_contacts else torch.zeros(n, 3, dtype=dtype, device=dev)
        )
        lam_psi, psi_total, psi_diag = self.head_Psi(
            contacts, h_Psi, weight * contacts.activation
        )
        if contacts.n_contacts:
            from .contact_operator import JT_times_contact_vector
            f_psi, t_psi = JT_times_contact_vector(contacts, lam_psi, n)
        else:
            f_psi = torch.zeros(n, 3, dtype=dtype, device=dev)
            t_psi = torch.zeros(n, 3, dtype=dtype, device=dev)

        f_M, t_M = memory_mod.zero_memory_force(n, dtype, dev)
        f_ext = gravity_force(particles, gravity)
        return {
            "force": f_V + f_psi + f_M + f_ext,
            "torque": t_psi + t_M,
            "force_V": f_V, "force_Psi": f_psi, "force_external": f_ext,
            "torque_Psi": t_psi,
            "V": V, "V_parts": v_parts, "Psi": psi_total, "Psi_diag": psi_diag,
            "lambda_psi": lam_psi,
        }

    def _free_gaps(self, particles, contacts, v_free, dt, surfaces, t) -> torch.Tensor:
        """Gap en la posición libre provisional `q* = q + dt v*` (CCD aproximado)."""
        if contacts.n_contacts == 0:
            return contacts.gap
        with torch.no_grad():
            q_free = particles.q.detach() + dt * v_free.detach()
            R = particles.radius
            gap = torch.empty_like(contacts.gap)
            pp = ~contacts.is_wall
            if bool(pp.any()):
                i, j = contacts.i[pp], contacts.j[pp]
                d = (q_free[j] - q_free[i]).norm(dim=-1)
                gap[pp] = d - (R[i] + R[j])
            pw = contacts.is_wall
            if bool(pw.any()):
                idx = pw.nonzero().flatten()
                phi = torch.zeros(idx.numel(), dtype=gap.dtype, device=gap.device)
                for s in surfaces.surfaces:
                    m = contacts.surface[idx] == s.surface_id
                    if bool(m.any()):
                        phi[m] = s.phi(q_free[contacts.i[idx][m]], t)
                gap[idx] = phi - R[contacts.i[idx]]
            return gap

    def _impulsive_branch(
        self, particles, contacts, mode, h_node, h_edge, kin, u_free, u_n_free,
        birth, age, dt, eval_mode,
    ) -> dict:
        n = particles.n
        dtype, dev = particles.dtype, particles.device
        zero = torch.zeros(n, 3, dtype=dtype, device=dev)
        empty = torch.zeros(0, dtype=dtype, device=dev)
        mask = impulsive_mask(mode)
        if not bool(mask.any()):
            return {"force": zero, "torque": zero.clone(), "lambda_n": empty,
                    "diag": None, "params": None, "subset": None, "layout": None}

        sub = contacts.subset(mask)
        idx = mask.nonzero().flatten()
        h_I = self.proc_I(contacts, h_node, h_edge, kin)[idx]
        params = self.head_I(
            sub, particles, h_I, u_n_free[idx], u_free[idx] - u_n_free[idx].unsqueeze(-1) * sub.n,
            birth[idx], age[idx],
        )

        layout = connected_components(sub, n)
        A = assemble_normal_delassus(sub, particles.mass, particles.inertia, layout)

        # b = u_n^* + iota e min(u_n^-, 0) + beta min(g, 0)/dt
        iota = birth[idx].to(dtype)
        restitution = iota * params.e * u_n_free[idx].clamp(max=0.0)
        stabilization = self.cfg.solver.beta * sub.gap.clamp(max=0.0) / dt
        b = u_n_free[idx] + restitution + stabilization

        lam_packed, sdiag = solve_normal(
            A, pack_by_component(b, layout), pack_by_component(params.kappa, layout),
            layout, self.cfg.solver, eval_mode=eval_mode,
        )
        lam_n = unpack_by_component(lam_packed, layout)
        f_imp, t_imp = JT_times_normal_scalars(sub, lam_n, n)
        with torch.no_grad():
            a_lam = (
                J_times_velocity(
                    sub,
                    f_imp.detach() / particles.mass.unsqueeze(-1),
                    t_imp.detach() / particles.inertia.unsqueeze(-1),
                    subtract_wall=False,
                ) * sub.n
            ).sum(dim=-1)
        return {
            "force": f_imp, "torque": t_imp, "lambda_n": lam_n, "diag": sdiag,
            "params": params, "subset": sub, "layout": layout, "b": b,
            "restitution_term": restitution, "A_lambda": a_lam,
            "u_n_free": u_n_free[idx], "index": idx,
        }

    def _internal_balance(self, particles, contacts, reg, imp, dt) -> dict:
        """Conservación medida sobre los canales que aplican `J^T`.

        Solo se suman contactos **partícula–partícula**: la pared no es grado
        de libertad y su transferencia se reporta aparte. El canal `V` no
        entra aquí porque su fuerza sale de `-grad_q V` y no de `J^T`; su
        conservación se verifica estructuralmente en
        `tests/v3/test_symmetry.py`, donde el coste del autograd extra es
        aceptable.
        """
        from .contact_operator import JT_times_contact_vector

        n = particles.n
        dtype, dev = particles.dtype, particles.device
        dp = torch.zeros(n, 3, dtype=dtype, device=dev)
        dL = torch.zeros(n, 3, dtype=dtype, device=dev)
        dp_w = torch.zeros_like(dp)
        dL_w = torch.zeros_like(dL)
        with torch.no_grad():
            if contacts.n_contacts:
                lam = reg["lambda_psi"].detach()
                pp = ~contacts.is_wall
                if bool(pp.any()):
                    f, tq = JT_times_contact_vector(contacts.subset(pp), lam[pp], n)
                    dp, dL = dp + dt * f, dL + dt * tq
                if bool(contacts.is_wall.any()):
                    f, tq = JT_times_contact_vector(
                        contacts.subset(contacts.is_wall), lam[contacts.is_wall], n
                    )
                    dp_w, dL_w = dp_w + dt * f, dL_w + dt * tq
            sub = imp["subset"]
            if sub is not None and sub.n_contacts:
                lam_v = imp["lambda_n"].detach().unsqueeze(-1) * sub.n
                pp = ~sub.is_wall
                if bool(pp.any()):
                    f, tq = JT_times_contact_vector(sub.subset(pp), lam_v[pp], n)
                    dp, dL = dp + f, dL + tq
                if bool(sub.is_wall.any()):
                    f, tq = JT_times_contact_vector(
                        sub.subset(sub.is_wall), lam_v[sub.is_wall], n
                    )
                    dp_w, dL_w = dp_w + f, dL_w + tq
            return {
                **internal_momentum_error(dp, dL, particles.q.detach()),
                **wall_transfer(dp_w, dL_w),
            }

    def _diagnostics(
        self, particles, contacts, mode, reg, imp, dp_reg, dL_reg, gravity, dt,
        birth, age,
    ) -> StepDiagnostics:
        d = StepDiagnostics()
        d.energies = {
            "V_pp": reg["V_parts"]["V_pp"], "V_pW": reg["V_parts"]["V_pW"],
            "V_total": float(reg["V"].detach()),
            "V_gravity": DISABLED,   # la gravedad es fuerza externa, no potencial
        }
        d.dissipation = {**{k: v for k, v in reg["Psi_diag"].items()},
                         "Psi_tau": DISABLED, "Psi_rot": DISABLED}
        d.regular = {
            "force_V_norm": float(reg["force_V"].detach().norm()),
            "force_Psi_norm": float(reg["force_Psi"].detach().norm()),
            "force_external_norm": float(reg["force_external"].detach().norm()),
            "torque_regular_norm": float(reg["torque"].detach().norm()),
            "force_M": DISABLED, "force_C": DISABLED,
        }
        n_imp = int(imp["lambda_n"].numel())
        if n_imp and imp["params"] is not None:
            p = imp["params"]
            with torch.no_grad():
                lam = imp["lambda_n"].detach()
                # D_imp = -Lambda^T u* - (1/2) Lambda^T A Lambda  (eq. 8.17).
                # A Lambda se obtiene aplicando el operador, sin densificar J.
                a_lam = imp["A_lambda"].detach()
                d_impact = float(-(lam * imp["u_n_free"].detach()).sum()
                                 - 0.5 * (lam * a_lam).sum())
            d.impact = {
                "n_contacts": n_imp,
                "Lambda_n_mean": float(lam.mean()),
                "Lambda_n_max": float(lam.max()),
                "Lambda_n_min": float(lam.min()),
                "e_mean": float(p.e.detach().mean()), "e_min": float(p.e.detach().min()),
                "e_max": float(p.e.detach().max()),
                "kappa_mean": float(p.kappa.detach().mean()),
                "mu": DISABLED,
                "D_impact": d_impact,
                # D_impact >= 0 está garantizado solo con beta = 0. La
                # estabilización de penetración de Baumgarte inyecta energía a
                # propósito para separar contactos ya solapados; se expone su
                # magnitud para que ese aporte sea atribuible y no se confunda
                # con un fallo de pasividad del solver.
                "stabilization_term_max": float(
                    (self.cfg.solver.beta * imp["subset"].gap.detach().clamp(max=0.0)
                     / dt).abs().max()
                ),
                "beta": self.cfg.solver.beta,
                "restitution_applied": int((imp["restitution_term"].detach() != 0).sum()),
                "n_births": int(birth.sum()),
            }
            d.solver = imp["diag"].summary()
        else:
            d.impact = {"n_contacts": 0, "mu": DISABLED}
            d.solver = {"n_solves": 0}
        d.router = {
            "n_free": int((mode == int(ContactMode.FREE)).sum()),
            "n_compliant": int((mode == int(ContactMode.COMPLIANT)).sum()),
            "n_impulsive": int((mode == int(ContactMode.IMPULSIVE)).sum()),
            "contact_birth": int(birth.sum()),
            "contact_age_mean": float(age.to(torch.float64).mean()) if age.numel() else 0.0,
            "profile": self.cfg.router.profile.value,
        }
        if contacts.n_contacts:
            d.geometry = {
                "n_candidates": contacts.n_contacts,
                "n_pp": contacts.n_pp, "n_pW": contacts.n_pw,
                "gap_min": float(contacts.gap.detach().min()),
                "delta_max": float(contacts.delta.detach().max()),
                "penetration_max": float((-contacts.gap.detach()).clamp_min(0).max()),
                "n_surfaces_touched": int(contacts.surface[contacts.is_wall].unique().numel()),
            }
        else:
            d.geometry = {"n_candidates": 0, "n_pp": 0, "n_pW": 0}
        d.balance = {
            **self._internal_balance(particles, contacts, reg, imp, dt),
            "delta_p_regular_norm": float(dp_reg.detach().norm()),
            "delta_p_impulse_norm": float(imp["force"].detach().norm()),
            "delta_L_regular_norm": float(dL_reg.detach().norm()),
            "delta_L_impulse_norm": float(imp["torque"].detach().norm()),
        }
        moving = contacts.n_contacts and bool((contacts.wall_velocity != 0).any())
        d.wall = {
            "wall_velocity_max": (
                float(contacts.wall_velocity.detach().norm(dim=-1).max())
                if contacts.n_contacts else 0.0
            ),
            "regular_work": (
                float((reg["lambda_psi"].detach()
                       * contacts.wall_velocity.detach()).sum())
                if contacts.n_contacts else 0.0
            ),
            "impulsive_work": (
                float((imp["lambda_n"].detach().unsqueeze(-1)
                       * imp["subset"].n.detach()
                       * imp["subset"].wall_velocity.detach()).sum())
                if n_imp else 0.0
            ),
            "is_moving": bool(moving),
        }
        return d

    # -- rollout -----------------------------------------------------------
    def rollout(
        self,
        state: V3State,
        dt: float,
        surfaces: SurfaceSet,
        n_steps: int,
        gravity: torch.Tensor | None = None,
        eval_mode: bool = False,
        create_graph: bool | None = None,
        reset: bool = True,
    ) -> list[StepResult]:
        """Rollout autoregresivo. El tiempo real avanza en cada paso: fijar
        `t = 0` durante el rollout está prohibido (§22.12)."""
        if reset:
            self.reset_lifecycle()
        out: list[StepResult] = []
        cur = state
        for _ in range(n_steps):
            res = self.step(cur, dt, surfaces, gravity, eval_mode, create_graph)
            out.append(res)
            cur = res.next_state
        return out
