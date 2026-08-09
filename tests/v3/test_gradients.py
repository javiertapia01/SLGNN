"""Autodiferenciación: rutas conectadas, doble backward y atribución por cabeza.

Los `detach` deliberados están documentados y probados; un `detach` accidental
en el camino de `V` produciría una fuerza normal silenciosamente incorrecta,
que es justo lo que §14.3 de la formulación prohíbe.
"""

import pytest
import torch

from slgnn_v3 import RouterProfile, V3State
from slgnn_v3.potential import conservative_force

from .conftest import DTYPE, build_set, default_box, make_particles, small_model

HEADS = {
    "V": ("head_V", "proc_V"),
    "Psi": ("head_Psi", "proc_Psi"),
    "I": ("head_I", "proc_I"),
    "encoder": ("encoder",),
}


def _grad_by_head(model) -> dict[str, float]:
    out = {}
    for name, prefixes in HEADS.items():
        out[name] = sum(
            float(p.grad.abs().sum())
            for n, p in model.named_parameters()
            if p.grad is not None and n.startswith(prefixes)
        )
    return out


def _contact_state(seed=0):
    return make_particles(
        [[2.0, 2.0, 0.44], [2.0, 2.0, 1.36], [2.6, 2.1, 1.9]],
        v=[[0.1, 0, -0.5], [0, 0.1, 0.2], [-0.2, 0, -0.1]],
        omega=[[0.2, 0, 0], [0, 0.1, 0], [0, 0, 0.3]], radius=0.5,
    )


def test_gap_and_phi_stay_in_the_graph():
    """Ningún `detach` en `gap`, `delta` ni `phi` en el camino energético."""
    model = small_model(seed=1)
    pb = _contact_state().requires_grad_q()
    cs = build_set(pb, default_box(), model.cfg)
    for name in ("gap", "delta", "activation"):
        t = getattr(cs, name)
        assert t.requires_grad, f"{name} está desconectado del grafo de autograd"
    (g,) = torch.autograd.grad(cs.gap.sum(), pb.q, allow_unused=True)
    assert g is not None and float(g.abs().sum()) > 0


def test_normals_are_detached_only_where_intended():
    """El punto de pared usa `phi.detach()` a propósito: es una proyección
    geométrica, no parte del camino energético, que pasa por `gap`."""
    model = small_model(seed=2)
    pb = _contact_state().requires_grad_q()
    wall = default_box().query(pb.q, pb.radius, pb.batch_id, 0.0, 0.35)
    assert wall.gap.requires_grad          # el gap sí transmite gradiente
    assert not wall.surface_point.requires_grad


@pytest.mark.parametrize("profile,expected_silent", [
    (RouterProfile.COMPLIANT, ("I",)),
    (RouterProfile.IMPULSIVE, ("V", "Psi")),
])
def test_disabled_heads_receive_no_gradient(profile, expected_silent):
    """§16.3: ninguna cabeza recibe gradiente donde está desactivada."""
    model = small_model(profile=profile, seed=3)
    res = model.step(V3State(_contact_state(), time=0.0), 0.05, default_box(),
                     torch.tensor([0.0, 0.0, -0.98], dtype=DTYPE))
    (res.delta_p.pow(2).sum() + res.delta_L.pow(2).sum()).backward()
    grads = _grad_by_head(model)
    for head in expected_silent:
        assert grads[head] == 0.0, f"{head} recibió gradiente en {profile.value}"
    assert grads["encoder"] > 0.0


@pytest.mark.parametrize("profile,expected_active", [
    (RouterProfile.COMPLIANT, ("V", "Psi")),
    (RouterProfile.IMPULSIVE, ("I",)),
])
def test_active_heads_receive_gradient(profile, expected_active):
    model = small_model(profile=profile, seed=4)
    res = model.step(V3State(_contact_state(), time=0.0), 0.05, default_box(),
                     torch.tensor([0.0, 0.0, -0.98], dtype=DTYPE))
    (res.delta_p.pow(2).sum() + res.delta_L.pow(2).sum()).backward()
    grads = _grad_by_head(model)
    for head in expected_active:
        assert grads[head] > 0.0, f"{head} no recibió gradiente en {profile.value}"


def test_gradient_flows_through_the_rollout():
    """El estado del paso `k+1` sigue conectado al del paso `k`."""
    model = small_model(seed=5)
    # La partícula debe **entrar en contacto** durante el rollout: en vuelo
    # libre puro no hay ninguna cabeza activa y el gradiente es cero por
    # física, no por un fallo de conexión.
    pb = make_particles([[2.0, 2.0, 0.62]], v=[[0.1, 0.0, -1.0]], radius=0.5)
    outs = model.rollout(V3State(pb, time=0.0), 0.05, default_box(), 4,
                         gravity=torch.tensor([0.0, 0.0, -0.98], dtype=DTYPE),
                         create_graph=True)
    assert any(o.diagnostics.router["n_compliant"] > 0 for o in outs),         "el rollout de prueba no llegó a tocar la pared"
    outs[-1].next_state.particles.q.pow(2).sum().backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in model.parameters())


def test_double_backward_on_the_force():
    """Una pérdida definida sobre las fuerzas necesita doble backward."""
    model = small_model(seed=6)
    pb = _contact_state().requires_grad_q()
    cs = build_set(pb, default_box(), model.cfg)
    h_node, h_edge, _ = model.encoder(cs, pb)
    V, _ = model.head_V.total_potential(cs, model.proc_V(cs, h_node, h_edge),
                                        torch.ones_like(cs.gap))
    F = conservative_force(V, pb.q, create_graph=True)
    params = list(model.head_V.parameters())
    g = torch.autograd.grad(F.pow(2).sum(), params, create_graph=True,
                            allow_unused=True)
    g = [x for x in g if x is not None]
    assert g, "el doble backward no alcanzó ningún parámetro de V"
    g2 = torch.autograd.grad(sum(x.pow(2).sum() for x in g), params,
                             allow_unused=True)
    assert all(x is None or torch.isfinite(x).all() for x in g2)


def test_no_nan_gradients_in_degenerate_contact():
    """Gap exactamente cero y velocidad relativa nula: caso límite típico de
    NaN por división por cero."""
    model = small_model(seed=7)
    pb = make_particles([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], radius=0.5)
    from slgnn_v3.surfaces import SurfaceSet
    res = model.step(V3State(pb, time=0.0), 0.05, SurfaceSet([]))
    res.delta_p.pow(2).sum().backward()
    for n, p in model.parameters_and_names() if hasattr(model, "parameters_and_names") \
            else model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"gradiente no finito en {n}"
