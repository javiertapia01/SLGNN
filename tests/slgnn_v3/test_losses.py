"""Pérdidas: adimensionalidad, sin duplicar garantías, y rollout por horizonte."""

import pytest
import torch

from slgnn_v3 import RouterProfile, V3State
from slgnn_v3.losses import (
    LossWeights,
    complementarity_loss,
    momentum_losses,
    next_gaps,
    penetration_loss,
    position_loss,
    rollout_loss,
    total_loss,
)

from .conftest import DTYPE, default_box, make_particles, small_model


def _step(profile=None, seed=0):
    model = small_model(profile=profile, seed=seed)
    pb = make_particles([[2.0, 2.0, 0.44], [2.0, 2.0, 1.36]],
                        v=[[0.1, 0, -0.5], [0, 0.1, 0.2]],
                        omega=[[0.2, 0, 0], [0, 0.1, 0]], radius=0.5)
    res = model.step(V3State(pb, time=0.0), 0.05, default_box(),
                     torch.tensor([0.0, 0.0, -0.98], dtype=DTYPE))
    return model, pb, res


def test_momentum_loss_is_zero_on_perfect_prediction():
    _, _, res = _step()
    l_dp, l_dL = momentum_losses(res, res.delta_p.detach(), res.delta_L.detach(),
                                 LossWeights())
    assert float(l_dp.detach()) == pytest.approx(0.0, abs=1e-24)
    assert float(l_dL.detach()) == pytest.approx(0.0, abs=1e-24)


def test_momentum_loss_scales_with_P0():
    """La pérdida es adimensional: cambiar `P0` la reescala como `1/P0^2`."""
    _, _, res = _step()
    target = torch.zeros_like(res.delta_p)
    a = momentum_losses(res, target, torch.zeros_like(res.delta_L), LossWeights(P0=1.0))[0]
    b = momentum_losses(res, target, torch.zeros_like(res.delta_L), LossWeights(P0=2.0))[0]
    assert float(a.detach()) == pytest.approx(4.0 * float(b.detach()), rel=1e-12)


def test_penetration_loss_is_zero_without_overlap():
    gaps = torch.tensor([0.1, 0.5, 0.0], dtype=DTYPE)
    assert float(penetration_loss(gaps, LossWeights())) == 0.0
    gaps = torch.tensor([-0.1, 0.0], dtype=DTYPE)
    assert float(penetration_loss(gaps, LossWeights())) > 0.0


def test_complementarity_is_diagnostic_not_a_learning_signal():
    """El solver ya garantiza `lambda >= 0`, así que la pérdida vale cero.

    §13.2: no se duplica como penalización una propiedad garantizada por
    construcción, salvo para detectar errores numéricos. Que valga cero es
    exactamente la señal de que la garantía se cumple.
    """
    _, _, res = _step(profile=RouterProfile.IMPULSIVE, seed=2)
    assert res.impulses.numel() > 0
    assert float(complementarity_loss(res, LossWeights()).detach()) == 0.0


def test_next_gaps_recomputed_from_next_state():
    _, _, res = _step()
    g = next_gaps(res)
    assert g.numel() >= 1 and torch.isfinite(g).all()


def test_total_loss_reports_every_active_part():
    _, _, res = _step()
    w = LossWeights(delta_p=1.0, delta_L=1.0, position=0.5, penetration=1.0)
    terms = total_loss(res, torch.zeros_like(res.delta_p),
                       torch.zeros_like(res.delta_L), w,
                       target_q=res.next_state.particles.q.detach(),
                       next_gap=next_gaps(res))
    assert set(terms.parts) >= {"delta_p", "delta_L", "position", "penetration", "total"}
    assert torch.isfinite(terms.total)


def test_total_loss_is_differentiable():
    model, _, res = _step()
    terms = total_loss(res, torch.zeros_like(res.delta_p),
                       torch.zeros_like(res.delta_L), LossWeights())
    terms.total.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in model.parameters())


def test_rollout_loss_weights_horizons():
    model = small_model(seed=3)
    pb = make_particles([[2.0, 2.0, 1.5]], v=[[0.1, 0.0, -0.3]], radius=0.5)
    outs = model.rollout(V3State(pb, time=0.0), 0.05, default_box(), 3,
                         create_graph=False)
    q = torch.stack([o.next_state.particles.q.detach() for o in outs])
    v = torch.stack([o.next_state.particles.v.detach() for o in outs])
    w = torch.stack([o.next_state.particles.omega.detach() for o in outs])
    perfect = rollout_loss(outs, q, v, w)
    assert float(perfect.detach()) == pytest.approx(0.0, abs=1e-24)
    shifted = rollout_loss(outs, q + 1.0, v, w,
                           weights=torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE))
    assert float(shifted.detach()) > 0.0


def test_delta_L_loss_measures_absent_physics_not_a_bug():
    """En el MVP normal `Delta L` predicho es cero, así que `L_dL` es la norma
    del target. Se reporta como física ausente, no como error de ajuste."""
    _, _, res = _step()
    target_dL = torch.full_like(res.delta_L, 0.01)
    _, l_dL = momentum_losses(res, torch.zeros_like(res.delta_p), target_dL,
                              LossWeights())
    expected = (target_dL**2).sum() / target_dL.shape[0]
    assert float(l_dL.detach()) == pytest.approx(float(expected), rel=1e-12)
