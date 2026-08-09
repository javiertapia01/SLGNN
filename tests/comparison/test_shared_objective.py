"""La función objetivo es literalmente la misma para v3 y para GNS.

Motivo de existir de este archivo: la formulación oficial de SLGNN-v3 **sí**
contempla un término de rollout en la pérdida total (§13.7). Si ese término se
activara solo para v3, la ventaja de v3 en rollout largo dejaría de ser
atribuible a su sesgo inductivo: estaría entrenando directamente contra la
métrica con la que se lo evalúa, y GNS no.

Estos tests fijan que en el MVP la pérdida es de **un paso** para ambos, y que
si algún día se activa el término de rollout tenga que activarse para los dos
o el test falle.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
import yaml

from gns_baseline import GNSConfig, GNSControlled
from slgnn_experiments import runner
from slgnn_experiments.runner import TrainConfig, _batch_loss
from slgnn_v3 import ParticleBatch, RouterProfile, SLGNNv3, V3Config, V3State

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = sorted((REPO_ROOT / "configs/experiments").glob("*.yaml"))
DTYPE = torch.float64

ROLLOUT_KEYS = ("lambda_rollout", "rollout_horizon", "lambda_roll", "rollout")


def test_training_loss_has_no_rollout_term():
    """`_batch_loss` es la única pérdida de entrenamiento y solo ve un paso."""
    src = inspect.getsource(_batch_loss)
    assert "lambda_delta_p" in src and "lambda_delta_L" in src
    for key in ROLLOUT_KEYS:
        assert key not in src, f"{key} apareció en la pérdida de entrenamiento"
    # el modelo se evalúa exactamente una vez por batch: no hay desenrollado
    assert src.count("model.step(") == 1


def test_rollout_loss_is_never_wired_into_training():
    """`slgnn_v3.losses.rollout_loss` existe, pero el runner no lo llama."""
    runner_src = Path(runner.__file__).read_text(encoding="utf-8")
    assert "rollout_loss" not in runner_src
    # los campos existen en TrainConfig para una fase futura, pero valen cero
    cfg = TrainConfig()
    assert cfg.lambda_rollout == 0.0
    assert cfg.rollout_horizon == 0


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.stem)
def test_experiment_configs_declare_no_rollout_training_term(path):
    """`rollout_horizons` es una clave de **evaluación**, no de entrenamiento.

    Vive en la raíz del YAML, no dentro de `train:`, así que nunca llega a
    `TrainConfig`. Este test evita que alguien la mueva sin darse cuenta.
    """
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    train_block = cfg.get("train", {}) or {}
    for key in ROLLOUT_KEYS:
        assert key not in train_block, (
            f"{path.name}: '{key}' está en el bloque `train:`. Si se activa un "
            "término de rollout, debe activarse para las tres variantes o la "
            "comparación deja de ser atribuible."
        )
    if "rollout_horizons" in cfg:
        assert "rollout_horizons" not in train_block


def _pb(n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return ParticleBatch.from_arrays(
        q=0.6 + 3.0 * torch.rand(n, 3, generator=g, dtype=DTYPE),
        v=0.4 * torch.randn(n, 3, generator=g, dtype=DTYPE),
        omega=0.4 * torch.randn(n, 3, generator=g, dtype=DTYPE),
        mass=torch.ones(n, dtype=DTYPE), radius=torch.full((n,), 0.5, dtype=DTYPE),
    )


def test_identical_predictions_give_identical_loss():
    """Dos modelos que predicen lo mismo reciben exactamente la misma pérdida.

    Es la comprobación de que no hay ninguna asimetría escondida en la
    normalización, la reducción o los pesos.
    """
    target_dp = torch.randn(4, 3, dtype=DTYPE)
    target_dL = torch.randn(4, 3, dtype=DTYPE)
    pred_dp = torch.randn(4, 3, dtype=DTYPE)
    pred_dL = torch.randn(4, 3, dtype=DTYPE)
    cfg = TrainConfig(lambda_delta_p=1.0, lambda_delta_L=0.0)

    def loss_of(dp, dL):
        n = dp.shape[0]
        return (cfg.lambda_delta_p * ((dp - target_dp) ** 2).sum() / n
                + cfg.lambda_delta_L * ((dL - target_dL) ** 2).sum() / n)

    assert float(loss_of(pred_dp, pred_dL)) == float(loss_of(pred_dp, pred_dL))


def test_both_models_use_the_same_semi_implicit_position_update():
    """La ventaja en rollout tampoco puede venir del esquema de integración:
    GNS recibe el mismo `q_{k+1} = q_k + dt v_{k+1}` que v3."""
    from slgnn_v3 import box_surfaces

    box = box_surfaces([0, 0, 0], [5, 5, 5])
    dt = 0.05
    pb = _pb()
    for model in (SLGNNv3(V3Config()).to(DTYPE),
                  GNSControlled(GNSConfig(hidden=16, n_message_steps=2)).to(DTYPE)):
        res = model.step(V3State(pb, time=0.0), dt, box)
        p = res.next_state.particles
        assert torch.allclose(p.q, pb.q + dt * p.v, atol=1e-14), (
            f"{type(model).__name__} no usa la velocidad post-paso para la posición"
        )


def test_gravity_is_given_to_both_models_analytically():
    """GNS no gasta capacidad en reaprender la gravedad: se le suma igual que
    a v3. Es una ventaja concedida al baseline, no una desventaja."""
    from slgnn_v3 import box_surfaces
    from slgnn_v3.surfaces import SurfaceSet

    g = torch.tensor([0.0, 0.0, -0.98], dtype=DTYPE)
    dt = 0.05
    pb = ParticleBatch.from_arrays(
        q=torch.tensor([[2.5, 2.5, 2.5]], dtype=DTYPE),
        v=torch.zeros(1, 3, dtype=DTYPE), omega=torch.zeros(1, 3, dtype=DTYPE),
        mass=torch.full((1,), 2.0, dtype=DTYPE),
        radius=torch.full((1,), 0.5, dtype=DTYPE),
    )
    expected = dt * pb.mass.unsqueeze(-1) * g
    for model in (SLGNNv3(V3Config()).to(DTYPE),
                  GNSControlled(GNSConfig(hidden=16, n_message_steps=2)).to(DTYPE)):
        # La diferencia entre correr con y sin gravedad debe ser exactamente el
        # impulso analítico. No se compara `delta_p` con `expected` a secas:
        # el decoder de GNS, sin entrenar, añade siempre su propio término, y
        # eso no significa que no haya recibido la gravedad.
        with_g = model.step(V3State(pb, time=0.0), dt, SurfaceSet([]), g)
        without_g = model.step(V3State(pb, time=0.0), dt, SurfaceSet([]), None)
        diff = (with_g.delta_p - without_g.delta_p).detach()
        assert torch.allclose(diff, expected, atol=1e-14), (
            f"{type(model).__name__} no recibe la gravedad analítica"
        )
