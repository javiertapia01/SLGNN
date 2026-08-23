"""Contratos de `ParticleBatch` y `V3State`."""

import pytest
import torch

from slgnn_v3 import ContactMemoryState, ParticleBatch, V3State
from slgnn_v3.state import ContactMemoryState as CMS

from .conftest import DTYPE, make_particles, random_particles


def test_shapes_and_positivity():
    pb = random_particles(5)
    pb.validate()
    assert pb.n == 5 and pb.n_systems == 1
    assert torch.allclose(pb.inertia, 0.4 * pb.mass * pb.radius**2)


@pytest.mark.parametrize("field", ["mass", "radius", "inertia"])
def test_non_positive_rejected(field):
    pb = random_particles(3)
    bad = getattr(pb, field).clone()
    bad[1] = 0.0
    with pytest.raises(ValueError, match="positivo"):
        pb.replace(**{field: bad}).validate()


def test_nan_rejected():
    pb = random_particles(3)
    q = pb.q.clone()
    q[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        pb.replace(q=q).validate()


def test_mixed_dtype_rejected():
    pb = random_particles(3)
    with pytest.raises(ValueError, match="dtypes"):
        pb.replace(v=pb.v.to(torch.float32)).validate()


def test_concat_reindexes_batch_id():
    a = random_particles(3, seed=1)
    b = random_particles(4, seed=2)
    both = ParticleBatch.concat([a, b])
    assert both.n == 7
    assert both.n_systems == 2
    assert both.batch_id.tolist() == [0, 0, 0, 1, 1, 1, 1]


def test_time_is_real_and_propagates():
    pb = random_particles(2)
    st = V3State(pb, time=0.37)
    assert st.time_scalar() == pytest.approx(0.37)


def test_memory_cannot_be_silently_enabled():
    """La memoria M no está implementada: activarla debe fallar, no simularse."""
    assert CMS.empty().enabled is False
    with pytest.raises(NotImplementedError, match="memoria tangencial"):
        ContactMemoryState(enabled=True)


def test_requires_grad_q_is_a_leaf():
    pb = random_particles(3).requires_grad_q()
    assert pb.q.requires_grad and pb.q.is_leaf
