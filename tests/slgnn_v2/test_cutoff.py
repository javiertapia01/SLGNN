import torch

from slgnn.cutoff import neg_part, quintic_window, softplus_compression


def test_window_endpoints():
    x = torch.tensor([0.5, 1.05, 1.2, 1.5, 2.0], dtype=torch.float64)
    w = quintic_window(x, 1.05, 1.5)
    assert w[0] == 1.0 and w[1] == 1.0
    assert 0.0 < w[2] < 1.0
    assert w[3] == 0.0 and w[4] == 0.0


def test_window_monotone_decreasing():
    x = torch.linspace(1.05, 1.5, 200, dtype=torch.float64)
    w = quintic_window(x, 1.05, 1.5)
    assert (w[1:] <= w[:-1] + 1e-12).all()


def test_window_c2_smooth_at_boundaries():
    # continuidad C¹ y C² en r_on y r_off por diferencias finitas. El salto de
    # la 3ª derivada del quintic hace que el mismatch de 2ª derivada sea O(h):
    # verificar que se contrae ~10x al reducir h confirma que el límite es 0.
    for x0 in (1.05, 1.5):
        mismatches = []
        for h in (1e-4, 1e-5):
            x = torch.tensor(
                [x0 - 2 * h, x0 - h, x0, x0 + h, x0 + 2 * h], dtype=torch.float64
            )
            w = quintic_window(x, 1.05, 1.5)
            d1_left = (w[2] - w[1]) / h
            d1_right = (w[3] - w[2]) / h
            assert abs(d1_left - d1_right) < 5e-3  # C¹: mismatch O(h·f'')
            d2_left = (w[0] - 2 * w[1] + w[2]) / h**2
            d2_right = (w[2] - 2 * w[3] + w[4]) / h**2
            mismatches.append(abs(d2_left - d2_right))
        assert mismatches[1] < 0.2 * mismatches[0] + 1e-9


def test_softplus_compression_approximates_negative_part():
    g = torch.tensor([-0.5, -0.1, 0.0, 0.1, 0.5], dtype=torch.float64)
    delta = softplus_compression(g, beta=20.0)
    assert (delta >= 0).all()
    assert torch.allclose(delta[0], torch.tensor(0.5, dtype=torch.float64), atol=1e-4)
    assert delta[4] < 1e-4


def test_neg_part_active_on_approach():
    # u_n < 0 (acercamiento) activa; u_n > 0 (separación) casi nula (§21)
    u = torch.tensor([-1.0, 1.0], dtype=torch.float64)
    r = neg_part(u, beta=20.0)
    assert r[0] > 0.99
    assert r[1] < 1e-8
