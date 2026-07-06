"""Funciones de corte y suavizado C² de SLGNN-v2 (§6, §10, §21)."""

import torch
import torch.nn.functional as F


def quintic_window(x: torch.Tensor, x_on: float, x_off: float) -> torch.Tensor:
    """Ventana C² decreciente: 1 para x <= x_on, 0 para x >= x_off (§10).

    En la zona de transición vale 1 - 10s³ + 15s⁴ - 6s⁵ con
    s = (x - x_on)/(x_off - x_on); primera y segunda derivada nulas en ambos
    extremos, de modo que energías y fuerzas se anulan suavemente antes de que
    una arista desaparezca del grafo.
    """
    s = (x - x_on) / (x_off - x_on)
    s = s.clamp(0.0, 1.0)
    return 1.0 - s**3 * (10.0 - 15.0 * s + 6.0 * s * s)


def softplus_compression(g: torch.Tensor, beta: float) -> torch.Tensor:
    """Compresión suave delta = sp_beta(-g) ~ max(-g, 0) (§6)."""
    return F.softplus(-g, beta=beta)


def neg_part(x: torch.Tensor, beta: float) -> torch.Tensor:
    """Parte negativa suave rho^-(x) = sp_beta(-x) ~ max(-x, 0) (§21).

    Se usa para que la disipación normal actúe principalmente durante el
    acercamiento (u_n < 0) y no durante la separación.
    """
    return F.softplus(-x, beta=beta)
