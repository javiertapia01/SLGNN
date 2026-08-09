"""Suavizados C² separados: unilateralidad de contacto y topología del grafo.

Son dos problemas distintos y v3 los mantiene separados (§16.2 de la
formulación oficial):

- `positive_part_c2` regulariza la **condición unilateral** `g >= 0`. Debe ser
  exactamente cero en separación, o el modelo produce fuerza elástica en vuelo
  libre.
- `quintic_window` regulariza la **entrada y salida de una arista del grafo**.
  Multiplica mensajes y salidas de cabeza para que un vecino que cruza el radio
  de corte no cause un salto.
"""

from __future__ import annotations

import torch


def positive_part_c2(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Parte positiva C², **exactamente nula** para `x <= 0` (§4.5).

        p(x) = 0                            x <= 0
             = eps (6 z^3 - 8 z^4 + 3 z^5)  0 < x < eps,  z = x/eps
             = x                            x >= eps

    Continua junto con su primera y segunda derivada en `0` y en `eps`:
    `p(0)=p'(0)=p''(0)=0`, `p(eps)=eps`, `p'(eps)=1`, `p''(eps)=0`.

    Reemplaza a `softplus(-g, beta)` del legacy, que vale `log(2)/beta > 0` en
    `g = 0` y nunca se anula.
    """
    if eps <= 0:
        raise ValueError("eps debe ser positivo")
    z = (x / eps).clamp(min=0.0, max=1.0)
    ramp = eps * (z**3 * (6.0 - 8.0 * z + 3.0 * z * z))
    # Fuera de la rampa: 0 por debajo (garantizado por el clamp de z, que da
    # ramp = 0), y x por encima.
    return torch.where(x >= eps, x, ramp)


def compression(gap: torch.Tensor, eps: float) -> torch.Tensor:
    """Compresión unilateral `delta = p_eps(-g) >= 0`."""
    return positive_part_c2(-gap, eps)


def quintic_window(x: torch.Tensor, x_on: float, x_off: float) -> torch.Tensor:
    """Ventana C² decreciente: 1 para `x <= x_on`, 0 para `x >= x_off`.

    `1 - 10 s^3 + 15 s^4 - 6 s^5` con `s = (x-x_on)/(x_off-x_on)`; derivadas
    primera y segunda nulas en ambos extremos.
    """
    if x_off <= x_on:
        raise ValueError(f"x_off ({x_off}) debe ser mayor que x_on ({x_on})")
    s = ((x - x_on) / (x_off - x_on)).clamp(0.0, 1.0)
    return 1.0 - s**3 * (10.0 - 15.0 * s + 6.0 * s * s)


def gauss_legendre_01(n: int, dtype=torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    """Nodos y pesos de Gauss-Legendre sobre `[0, 1]`.

    Se registran como buffers no entrenables: la cuadratura de `U = int_0^delta f`
    debe ser una constante del modelo, no un parámetro aprendible.
    """
    import numpy as np

    x, w = np.polynomial.legendre.leggauss(n)
    return (
        torch.as_tensor(0.5 * (x + 1.0), dtype=dtype),
        torch.as_tensor(0.5 * w, dtype=dtype),
    )
