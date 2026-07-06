"""Configuración de SLGNN-v2.

Las longitudes están en unidades adimensionales (por defecto, diámetros de
partícula: ver `slgnn.data.default_scales`). El contacto entre dos esferas
iguales ocurre a d = 1.0 en esas unidades.
"""

from dataclasses import dataclass


@dataclass
class SLGNNConfig:
    # --- grafo partícula-partícula (§8, §10) ---
    r_on: float = 1.05    # ventana chi_pp = 1 para d <= r_on
    r_off: float = 1.5    # chi_pp = 0 para d >= r_off (C² en ambos extremos)
    r_list: float = 2.0   # radio de lista de vecinos con skin, r_list > r_off

    # --- ventana partícula-pared sobre el gap g = phi - R (§6, §10) ---
    g_on: float = 0.05
    g_off: float = 0.5

    # --- suavizados (§6, §21) ---
    beta: float = 20.0    # softplus escalada: sp_beta(x) = log(1+exp(beta x))/beta
    eps: float = 1e-8

    # --- materiales (§12) ---
    n_types: int = 1
    n_props: int = 0
    h_mat: int = 8

    # --- redes (§15) ---
    hidden: int = 32
    layers: int = 2
    rayleigh_bias_init: float = -2.0  # softplus(-2) ~ 0.13: disipación inicial suave

    # --- canales (§14) ---
    use_history: bool = True
