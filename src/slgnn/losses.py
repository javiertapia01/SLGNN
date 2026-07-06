"""Pérdidas de entrenamiento de SLGNN-v2 (§34)."""

import torch
import torch.nn.functional as F

from .cutoff import softplus_compression


def acceleration_loss(a_hat, a_ref, sigma: float = 1.0):
    """L_a (§34.1): MSE normalizado de aceleración translacional."""
    return ((a_hat - a_ref) / sigma).pow(2).sum(dim=-1).mean()


def angular_acceleration_loss(alpha_hat, alpha_ref, sigma: float = 1.0):
    """L_alpha (§34.2)."""
    return ((alpha_hat - alpha_ref) / sigma).pow(2).sum(dim=-1).mean()


def rollout_loss(q_hat, q_ref, v_hat, v_ref, omega_hat=None, omega_ref=None,
                 sigma_q: float = 1.0, sigma_v: float = 1.0,
                 sigma_w: float = 1.0, lambda_v: float = 1.0,
                 lambda_w: float = 1.0, weights=None):
    """L_roll (§34.3) sobre un horizonte [H+1, N, 3] (incluye estado inicial)."""
    h = q_hat.shape[0] - 1
    if weights is None:
        weights = torch.ones(h, dtype=q_hat.dtype, device=q_hat.device)
    loss = q_hat.new_zeros(())
    for s in range(1, h + 1):
        term = ((q_hat[s] - q_ref[s]) / sigma_q).pow(2).sum(dim=-1).mean()
        term = term + lambda_v * ((v_hat[s] - v_ref[s]) / sigma_v).pow(2).sum(dim=-1).mean()
        if omega_hat is not None:
            term = term + lambda_w * (
                (omega_hat[s] - omega_ref[s]) / sigma_w
            ).pow(2).sum(dim=-1).mean()
        loss = loss + weights[s - 1] * term
    return loss


def residual_regularization(f_hist, tau_hist):
    """L_res (§34.5): fuerza al modelo a explicar primero con V y Rayleigh."""
    return f_hist.pow(2).sum(dim=-1).mean() + tau_hist.pow(2).sum(dim=-1).mean()


def passivity_loss(p_hist_pp, p_hist_pw):
    """L_pass (§34.6): penaliza potencia positiva del residual histórico."""
    loss = p_hist_pp.new_zeros(())
    if p_hist_pp.numel():
        loss = loss + F.relu(p_hist_pp).pow(2).sum()
    if p_hist_pw.numel():
        loss = loss + F.relu(p_hist_pw).pow(2).sum()
    return loss


def penetration_loss(phi, radii, beta: float):
    """L_pen (§34.7): regularización de no penetración durante rollout."""
    return softplus_compression(phi - radii, beta).pow(2).mean()
