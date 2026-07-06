"""SLGNN — SDF-Lagrangian Graph Neural Network (versión 2, con rotación)."""

from .config import SLGNNConfig
from .data import (Scales, Trajectory, default_scales,
                   finite_difference_accelerations, load_case)
from .integrator import rollout, semi_implicit_step
from .model import SLGNN
from .sdf import BoxSDF, RotatingCylinderSDF, dynamical_cylinder_omega
from .state import Particles, SLGNNOutput
from . import losses

__all__ = [
    "SLGNN", "SLGNNConfig", "Particles", "SLGNNOutput",
    "BoxSDF", "RotatingCylinderSDF", "dynamical_cylinder_omega",
    "rollout", "semi_implicit_step",
    "Trajectory", "load_case", "finite_difference_accelerations",
    "Scales", "default_scales", "losses",
]
