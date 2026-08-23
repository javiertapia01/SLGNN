"""Cinemática verificada del cilindro rotatorio del dataset de extrapolación.

Este módulo existe por un hallazgo empírico que corrige una suposición vigente
en el repositorio.

`DATA_NOTES.md` dejó pendiente decidir entre dos lecturas del perfil `omega(t)`
del PDF del dataset, y `slgnn.sdf.dynamical_cylinder_omega` adoptó
provisionalmente la **triangular** (acelera 0→2, desacelera 2→0, reposo),
argumentando que la fórmula literal del PDF se vuelve negativa para t > 1 y eso
contradecía la prosa del propio PDF ("la dirección de rotación permanece
constante").

Los datos dicen lo contrario, con dos señales independientes medidas sobre
`Extrapolation_2073Spheres/CASE08`:

1. **Signo de la rotación del lecho.** La velocidad angular media de las
   partículas próximas a la pared lateral es positiva hasta t ≈ 1.0 s y pasa a
   ser claramente **negativa** entre 1.0 y 1.5 s (≈ −2.3, −3.8, −2.9 rad/s). El
   perfil triangular predice reposo en ese tramo: nada arrastraría la carga en
   sentido inverso.

2. **Spin propio de las partículas.** `|omega_i·ẑ|` tiene **dos** máximos, en
   t ≈ 0.5 s y t ≈ 1.5 s (≈ 49 y 44 rad/s), que son exactamente los dos
   extremos de `|omega(t)|` de la fórmula literal. El perfil triangular predice
   un único máximo en 0.5 s y nada después de 1.0 s.

Conclusión: la fórmula del PDF es la que usó el solver; lo equivocado es su
prosa. El tambor **invierte el sentido de giro**. Ver
`experiments/twin/exp_H_hysteresis.py`, que reproduce la verificación.

Consecuencia para SLGNN: `slgnn.sdf.dynamical_cylinder_omega` entrega una
velocidad de pared incorrecta para t > 1 s, lo que afecta directamente a
`v_W(x, t)` — el canal que la v2 introdujo precisamente para este caso — y por
tanto a cualquier rollout de extrapolación (E2-b). Se deja aquí y no se corrige
en `src/slgnn/` para no cambiar el comportamiento del modelo sin decisión
explícita del equipo.
"""

import math

import torch

from slgnn.sdf import RotatingCylinderSDF

# Geometría del dataset (DATA_NOTES.md §5), en metros.
CENTER_XY = (0.0, 0.002)
RADIUS = 0.05
Z_MIN = 0.0
Z_MAX = 0.1
DT_RECORD = 1.0e-3
GRAVITY_AXIS = "y"   # verificado: la carga sedimenta en −y, no en −z


def omega_pdf_literal(t: float) -> float:
    """Perfil del PDF tomado al pie de la letra. **Verificado contra los datos.**

    omega(t)/2pi = 4t en [0, 0.5); 4 - 4t en [0.5, 1.5]; 0 después.
    Acelera hasta +2 (unidades normalizadas) en t=0.5, cruza cero en t=1.0 e
    invierte hasta −2 en t=1.5, donde se detiene.
    """
    if t < 0.0:
        return 0.0
    if t < 0.5:
        w = 4.0 * t
    elif t <= 1.5:
        w = 4.0 - 4.0 * t
    else:
        w = 0.0
    return 2.0 * math.pi * w


def omega_triangular(t: float) -> float:
    """Lectura triangular, la que implementa `slgnn.sdf`. Se conserva solo para
    poder comparar ambas hipótesis en el mismo experimento."""
    return 2.0 * math.pi * max(min(4.0 * t, 4.0 - 4.0 * t), 0.0)


PROFILES = {"pdf_literal": omega_pdf_literal, "triangular": omega_triangular}


def make_cylinder_sdf(scaling, profile: str = "pdf_literal") -> RotatingCylinderSDF:
    """`RotatingCylinderSDF` en unidades **adimensionales**, coherente con las
    trayectorias que produce `slgnn.data.Scales.nondim` (§6)."""
    if profile not in PROFILES:
        raise ValueError(f"perfil desconocido: {profile}; opciones {sorted(PROFILES)}")
    omega_si = PROFILES[profile]

    def omega_nondim(t_nondim: float) -> float:
        return omega_si(t_nondim * scaling.T) * scaling.T

    return RotatingCylinderSDF(
        center_xy=(CENTER_XY[0] / scaling.L, CENTER_XY[1] / scaling.L),
        radius=RADIUS / scaling.L,
        z_min=Z_MIN / scaling.L,
        z_max=Z_MAX / scaling.L,
        omega_fn=omega_nondim,
    )


def verify_omega_profile(traj, scaling, *, near_wall_frac: float = 0.85,
                         n_probes: int = 21) -> dict:
    """Contrasta ambas hipótesis de `omega(t)` contra la trayectoria.

    Devuelve, por instante de sondeo, la velocidad angular media del lecho
    próximo a la pared y el spin propio medio, junto al valor que predice cada
    perfil. La discriminación no depende de un ajuste: basta el **signo** de la
    rotación del lecho entre 1.0 y 1.5 s.
    """
    q = traj.q.detach().cpu().numpy() if torch.is_tensor(traj.q) else traj.q
    v = traj.v.detach().cpu().numpy() if torch.is_tensor(traj.v) else traj.v
    w = traj.omega.detach().cpu().numpy() if torch.is_tensor(traj.omega) else traj.omega
    dt = float(traj.dt)

    cx, cy = CENTER_XY[0] / scaling.L, CENTER_XY[1] / scaling.L
    radius = RADIUS / scaling.L
    dx, dy = q[..., 0] - cx, q[..., 1] - cy
    r = (dx**2 + dy**2) ** 0.5
    omega_bed = ((-dy * v[..., 0] + dx * v[..., 1]) / (r**2).clip(1e-12)) / scaling.T
    spin = abs(w[..., 2]) / scaling.T
    near = r > near_wall_frac * radius

    T = q.shape[0]
    probes = []
    for k in range(0, T, max(T // n_probes, 1)):
        t = k * dt * scaling.T
        sel = near[k]
        probes.append({
            "t": float(t),
            "n_near_wall": int(sel.sum()),
            "omega_bed": float(omega_bed[k][sel].mean()) if sel.any() else float("nan"),
            "spin_mean": float(spin[k].mean()),
            "omega_pdf_literal": omega_pdf_literal(t),
            "omega_triangular": omega_triangular(t),
        })

    reverse = [p for p in probes if 1.05 <= p["t"] <= 1.45]
    reversed_bed = bool(reverse) and all(p["omega_bed"] < 0 for p in reverse)
    spins = [(p["t"], p["spin_mean"]) for p in probes]
    late_peak = max((s for t, s in spins if t > 1.2), default=0.0)
    early_peak = max((s for t, s in spins if t < 1.0), default=0.0)

    return {
        "probes": probes,
        "bed_reverses_between_1_0_and_1_5s": reversed_bed,
        "late_spin_peak": late_peak,
        "early_spin_peak": early_peak,
        "second_spin_peak_present": bool(late_peak > 0.5 * early_peak),
        "verdict": ("pdf_literal" if reversed_bed else "triangular"),
    }
