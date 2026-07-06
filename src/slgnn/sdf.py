"""Signed Distance Functions de pared móvil (§4-§5 de SLGNN-v2).

Convención (§4.1): phi > 0 en el interior admisible, phi = 0 en la pared,
y el gradiente de phi apunta hacia el interior (normal entrante).

`phi(x, t)` se implementa con operaciones torch diferenciables para que la
fuerza conservativa de pared emerja por autograd como -(dU/dphi)·grad(phi)
(§19, §30). La normal para las bases geométricas se obtiene con un pase de
autograd desacoplado (`grad_phi`), lo cual es válido porque solo se usa en
descomposiciones de velocidad (Rayleigh, canal H), nunca en el camino del
gradiente de la energía.
"""

import math

import torch


class SDF:
    """Interfaz mínima de una pared representada por SDF."""

    def phi(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Distancia con signo, [.., 3] -> [..]. Diferenciable en x."""
        raise NotImplementedError

    def wall_velocity(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Velocidad local de la pared v_W(x, t) en puntos de superficie (§5)."""
        raise NotImplementedError

    def grad_phi(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Gradiente de phi evaluado fuera del grafo de autograd principal."""
        with torch.enable_grad():
            x_ = x.detach().requires_grad_(True)
            p = self.phi(x_, t)
            (g,) = torch.autograd.grad(p.sum(), x_)
        return g

    def query(self, q: torch.Tensor, t: float, eps: float):
        """Consulta por partícula (§4.3): phi (en grafo), normal entrante nu,
        punto de pared más próximo x_W y velocidad local de pared v_W."""
        phi = self.phi(q, t)
        gamma = self.grad_phi(q, t)
        nu = gamma / (gamma.norm(dim=-1, keepdim=True) + eps)
        x_w = q.detach() - phi.detach().unsqueeze(-1) * nu
        v_w = self.wall_velocity(x_w, t)
        return phi, nu, x_w, v_w


class BoxSDF(SDF):
    """Caja rectangular estática [xmin, xmax], con pose rígida opcional.

    Con pose (Q, c): phi(x) = phi0(Q^T (x - c)) (§4.2). La caja de los
    datasets de 60 esferas es [0, 0.03]^3 m (sin pose).
    """

    def __init__(self, xmin, xmax, pose=None):
        self.xmin = torch.as_tensor(xmin, dtype=torch.float64)
        self.xmax = torch.as_tensor(xmax, dtype=torch.float64)
        if pose is not None:
            Q, c = pose
            pose = (
                torch.as_tensor(Q, dtype=torch.float64),
                torch.as_tensor(c, dtype=torch.float64),
            )
        self.pose = pose

    def _local(self, x: torch.Tensor) -> torch.Tensor:
        if self.pose is None:
            return x
        Q, c = self.pose
        # fila y = Q^T (x - c)  <=>  y_fila = (x - c)_fila @ Q
        return (x - c.to(x)) @ Q.to(x)

    def phi(self, x: torch.Tensor, t: float) -> torch.Tensor:
        y = self._local(x)
        lo = self.xmin.to(x)
        hi = self.xmax.to(x)
        d = torch.cat([y - lo, hi - y], dim=-1)  # [.., 6] distancias a caras
        return d.min(dim=-1).values

    def wall_velocity(self, x: torch.Tensor, t: float) -> torch.Tensor:
        return torch.zeros_like(x)


def dynamical_cylinder_omega(t: float) -> float:
    """Perfil omega(t) [rad/s] del cilindro del dataset Dynami-CAL.

    El PDF del dataset da omega(t) = {4t en [0,0.5); 4-4t en [0.5,1.5]; 0 después}
    (normalizado, x 2*pi). Tomado literalmente, 4-4t se hace negativo para
    t > 1, contradiciendo la afirmación del propio PDF de que la dirección de
    rotación es constante y la fase de desaceleración es simétrica. La lectura
    consistente es el perfil triangular: aceleración 0->2 en [0,0.5], simétrica
    desaceleración 2->0 en [0.5,1.0], reposo después. Documentado en
    DATA_NOTES.md; verificar contra los datos al llegar a la extrapolación.
    """
    w = min(4.0 * t, 4.0 - 4.0 * t)
    return 2.0 * math.pi * max(w, 0.0)


class RotatingCylinderSDF(SDF):
    """Cilindro vertical con pared lateral rotatoria y tapas fijas.

    Geometría del dataset de extrapolación: eje z, radio 0.05 m, centro del
    eje en (0.0, 0.002), z en [0, 0.1] m. La SDF es estacionaria (cilindro
    axisimétrico rotando sobre su propio eje: d(phi)/dt = 0), pero la pared
    lateral tiene v_W = Omega(t) x r distinto de cero — exactamente el "punto
    ciego" que la v2 resuelve con velocidad de pared explícita (§5).

    Las tapas (z = z_min, z = z_max) no rotan en el dataset, por lo que
    v_W = 0 cuando la característica más próxima es una tapa.
    """

    def __init__(self, center_xy, radius, z_min, z_max, omega_fn=dynamical_cylinder_omega):
        self.cx, self.cy = float(center_xy[0]), float(center_xy[1])
        self.radius = float(radius)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.omega_fn = omega_fn

    def _features(self, x: torch.Tensor):
        dx = x[..., 0] - self.cx
        dy = x[..., 1] - self.cy
        r = torch.sqrt(dx * dx + dy * dy + 1e-30)
        d_lat = self.radius - r
        d_bot = x[..., 2] - self.z_min
        d_top = self.z_max - x[..., 2]
        return dx, dy, d_lat, d_bot, d_top

    def phi(self, x: torch.Tensor, t: float) -> torch.Tensor:
        _, _, d_lat, d_bot, d_top = self._features(x)
        return torch.stack([d_lat, d_bot, d_top], dim=-1).min(dim=-1).values

    def wall_velocity(self, x: torch.Tensor, t: float) -> torch.Tensor:
        dx, dy, d_lat, d_bot, d_top = self._features(x)
        omega = self.omega_fn(t)
        v = torch.zeros_like(x)
        v[..., 0] = -omega * dy
        v[..., 1] = omega * dx
        lateral = (d_lat <= d_bot) & (d_lat <= d_top)
        return torch.where(lateral.unsqueeze(-1), v, torch.zeros_like(v))
