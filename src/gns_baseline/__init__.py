"""Baselines GNS para la comparación controlada.

Dos configuraciones:

- `GNSControlled` (**comparación principal**): recibe exactamente los mismos
  datos, grafo, consultas de pared, targets, splits, sampler, semillas y
  presupuesto de actualizaciones que SLGNN-v3, y decodifica
  `(Delta p, Delta L)` directamente por nodo. Sin potencial, sin convexidad,
  sin solver, sin conservación impuesta. Es la comparación que aísla el efecto
  del sesgo inductivo, porque todo lo demás es idéntico.

- `GNSClassicReduced` (**comparación secundaria**): se acerca al GNS clásico
  —secuencia de posiciones, velocidades discretas, aristas dirigidas por
  radio, decoder de aceleración normalizada—. No es la comparación principal
  porque cambia historia, target y representación a la vez.

Ninguno importa `slgnn_v3`: comparten solo `slgnn_experiments`
(`tests/comparison/test_shared_data.py` lo verifica sobre el código fuente).

Referencia arquitectónica: Sanchez-Gonzalez et al., *Learning to Simulate
Complex Physics with Graph Networks*, ICML 2020. Esta implementación está
escrita desde cero contra esa descripción; **no** es una réplica del
repositorio `geoelements/gns` ni copia su código.
"""

from .config import GNSConfig
from .model import GNSControlled
from .classic import GNSClassicReduced

__all__ = ["GNSConfig", "GNSControlled", "GNSClassicReduced"]
