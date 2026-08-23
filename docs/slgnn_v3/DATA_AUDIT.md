# DATA_AUDIT — auditoría temporal del dataset Dynami-CAL

Generado por `scripts/slgnn_v3/audit_temporal.py`. **No editar a mano**: vuelve a
correr el script si cambia el dataset o la banda de detección.

- Fecha de la corrida: `2026-08-09T03:02:45+00:00`
- Commit: `d6e5f4b47094163084da0508bdb9183da940bd41` (dirty: True)
- Banda de candidatos: `0.1 x diámetro`

## 0. Qué no contiene el dataset

Los CSV traen únicamente `q`, `v`, `omega`, `Density` y (a veces) `Diameter`.
**No hay** fuerzas de contacto, impulsos por contacto, ni los subpasos internos
del solver MFiX (`dt_DEM = 1e-7 s` según la documentación del dataset). Por eso
toda la clasificación de régimen que sigue es geométrica y cinemática.

## 1. Verificación de `dt` y de la gravedad contra los datos

`dt` se re-estima por mínimos cuadrados con la regla del punto medio
`q_{k+1} - q_k = dt (v_k + v_{k+1})/2` sobre transiciones en vuelo libre.
La gravedad es la aceleración media de esas mismas transiciones.

| Caso | dt doc [s] | dt estimado [s] | error rel. | g medido [m/s²] | eje |
|---|---|---|---|---|---|
| `two_spheres/1x` | 1.0e-04 | 1.0003e-04 | 3.00e-04 | 0.000 | x |
| `two_spheres/2x` | 1.0e-04 | 1.0029e-04 | 2.94e-03 | 0.000 | x |
| `two_spheres/4x` | 1.0e-04 | 1.0002e-04 | 2.49e-04 | 0.000 | x |
| `one_sphere_wall/10` | 1.0e-04 | 1.0006e-04 | 6.24e-04 | 0.000 | x |
| `one_sphere_wall/30` | 1.0e-04 | 1.0008e-04 | 7.55e-04 | 0.000 | x |
| `one_sphere_wall/45` | 1.0e-04 | 1.0006e-04 | 6.46e-04 | 0.000 | x |
| `one_sphere_wall/60` | 1.0e-04 | 1.0007e-04 | 7.01e-04 | 0.000 | x |
| `one_sphere_wall/90` | 1.0e-04 | 1.0006e-04 | 5.92e-04 | 0.000 | x |
| `sixty_gravity/CASE01` | 1.0e-04 | 1.0001e-04 | 9.06e-05 | 9.807 | y |
| `sixty_gravity/CASE02` | 1.0e-04 | 1.0001e-04 | 7.31e-05 | 9.807 | y |
| `sixty_gravity/CASE06` | 1.0e-04 | 1.0001e-04 | 1.50e-04 | 9.806 | y |
| `sixty_gravity/CASE07` | 1.0e-04 | 1.0003e-04 | 3.01e-04 | 9.807 | y |
| `sixty_homogeneous/CASE01` | 1.0e-04 | 1.0000e-04 | 4.52e-05 | 0.000 | y |
| `sixty_homogeneous/CASE06` | 1.0e-04 | 1.0004e-04 | 4.14e-04 | 0.000 | z |
| `rotating_cylinder/CASE08` | 1.0e-03 | 9.9916e-04 | 8.44e-04 | 5.171 | y |

## 2. Episodios de contacto por clave estable

Un episodio es un run contiguo de snapshots con `g <= 0` para la misma clave
`(i,j)` o `(i, cara)`. La mediana en snapshots es la magnitud que decide el
perfil: si vale 1, el intervalo de grabación contiene el choque entero.

| Caso | episodios | mediana [snap] | % de 1 snap | pico interior | ρ_t | pen. máx / R |
|---|---|---|---|---|---|---|
| `two_spheres/1x` | 1 | 11 | 0% | 100% | 0.09 | 0.195 |
| `two_spheres/2x` | 1 | 11 | 0% | 100% | 0.09 | 0.452 |
| `two_spheres/4x` | 1 | 11 | 0% | 100% | 0.09 | 0.995 |
| `one_sphere_wall/10` | 1 | 16 | 0% | 100% | 0.06 | 0.380 |
| `one_sphere_wall/30` | 1 | 16 | 0% | 100% | 0.06 | 0.336 |
| `one_sphere_wall/45` | 1 | 16 | 0% | 100% | 0.06 | 0.273 |
| `one_sphere_wall/60` | 1 | 16 | 0% | 100% | 0.06 | 0.194 |
| `one_sphere_wall/90` | 1 | 16 | 0% | 100% | 0.06 | 0.387 |
| `sixty_gravity/CASE01` | 2035 | 4 | 0% | 72% | 0.25 | 0.026 |
| `sixty_gravity/CASE02` | 1518 | 4.0 | 0% | 77% | 0.25 | 0.090 |
| `sixty_gravity/CASE06` | 1831 | 4 | 1% | 74% | 0.25 | 0.148 |
| `sixty_gravity/CASE07` | 1709 | 4 | 0% | 74% | 0.25 | 0.257 |
| `sixty_homogeneous/CASE01` | 1076 | 12.0 | 0% | 85% | 0.08 | 0.398 |
| `sixty_homogeneous/CASE06` | 1156 | 12.0 | 0% | 86% | 0.08 | 0.446 |
| `rotating_cylinder/CASE08` | 39333 | 1 | 77% | 46% | 1.00 | 0.052 |

`ρ_t = dt_snapshot / t_contacto`. `ρ_t << 1` es contacto resuelto (v3-C);
`ρ_t >= 1` es contacto submuestreado (v3-I).

### 2b. Contraste con la duración analítica del contacto DEM

Si la detección geométrica está midiendo el contacto real y no ruido de
discretización, la duración de episodio debe coincidir con el semiperiodo
del oscilador masa–resorte del DEM, `t_c = π sqrt(m_eff/k_n)`, con la
rigidez `k_n` documentada en `data/DATA_NOTES.md` y `m_eff = m/2` para
partícula–partícula, `m_eff = m` para partícula–pared.

| Caso | k_n [N/m] | t_c pp [snap] | t_c pw [snap] | mediana pp medida | mediana pw medida |
|---|---|---|---|---|---|
| `two_spheres/1x` | 1000.0 | 11.4 | 16.1 | 11.0 | — |
| `two_spheres/2x` | 1000.0 | 11.4 | 16.1 | 11.0 | — |
| `two_spheres/4x` | 1000.0 | 11.4 | 16.1 | 11.0 | — |
| `one_sphere_wall/10` | 1000.0 | 11.4 | 16.1 | — | 16.0 |
| `one_sphere_wall/30` | 1000.0 | 11.4 | 16.1 | — | 16.0 |
| `one_sphere_wall/45` | 1000.0 | 11.4 | 16.1 | — | 16.0 |
| `one_sphere_wall/60` | 1000.0 | 11.4 | 16.1 | — | 16.0 |
| `one_sphere_wall/90` | 1000.0 | 11.4 | 16.1 | — | 16.0 |
| `sixty_gravity/CASE01` | 10000.0 | 3.6 | 5.1 | 4.0 | 5.0 |
| `sixty_gravity/CASE02` | 10000.0 | 3.6 | 5.1 | 4.0 | 5.0 |
| `sixty_gravity/CASE06` | 10000.0 | 3.6 | 5.1 | 4.0 | 5.0 |
| `sixty_gravity/CASE07` | 10000.0 | 3.6 | 5.1 | 4.0 | 5.0 |
| `sixty_homogeneous/CASE01` | 1000.0 | 11.4 | 16.1 | 11.0 | 16.0 |
| `sixty_homogeneous/CASE06` | 1000.0 | 11.4 | 16.1 | 11.0 | 16.0 |
| `rotating_cylinder/CASE08` | 10000.0 | 0.4 | 0.5 | 1.0 | — |

La coincidencia entre columnas analíticas y medidas es la validación de
que esta auditoría mide física y no artefactos. Una discrepancia grande
indicaría o un `k_n` mal documentado o un detector de contacto roto.

## 3. Nacimiento, persistencia y ruptura

| Caso | nacimientos | muertes | persistencias | % nacimientos sobre activos |
|---|---|---|---|---|
| `two_spheres/1x` | 1 | 1 | 10 | 9.1% |
| `two_spheres/2x` | 1 | 1 | 10 | 9.1% |
| `two_spheres/4x` | 1 | 1 | 10 | 9.1% |
| `one_sphere_wall/10` | 1 | 1 | 15 | 6.2% |
| `one_sphere_wall/30` | 1 | 1 | 15 | 6.2% |
| `one_sphere_wall/45` | 1 | 1 | 15 | 6.2% |
| `one_sphere_wall/60` | 1 | 1 | 15 | 6.2% |
| `one_sphere_wall/90` | 1 | 1 | 15 | 6.2% |
| `sixty_gravity/CASE01` | 2035 | 2027 | 6793 | 23.1% |
| `sixty_gravity/CASE02` | 1518 | 1512 | 5080 | 23.0% |
| `sixty_gravity/CASE06` | 1831 | 1828 | 6071 | 23.2% |
| `sixty_gravity/CASE07` | 1709 | 1707 | 5725 | 23.0% |
| `sixty_homogeneous/CASE01` | 1076 | 1071 | 15436 | 6.5% |
| `sixty_homogeneous/CASE06` | 1156 | 1156 | 17525 | 6.2% |
| `rotating_cylinder/CASE08` | 39333 | 38382 | 16767 | 70.1% |

## 4. Saltos de velocidad condicionados a contacto

| Caso | ‖Δv‖ libre (media) | ‖Δv‖ contacto (media) | razón | ‖Δv‖ contacto (máx) |
|---|---|---|---|---|
| `two_spheres/1x` | 0.000e+00 | 1.119e-01 | — | 2.122e-01 |
| `two_spheres/2x` | 0.000e+00 | 2.581e-01 | — | 4.657e-01 |
| `two_spheres/4x` | 0.000e+00 | 3.801e-01 | — | 1.064e+00 |
| `one_sphere_wall/10` | 0.000e+00 | 2.211e-01 | — | 4.173e-01 |
| `one_sphere_wall/30` | 0.000e+00 | 1.949e-01 | — | 3.657e-01 |
| `one_sphere_wall/45` | 0.000e+00 | 1.588e-01 | — | 2.997e-01 |
| `one_sphere_wall/60` | 0.000e+00 | 1.123e-01 | — | 2.088e-01 |
| `one_sphere_wall/90` | 0.000e+00 | 2.238e-01 | — | 4.205e-01 |
| `sixty_gravity/CASE01` | 9.807e-04 | 2.703e-02 | 27.6x | 2.728e-01 |
| `sixty_gravity/CASE02` | 9.807e-04 | 8.386e-02 | 85.5x | 9.760e-01 |
| `sixty_gravity/CASE06` | 9.807e-04 | 1.145e-01 | 116.8x | 1.616e+00 |
| `sixty_gravity/CASE07` | 9.807e-04 | 2.037e-01 | 207.7x | 2.758e+00 |
| `sixty_homogeneous/CASE01` | 4.182e-08 | 4.723e-02 | 1129371.5x | 3.394e-01 |
| `sixty_homogeneous/CASE06` | 7.676e-09 | 5.119e-02 | 6668800.7x | 4.151e-01 |
| `rotating_cylinder/CASE08` | 3.520e-02 | 1.132e-01 | 3.2x | 1.373e+00 |

## 5. Distribución de gaps y separación pp / pw

| Caso | pp: mediana | pp: mín | % pp<0 | pw: mediana | pw: mín | % pw<0 |
|---|---|---|---|---|---|---|
| `two_spheres/1x` | -1.07e-04 | -4.87e-04 | 57.89% | — | — | — |
| `two_spheres/2x` | -5.07e-04 | -1.13e-03 | 73.33% | — | — | — |
| `two_spheres/4x` | -1.42e-03 | -2.49e-03 | 91.67% | — | — | — |
| `one_sphere_wall/10` | — | — | — | 1.26e-02 | -9.50e-04 | 7.96% |
| `one_sphere_wall/30` | — | — | — | 1.08e-02 | -8.40e-04 | 7.96% |
| `one_sphere_wall/45` | — | — | — | 8.43e-03 | -6.83e-04 | 7.96% |
| `one_sphere_wall/60` | — | — | — | 5.31e-03 | -4.85e-04 | 7.96% |
| `one_sphere_wall/90` | — | — | — | 1.29e-02 | -9.68e-04 | 7.96% |
| `sixty_gravity/CASE01` | 2.27e-04 | -6.61e-05 | 2.95% | 1.10e-03 | -5.70e-05 | 4.32% |
| `sixty_gravity/CASE02` | 2.42e-04 | -2.24e-04 | 5.05% | 1.37e-03 | -1.89e-04 | 2.97% |
| `sixty_gravity/CASE06` | 2.35e-04 | -2.37e-04 | 6.74% | 1.38e-03 | -3.69e-04 | 3.45% |
| `sixty_gravity/CASE07` | 2.43e-04 | -5.97e-04 | 9.23% | 1.43e-03 | -6.42e-04 | 3.42% |
| `sixty_homogeneous/CASE01` | 2.36e-04 | -7.32e-04 | 16.35% | 1.82e-03 | -9.96e-04 | 6.57% |
| `sixty_homogeneous/CASE06` | 2.31e-04 | -8.51e-04 | 18.67% | 1.64e-03 | -1.12e-03 | 7.21% |
| `rotating_cylinder/CASE08` | 1.83e-04 | -1.31e-04 | 6.11% | — | — | — |

## 6. Contactos múltiples con la pared (aristas y esquinas)

Número de caras de la caja simultáneamente en contacto con una misma
partícula, contado sobre todos los pares (frame, partícula):

| Caso | 0 caras | 1 cara | 2 caras | 3 caras |
|---|---|---|---|---|
| `two_spheres/1x` | 0 | 0 | 0 | 0 |
| `two_spheres/2x` | 0 | 0 | 0 | 0 |
| `two_spheres/4x` | 0 | 0 | 0 | 0 |
| `one_sphere_wall/10` | 185 | 16 | 0 | 0 |
| `one_sphere_wall/30` | 185 | 16 | 0 | 0 |
| `one_sphere_wall/45` | 185 | 16 | 0 | 0 |
| `one_sphere_wall/60` | 185 | 16 | 0 | 0 |
| `one_sphere_wall/90` | 185 | 16 | 0 | 0 |
| `sixty_gravity/CASE01` | 86126 | 3904 | 30 | 0 |
| `sixty_gravity/CASE02` | 87382 | 2646 | 32 | 0 |
| `sixty_gravity/CASE06` | 86946 | 3056 | 58 | 0 |
| `sixty_gravity/CASE07` | 86973 | 3066 | 21 | 0 |
| `sixty_homogeneous/CASE01` | 84144 | 5690 | 226 | 0 |
| `sixty_homogeneous/CASE06` | 83567 | 6105 | 354 | 34 |
| `rotating_cylinder/CASE08` | 0 | 0 | 0 | 0 |

Cualquier entrada distinta de cero en las columnas de 2 y 3 caras refuta
`min` sobre caras como interfaz única de contacto (§4.2 de las instrucciones).

## 7. Ventanas recomendadas para micro-overfit

Índices de transición `k` por régimen, listos para fijar un conjunto
versionado pequeño:

```json
{
  "two_spheres/1x": {
    "free": [
      0,
      1,
      2,
      3,
      4,
      5
    ],
    "pw_birth": [],
    "pw_persistent": [],
    "pp_contact": [
      31,
      32,
      33,
      34,
      35,
      36
    ],
    "mixed": []
  },
  "two_spheres/2x": {
    "free": [
      0,
      1,
      2,
      3,
      4,
      5
    ],
    "pw_birth": [],
    "pw_persistent": [],
    "pp_contact": [
      14,
      15,
      16,
      17,
      18,
      19
    ],
    "mixed": []
  },
  "two_spheres/4x": {
    "free": [
      0,
      1,
      2,
      3,
      4,
      5
    ],
    "pw_birth": [],
    "pw_persistent": [],
    "pp_contact": [
      8,
      9,
      10,
      11,
      12,
      13
    ],
    "mixed": []
  },
  "one_sphere_wall/10": {
    "free": [
      0,
      1,
      2,
      3,
      4,
      5
    ],
    "pw_birth": [
      12
    ],
    "pw_persistent": [
      13,
      14,
      15,
      16,
      17,
      18
    ],
    "pp_contact": [],
    "mixed": []
  },
  "one_sphere_wall/30": {
    "free": [
      0,
      1,
      2,
      3,
      4,
      5
    ],
    "pw_birth": [
      14
    ],
    "pw_persistent": [
      15,
      16,
      17,
      18,
      19,
      20
    ],
    "pp_contact": [],
    "mixed": []
  },
  "one_sphere_wall/45": {
    "free": [
      0,
      1,
      2,
      3,
      4,
      5
    ],
    "pw_birth": [
      17
    ],
    "pw_persistent": [
      18,
      19,
      20,
      21,
      22,
      23
    ],
    "pp_contact": [],
    "mixed": []
  },
  "one_sphere_wall/60": {
    "free": [
      0,
      1,
      2,
      3,
      4,
      5
    ],
    "pw_birth": [
      24
    ],
    "pw_persistent": [
      25,
      26,
      27,
      28,
      29,
      30
    ],
    "pp_contact": [],
    "mixed": []
  },
  "one_sphere_wall/90": {
    "free": [
      0,
      1,
      2,
      3,
      4,
      5
    ],
    "pw_birth": [
      12
    ],
    "pw_persistent": [
      13,
      14,
      15,
      16,
      17,
      18
    ],
    "pp_contact": [],
    "mixed": []
  },
  "sixty_gravity/CASE01": {
    "free": [
      0,
      1,
      2,
      3,
      4,
      5
    ],
    "pw_birth": [
      190,
      207,
      208,
      209,
      211,
      212
    ],
    "pw_persistent": [
      191,
      192,
      193,
      194,
      208,
      209
    ],
    "pp_contact": [
      201,
      202,
      203,
      204,
      205,
      207
    ],
    "mixed": [
      208,
      209,
      210,
      211,
      212,
      213
    ]
  },
  "sixty_gravity/CASE02": {
    "free": [
      0,
      1,
      2,
      3,
      4,
      5
    ],
    "pw_birth": [
      90,
      91,
      111,
      113,
      115,
      117
    ],
    "pw_persistent": [
      91,
      92,
      93,
      94,
      95,
      96
    ],
    "pp_contact": [
      95,
      96,
      97,
      98,
      99,
      100
    ],
    "mixed": [
      95,
      96,
      97,
      98,
      99,
      100
    ]
  },
  "sixty_gravity/CASE06": {
    "free": [
      0,
      1,
      2,
      3,
      4,
      5
    ],
    "pw_birth": [
      62,
      69,
      74,
      75,
      76,
      77
    ],
    "pw_persistent": [
      63,
      64,
      65,
      66,
      67,
      68
    ],
    "pp_contact": [
      67,
      68,
      69,
      70,
      71,
      72
    ],
    "mixed": [
      67,
      68,
      69,
      70,
      71,
      72
    ]
  },
  "sixty_gravity/CASE07": {
    "free": [
      0,
      1,
      2,
      3,
      4,
      5
    ],
    "pw_birth": [
      25,
      30,
      41,
      42,
      44,
      45
    ],
    "pw_persistent": [
      26,
      27,
      28,
      29,
      30,
      31
    ],
    "pp_contact": [
      29,
      30,
      31,
      32,
      33,
      34
    ],
    "mixed": [
      29,
      30,
      31,
      32,
      33,
      34
    ]
  },
  "sixty_homogeneous/CASE01": {
    "free": [
      0,
      1,
      2,
      3,
      4,
      5
    ],
    "pw_birth": [
      77,
      90,
      91,
      93,
      121,
      122
    ],
    "pw_persistent": [
      78,
      79,
      80,
      81,
      82,
      83
    ],
    "pp_contact": [
      85,
      86,
      87,
      88,
      89,
      90
    ],
    "mixed": [
      85,
      86,
      87,
      88,
      89,
      90
    ]
  },
  "sixty_homogeneous/CASE06": {
    "free": [
      0,
      1,
      2,
      3,
      4,
      5
    ],
    "pw_birth": [
      60,
      77,
      78,
      85,
      86,
      87
    ],
    "pw_persistent": [
      61,
      62,
      63,
      64,
      65,
      66
    ],
    "pp_contact": [
      68,
      69,
      70,
      71,
      72,
      73
    ],
    "mixed": [
      68,
      69,
      70,
      71,
      72,
      73
    ]
  },
  "rotating_cylinder/CASE08": {
    "free": [
      0,
      1,
      2,
      3,
      4
    ],
    "pw_birth": [],
    "pw_persistent": [],
    "pp_contact": [
      6,
      7,
      8,
      9,
      10,
      11
    ],
    "mixed": []
  }
}
```

## 8. Recomendación provisional de perfil

**v3-C** — los solapamientos duran varios snapshots y el pico de compresión cae dentro del episodio: el contacto está temporalmente resuelto.

- mediana de duración de episodio: `11` snapshots
- fracción media de episodios de un solo snapshot: `5.3%`
- fracción media con pico de compresión interior: `87.6%`
- mediana de `ρ_t`: `0.09`

> El dataset no contiene fuerzas de contacto, impulsos ni los subpasos internos del solver (dt_DEM = 1e-7 s). La duración de contacto se infiere de solapamiento geométrico entre snapshots grabados, que es una cota inferior: un choque puede empezar y terminar dentro de un intervalo sin dejar ningún frame con g < 0.

## 9. Comparación con las estimaciones previas del repositorio

`data/DATA_NOTES.md` documenta `dt = 1e-4 s` para caja y benchmarks y
`1e-3 s` para el cilindro, con `dt` interno del solver `1e-7 s`. La
estimación por punto medio de la sección 1 confirma o refuta ese valor
caso por caso; cualquier discrepancia mayor que `1e-3` relativa aparece
en la columna de error de esa tabla y debe resolverse antes de entrenar.

El informe previo `docs/slgnn_v2/training/Informe_Estrategia_Entrenamiento_SLGNN.md` asumía un
régimen compliant implícito al usar aceleración como target. Esta
auditoría reemplaza ese supuesto: la decisión ahora se toma sobre la
mediana de duración de episodio medida arriba, no sobre la conveniencia
de la parametrización.
