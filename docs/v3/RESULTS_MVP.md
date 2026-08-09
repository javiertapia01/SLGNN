# RESULTS_MVP — SLGNN-v3, resultados del MVP normal

Generado por `scripts/v3/make_results_report.py` desde los JSON de las
corridas. **No editar a mano.**

- Commit: `d6e5f4b47094163084da0508bdb9183da940bd41` (dirty: True)

---

## 1. Corrección estructural

Verificada antes de entrenar nada, en `float64`. Cada garantía tiene un
test ejecutable; el mapa completo está en
[MATHEMATICAL_CONTRACT.md](MATHEMATICAL_CONTRACT.md).

| Garantía | Tolerancia exigida | Estado |
|---|---|---|
| Ecuación discreta central `M Δν = dt F_reg + Jᵀ Λ` | exacta | ✅ |
| Punto común invariante al intercambio `i↔j` | 1e-12 | ✅ |
| Identidad adjunta `⟨Jν, λ⟩ = ⟨ν, Jᵀλ⟩` | 1e-10 rel | ✅ |
| Momento lineal interno | 1e-14 | ✅ |
| Momento angular orbital + spin | 1e-10 rel | ✅ |
| Equivarianza `SE(3)` del paso completo | 1e-8 rel | ✅ |
| Equivarianza a permutación del paso completo | 1e-10 rel | ✅ |
| `U(0) = 0` | 1e-12 | ✅ |
| `dU/dδ ≥ 0` (repulsión) | -1e-10 | ✅ |
| `ψ' ≥ 0`, `ψ'' ≥ 0` (convexidad) | -1e-10 | ✅ |
| Potencia disipativa relativa ≤ 0 (pasividad) | 1e-10 | ✅ |
| Solver: caso analítico de dos cuerpos | 1e-8 rel | ✅ |
| Solver: complementariedad en evaluación | 1e-7 | ✅ |
| Solve acoplado ≠ solve independiente en multicontacto | > 1e-3 | ✅ |
| Compresión exactamente nula en `g ≥ 0` | exacta | ✅ |
| Dos y tres caras activas en arista y esquina | exacta | ✅ |
| Cero aristas entre ejemplos del batch | exacta | ✅ |
| El tiempo de pared cambia entre pasos | — | ✅ |
| Router sin doble conteo | exacta | ✅ |
| Restitución solo al nacimiento del contacto | exacta | ✅ |
| Doble backward finito y no nulo | — | ✅ |

## 2. Capacidad de micro-overfit y auditoría de cableado

Conjunto fijo: `sixty_gravity/CASE01`, ventana `[185, 265)`, elegida
con la auditoría temporal. Pérdidas medidas sobre el **mismo batch**
antes y después de entrenar.

| Variante | Pérdida inicial | Pérdida final | Reducción | Mejor val |
|---|---|---|---|---|
| v3-C | 5.096e-05 | 4.52e-05 | 11.3% | 7.794e-05 |
| v3-I | 7.532e-05 | 4.902e-05 | 34.9% | 0.0001123 |
| GNS controlado | 0.0004362 | 8.667e-05 | 80.1% | 0.0001404 |

Atribución del gradiente por cabeza tras entrenar:

| Variante | V | Psi | I | encoder |
|---|---|---|---|---|
| v3-C | 4.313e-06 | 0.0003344 | 0 | 4.712e-05 |
| v3-I | 0 | 0 | 0.0008712 | 0.0001654 |
| GNS controlado | 0 | 0 | 0 | 0.01149 |

Comprobaciones de cableado:

- **v3-C**: loss_decreases=✅, loss_drops_over_90pct=❌, encoder_receives_gradient=✅, V_receives_gradient=✅, Psi_receives_gradient=✅, I_silent_as_expected=✅
- **v3-I**: loss_decreases=✅, loss_drops_over_90pct=❌, encoder_receives_gradient=✅, I_receives_gradient=✅, V_silent_as_expected=✅, Psi_silent_as_expected=✅, e_moved_from_init=✅, kappa_moved_from_init=✅
- **GNS controlado**: loss_decreases=✅, loss_drops_over_90pct=❌, encoder_receives_gradient=✅

**Alcanzabilidad del target por un canal puramente normal:** 95.4% alcanzable, 4.6% inalcanzable.

Ese segundo número es el techo que impone la ausencia de fricción.
Un residual mayor **no** se explica por física ausente: es
limitación de aprendizaje, y así hay que reportarlo.

## 3.1 Experimento `gravity60_small`

Semillas: `[0, 1, 2]`. Selección de lr por validación con presupuesto idéntico por familia.

| Variante | lr elegido | rejilla (pérdida de validación) |
|---|---|---|
| v3-C | 0.0001 | lr=0.003: 0.0002384, lr=0.001: 0.0002368, lr=0.0003: 0.0002329, lr=0.0001: 0.0002316 |
| v3-I | 0.003 | lr=0.003: 0.000332, lr=0.001: 0.000433, lr=0.0003: 0.0004331, lr=0.0001: 0.0004332 |
| GNS controlado | 0.0001 | lr=0.003: 0.0003342, lr=0.001: 0.0003336, lr=0.0003: 0.0003335, lr=0.0001: 0.0003332 |

### Generalización de un paso (validación)

| Métrica | v3-C | v3-I | GNS controlado |
|---|---|---|---|
| RMSE `Δp` global | 0.007807 ± 6.8e-05 | 0.008022 ± 0.0018 | 0.009111 ± 9.9e-06 |
| RMSE `Δp` vuelo libre | 0.0004962 ± 2.1e-05 | 0.001294 ± 0.00058 | 0.001055 ± 1.1e-05 |
| RMSE `Δp` partícula–partícula | 0.02297 ± 0.00023 | 0.02353 ± 0.0066 | 0.02741 ± 2.9e-05 |
| RMSE `Δp` partícula–pared | 0.0247 ± 0.00013 | 0.02425 ± 0.0043 | 0.02679 ± 3e-05 |
| RMSE `Δp` mixto | 0.006247 ± 1.7e-06 | 0.006277 ± 0 | 0.006285 ± 2.3e-05 |
| MAE `Δp` global | 0.001512 ± 1.2e-05 | 0.001543 ± 0.00028 | 0.001976 ± 1.5e-05 |
| RMSE `ΔL` global (⚠ ver nota) | 0.0005404 ± 0 | 0.0005404 ± 0 | 0.0151 ± 0.0066 |
| fracción normal del error | 0.956 ± 0.00054 | 0.915 ± 0.056 | 0.9357 ± 0.0013 |
| fracción tangencial del error | 0.04396 ± 0.00054 | 0.08498 ± 0.056 | 0.06432 ± 0.0013 |

> ⚠ **La fila de `ΔL` no es una comparación.** Con `λ_ΔL = 0` —igual
> para las tres variantes— la rotación no está supervisada. v3 predice
> `ΔL = 0` exactamente por construcción, así que su cifra es la norma
> del target; GNS tiene una salida rotacional libre que deriva sin
> señal. Ninguno de los dos números mide capacidad rotacional.

### Estabilidad de rollout

| Métrica | v3-C | v3-I | GNS controlado |
|---|---|---|---|
| RMSE `q` a h=1 | 0.0004313 ± 3.9e-06 | 0.0005473 ± 1.8e-05 | 0.0005412 ± 7.7e-07 |
| RMSE `q` a h=5 | 0.007331 ± 0.00011 | 0.0081 ± 0.0027 | 0.009793 ± 4.8e-05 |
| RMSE `q` a h=10 | 0.01935 ± 0.00034 | 0.02303 ± 0.0082 | 0.02906 ± 0.0011 |
| RMSE `q` a h=25 | 0.05255 ± 0.00083 | 0.07462 ± 0.026 | 0.133 ± 0.045 |
| RMSE `v` a h=25 | 0.03635 ± 0.0005 | 0.04681 ± 0.014 | 0.1177 ± 0.063 |
| pasos hasta NaN (H+1 = nunca falló) | 26 ± 0 | 26 ± 0 | 26 ± 0 |

### Coste

| Métrica | v3-C | v3-I | GNS controlado |
|---|---|---|---|
| parámetros | 1.769e+05 ± 0 | 1.769e+05 ± 0 | 1.715e+05 ± 0 |
| segundos de entrenamiento | 39.55 ± 0.59 | 49.15 ± 0.068 | 15.3 ± 0.099 |
| segundos por paso de inferencia | 0.04127 ± 0.0017 | 0.05505 ± 0.0023 | 0.02584 ± 0.00043 |
| mejor pérdida de validación | 0.0002366 ± 5.2e-06 | 0.0002649 ± 0.00012 | 0.000334 ± 9.7e-07 |

### Extrapolación (evaluada una sola vez, al cerrar la fase)

Caso reservado: CASE07 — energía cinética inicial ≈3× la de entrenamiento.

| Métrica | v3-C | v3-I | GNS controlado |
|---|---|---|---|
| RMSE `Δp` global | 0.01064 ± 6.3e-05 | 0.01068 ± 0.0017 | 0.0117 ± 4.4e-06 |
| RMSE `Δp` partícula–partícula | 0.03037 ± 0.00018 | 0.03047 ± 0.0048 | 0.03333 ± 1.9e-05 |
| RMSE `Δp` partícula–pared | 0.02785 ± 0.00015 | 0.02693 ± 0.0055 | 0.03013 ± 4.8e-05 |
| RMSE `q` a h=1 | 0.0005797 ± 4.9e-06 | 0.0006537 ± 2e-05 | 0.0006697 ± 2.9e-06 |
| RMSE `q` a h=25 | 0.07612 ± 0.0012 | 0.09952 ± 0.035 | 0.1587 ± 0.042 |
| precisión de contacto | 0.9759 ± 0 | 0.9675 ± 0.0056 | 0.9643 ± 0 |
| penetración máxima | 0.05269 ± 0.00031 | 0.05333 ± 0.0081 | 0.05808 ± 0.0001 |

## 3.2 Experimento `gravity60_small_dL`

Semillas: `[0, 1, 2]`. Selección de lr por validación con presupuesto idéntico por familia.

| Variante | lr elegido | rejilla (pérdida de validación) |
|---|---|---|
| v3-C | 0.0001 | lr=0.003: 0.0002395, lr=0.001: 0.0002379, lr=0.0003: 0.000234, lr=0.0001: 0.0002327 |
| v3-I | 0.003 | lr=0.003: 0.0003331, lr=0.001: 0.0004341, lr=0.0003: 0.0004342, lr=0.0001: 0.0004343 |
| GNS controlado | 0.0003 | lr=0.003: 0.0003353, lr=0.001: 0.0003345, lr=0.0003: 0.000334, lr=0.0001: 0.000335 |

### Generalización de un paso (validación)

| Métrica | v3-C | v3-I | GNS controlado |
|---|---|---|---|
| RMSE `Δp` global | 0.007807 ± 6.8e-05 | 0.008022 ± 0.0018 | 0.009101 ± 4.6e-06 |
| RMSE `Δp` vuelo libre | 0.0004962 ± 2.1e-05 | 0.001294 ± 0.00058 | 0.001012 ± 8.2e-06 |
| RMSE `Δp` partícula–partícula | 0.02297 ± 0.00023 | 0.02353 ± 0.0066 | 0.02739 ± 6.5e-06 |
| RMSE `Δp` partícula–pared | 0.0247 ± 0.00013 | 0.02425 ± 0.0043 | 0.02678 ± 7.7e-05 |
| RMSE `Δp` mixto | 0.006247 ± 1.7e-06 | 0.006277 ± 0 | 0.006321 ± 9e-05 |
| MAE `Δp` global | 0.001512 ± 1.2e-05 | 0.001543 ± 0.00028 | 0.001898 ± 2.1e-05 |
| RMSE `ΔL` global (⚠ ver nota) | 0.0005404 ± 0 | 0.0005404 ± 0 | 0.0006024 ± 1.3e-05 |
| fracción normal del error | 0.956 ± 0.00054 | 0.915 ± 0.056 | 0.9406 ± 0.001 |
| fracción tangencial del error | 0.04396 ± 0.00054 | 0.08498 ± 0.056 | 0.05941 ± 0.001 |

> ⚠ **La fila de `ΔL` no es una comparación.** Con `λ_ΔL = 0` —igual
> para las tres variantes— la rotación no está supervisada. v3 predice
> `ΔL = 0` exactamente por construcción, así que su cifra es la norma
> del target; GNS tiene una salida rotacional libre que deriva sin
> señal. Ninguno de los dos números mide capacidad rotacional.

### Estabilidad de rollout

| Métrica | v3-C | v3-I | GNS controlado |
|---|---|---|---|
| RMSE `q` a h=1 | 0.0004313 ± 3.9e-06 | 0.0005473 ± 1.8e-05 | 0.0005398 ± 5.5e-07 |
| RMSE `q` a h=5 | 0.007331 ± 0.00011 | 0.0081 ± 0.0027 | 0.009724 ± 3.1e-05 |
| RMSE `q` a h=10 | 0.01935 ± 0.00034 | 0.02303 ± 0.0082 | 0.02809 ± 0.00013 |
| RMSE `q` a h=25 | 0.05255 ± 0.00083 | 0.07462 ± 0.026 | 0.09244 ± 0.001 |
| RMSE `v` a h=25 | 0.03635 ± 0.0005 | 0.04681 ± 0.014 | 0.05661 ± 0.00085 |
| fracción de rollouts sin NaN | 1 ± 0 | 1 ± 0 | 1 ± 0 |
| pasos hasta NaN (H+1 = nunca falló) | 26 ± 0 | 26 ± 0 | 26 ± 0 |

### Coste

| Métrica | v3-C | v3-I | GNS controlado |
|---|---|---|---|
| parámetros | 1.769e+05 ± 0 | 1.769e+05 ± 0 | 1.715e+05 ± 0 |
| segundos de entrenamiento | 55.74 ± 1.6 | 122.9 ± 46 | 13.77 ± 0.041 |
| segundos por paso de inferencia | 0.05848 ± 0.0015 | 0.1714 ± 0.027 | 0.02324 ± 0.0012 |
| mejor pérdida de validación | 0.0002377 ± 5.2e-06 | 0.000266 ± 0.00012 | 0.0003347 ± 6.8e-07 |

### Extrapolación (evaluada una sola vez, al cerrar la fase)

Caso reservado: gravity60_small_dL.

| Métrica | v3-C | v3-I | GNS controlado |
|---|---|---|---|
| RMSE `Δp` global | 0.01064 ± 6.3e-05 | 0.01068 ± 0.0017 | 0.0117 ± 5.9e-06 |
| RMSE `Δp` partícula–partícula | 0.03037 ± 0.00018 | 0.03047 ± 0.0048 | 0.03332 ± 4.1e-06 |
| RMSE `Δp` partícula–pared | 0.02785 ± 0.00015 | 0.02693 ± 0.0055 | 0.03013 ± 4.3e-05 |
| RMSE `q` a h=1 | 0.0005797 ± 4.9e-06 | 0.0006537 ± 2e-05 | 0.0006684 ± 8.7e-07 |
| RMSE `q` a h=25 | 0.07612 ± 0.0012 | 0.09952 ± 0.035 | 0.1226 ± 0.0012 |
| precisión de contacto | 0.9759 ± 0 | 0.9675 ± 0.0056 | 0.9644 ± 0.00012 |
| penetración máxima | 0.05269 ± 0.00031 | 0.05333 ± 0.0081 | 0.05809 ± 2.3e-05 |

## 3.3 Experimento `one_sphere_wall`

Semillas: `[0, 1, 2]`. Selección de lr por validación con presupuesto idéntico por familia.

| Variante | lr elegido | rejilla (pérdida de validación) |
|---|---|---|
| v3-C | 0.0001 | lr=0.003: 5.945e-05, lr=0.001: 5.669e-05, lr=0.0003: 4.321e-05, lr=0.0001: 2.991e-05 |
| v3-I | 0.0003 | lr=0.003: 6.577e-05, lr=0.001: 3.38e-05, lr=0.0003: 3.012e-05, lr=0.0001: 3.833e-05 |
| GNS controlado | 0.003 | lr=0.003: 4.564e-05, lr=0.001: 4.859e-05, lr=0.0003: 5.237e-05, lr=0.0001: 5.535e-05 |

### Generalización de un paso (validación)

| Métrica | v3-C | v3-I | GNS controlado |
|---|---|---|---|
| RMSE `Δp` global | 0.003785 ± 0.00016 | 0.003117 ± 1.3e-05 | 0.003371 ± 0.00011 |
| RMSE `Δp` vuelo libre | 3.326e-06 ± 1.4e-07 | 0.000452 ± 1.7e-05 | 0.001677 ± 0.00031 |
| RMSE `Δp` partícula–pared | 0.01338 ± 0.00057 | 0.01091 ± 5.4e-05 | 0.01044 ± 0.0002 |
| MAE `Δp` global | 0.0005942 ± 2.3e-05 | 0.0005112 ± 1.7e-06 | 0.001365 ± 0.00026 |
| RMSE `ΔL` global (⚠ ver nota) | 0.0001803 ± 0 | 0.0001803 ± 0 | 0.001951 ± 0.0004 |
| fracción normal del error | 0.9267 ± 0.0089 | 0.9151 ± 0.00085 | 0.8866 ± 0.017 |
| fracción tangencial del error | 0.07329 ± 0.0089 | 0.08493 ± 0.00085 | 0.1134 ± 0.017 |

> ⚠ **La fila de `ΔL` no es una comparación.** Con `λ_ΔL = 0` —igual
> para las tres variantes— la rotación no está supervisada. v3 predice
> `ΔL = 0` exactamente por construcción, así que su cifra es la norma
> del target; GNS tiene una salida rotacional libre que deriva sin
> señal. Ninguno de los dos números mide capacidad rotacional.

### Estabilidad de rollout

| Métrica | v3-C | v3-I | GNS controlado |
|---|---|---|---|
| RMSE `q` a h=1 | 0.002423 ± 0 | 0.002423 ± 0 | 0.002496 ± 1.4e-05 |
| RMSE `q` a h=5 | 0.0001176 ± 4.2e-06 | 0.0004195 ± 1.1e-05 | 0.002728 ± 0.0005 |
| RMSE `q` a h=10 | 0.002003 ± 5.8e-05 | 0.0005279 ± 2.4e-05 | 0.008015 ± 0.0014 |
| RMSE `q` a h=25 | 0.02711 ± 0.0019 | 0.01523 ± 0.00017 | 0.05288 ± 0.0039 |
| RMSE `v` a h=25 | 0.01558 ± 0.0023 | 0.008399 ± 0.00014 | 0.03427 ± 3.2e-05 |
| pasos hasta NaN (H+1 = nunca falló) | 26 ± 0 | 26 ± 0 | 26 ± 0 |

> Con una o dos partículas la curva por horizonte **no es monótona**:
> el error a un horizonte dado lo domina si en ese instante hay o no
> un contacto activo, y con tan pocos eventos la media sobre puntos de
> partida no lo suaviza. La curva de `gravity60_small`, con 60
> partículas, sí es monótona y es la que conviene leer como
> estabilidad.

### Coste

| Métrica | v3-C | v3-I | GNS controlado |
|---|---|---|---|
| parámetros | 1.769e+05 ± 0 | 1.769e+05 ± 0 | 1.715e+05 ± 0 |
| segundos de entrenamiento | 14.33 ± 0.095 | 24.83 ± 0.15 | 2.978 ± 0.19 |
| segundos por paso de inferencia | 0.007391 ± 0.0001 | 0.008302 ± 0.00021 | 0.004215 ± 5.3e-05 |
| mejor pérdida de validación | 3.192e-05 ± 2e-06 | 3.072e-05 ± 1.6e-06 | 4.372e-05 ± 2.9e-06 |

### Extrapolación (evaluada una sola vez, al cerrar la fase)

Caso reservado: ángulos oblicuos 45°, 30° y 10° — diagnóstico de física pendiente.

> Estos ángulos **no** miden el canal normal: sin fricción ni spin,
> la componente tangencial del impacto oblicuo es irreproducible por
> construcción. La cifra sirve para dimensionar la física pendiente,
> no para juzgar el MVP (§16.4).

| Métrica | v3-C | v3-I | GNS controlado |
|---|---|---|---|
| RMSE `Δp` global | 0.00647 ± 0.00028 | 0.005374 ± 2.3e-05 | 0.005594 ± 0.00018 |
| RMSE `Δp` partícula–pared | 0.02287 ± 0.00097 | 0.01887 ± 9e-05 | 0.01893 ± 0.00053 |
| RMSE `q` a h=1 | 0.002295 ± 1.7e-05 | 0.00221 ± 1.5e-06 | 0.002287 ± 1.4e-05 |
| RMSE `q` a h=25 | 0.04704 ± 0.005 | 0.03384 ± 0.00014 | 0.05132 ± 0.0032 |
| precisión de contacto | 1 ± 0 | 1 ± 0 | 1 ± 0 |
| penetración máxima | 0.1942 ± 0.00028 | 0.1926 ± 2.9e-05 | 0.1928 ± 0.00023 |

## 3.4 Experimento `one_sphere_wall_dL`

Semillas: `[0, 1, 2]`. Selección de lr por validación con presupuesto idéntico por familia.

| Variante | lr elegido | rejilla (pérdida de validación) |
|---|---|---|
| v3-C | 0.0001 | lr=0.003: 5.956e-05, lr=0.001: 5.68e-05, lr=0.0003: 4.332e-05, lr=0.0001: 3.002e-05 |
| v3-I | 0.0003 | lr=0.003: 6.589e-05, lr=0.001: 3.392e-05, lr=0.0003: 3.023e-05, lr=0.0001: 3.845e-05 |
| GNS controlado | 0.003 | lr=0.003: 4.623e-05, lr=0.001: 5.043e-05, lr=0.0003: 5.337e-05, lr=0.0001: 5.677e-05 |

### Generalización de un paso (validación)

| Métrica | v3-C | v3-I | GNS controlado |
|---|---|---|---|
| RMSE `Δp` global | 0.003785 ± 0.00016 | 0.003117 ± 1.3e-05 | 0.003356 ± 0.00013 |
| RMSE `Δp` vuelo libre | 3.326e-06 ± 1.4e-07 | 0.000452 ± 1.7e-05 | 0.001647 ± 0.00017 |
| RMSE `Δp` partícula–pared | 0.01338 ± 0.00057 | 0.01091 ± 5.4e-05 | 0.01046 ± 0.00047 |
| MAE `Δp` global | 0.0005942 ± 2.3e-05 | 0.0005112 ± 1.7e-06 | 0.001319 ± 7.4e-05 |
| RMSE `ΔL` global (⚠ ver nota) | 0.0001803 ± 0 | 0.0001803 ± 0 | 0.000309 ± 9.9e-05 |
| fracción normal del error | 0.9267 ± 0.0089 | 0.9151 ± 0.00085 | 0.9328 ± 0.019 |
| fracción tangencial del error | 0.07329 ± 0.0089 | 0.08493 ± 0.00085 | 0.06724 ± 0.019 |

> ⚠ **La fila de `ΔL` no es una comparación.** Con `λ_ΔL = 0` —igual
> para las tres variantes— la rotación no está supervisada. v3 predice
> `ΔL = 0` exactamente por construcción, así que su cifra es la norma
> del target; GNS tiene una salida rotacional libre que deriva sin
> señal. Ninguno de los dos números mide capacidad rotacional.

### Estabilidad de rollout

| Métrica | v3-C | v3-I | GNS controlado |
|---|---|---|---|
| RMSE `q` a h=1 | 0.002423 ± 0 | 0.002423 ± 0 | 0.002494 ± 9.9e-06 |
| RMSE `q` a h=5 | 0.0001176 ± 4.2e-06 | 0.0004195 ± 1.1e-05 | 0.002629 ± 0.00026 |
| RMSE `q` a h=10 | 0.002003 ± 5.8e-05 | 0.0005279 ± 2.4e-05 | 0.0075 ± 0.00088 |
| RMSE `q` a h=25 | 0.02711 ± 0.0019 | 0.01523 ± 0.00017 | 0.04924 ± 0.0034 |
| RMSE `v` a h=25 | 0.01558 ± 0.0023 | 0.008399 ± 0.00014 | 0.02865 ± 0.0025 |
| fracción de rollouts sin NaN | 1 ± 0 | 1 ± 0 | 1 ± 0 |
| pasos hasta NaN (H+1 = nunca falló) | 26 ± 0 | 26 ± 0 | 26 ± 0 |

### Coste

| Métrica | v3-C | v3-I | GNS controlado |
|---|---|---|---|
| parámetros | 1.769e+05 ± 0 | 1.769e+05 ± 0 | 1.715e+05 ± 0 |
| segundos de entrenamiento | 13.78 ± 0.21 | 22.78 ± 0.32 | 2.68 ± 0.024 |
| segundos por paso de inferencia | 0.006593 ± 0.00035 | 0.007082 ± 5.8e-05 | 0.003656 ± 6.3e-05 |
| mejor pérdida de validación | 3.204e-05 ± 2e-06 | 3.083e-05 ± 1.6e-06 | 4.373e-05 ± 3.3e-06 |

### Extrapolación (evaluada una sola vez, al cerrar la fase)

Caso reservado: one_sphere_wall_dL.

| Métrica | v3-C | v3-I | GNS controlado |
|---|---|---|---|
| RMSE `Δp` global | 0.00647 ± 0.00028 | 0.005374 ± 2.3e-05 | 0.005607 ± 0.00031 |
| RMSE `Δp` partícula–pared | 0.02287 ± 0.00097 | 0.01887 ± 9e-05 | 0.01907 ± 0.0011 |
| RMSE `q` a h=1 | 0.002295 ± 1.7e-05 | 0.00221 ± 1.5e-06 | 0.002285 ± 1.6e-05 |
| RMSE `q` a h=25 | 0.04704 ± 0.005 | 0.03384 ± 0.00014 | 0.04524 ± 0.0029 |
| precisión de contacto | 1 ± 0 | 1 ± 0 | 1 ± 0 |
| penetración máxima | 0.1942 ± 0.00028 | 0.1926 ± 2.9e-05 | 0.1929 ± 0.00037 |

## 3.5 Experimento `two_spheres`

Semillas: `[0, 1, 2]`. Selección de lr por validación con presupuesto idéntico por familia.

| Variante | lr elegido | rejilla (pérdida de validación) |
|---|---|---|
| v3-C | 0.001 | lr=0.003: 2.123e-05, lr=0.001: 2.071e-05, lr=0.0003: 2.249e-05, lr=0.0001: 2.978e-05 |
| v3-I | 0.001 | lr=0.003: 0.0003852, lr=0.001: 0.0001522, lr=0.0003: 0.000188, lr=0.0001: 0.0009505 |
| GNS controlado | 0.003 | lr=0.003: 0.0003287, lr=0.001: 0.0003708, lr=0.0003: 0.0004069, lr=0.0001: 0.0004303 |

### Generalización de un paso (validación)

| Métrica | v3-C | v3-I | GNS controlado |
|---|---|---|---|
| RMSE `Δp` global | 0.003511 ± 0.00043 | 0.008699 ± 0.00018 | 0.01105 ± 0.00064 |
| RMSE `Δp` vuelo libre | 0.0002553 ± 1.1e-05 | 0.001491 ± 0.00016 | 0.005433 ± 0.002 |
| RMSE `Δp` partícula–partícula | 0.01056 ± 0.0013 | 0.02588 ± 0.00062 | 0.02916 ± 0.0013 |
| MAE `Δp` global | 0.000771 ± 8.7e-05 | 0.001845 ± 3.1e-05 | 0.005886 ± 0.0012 |
| RMSE `ΔL` global (⚠ ver nota) | 0.00048 ± 6.3e-20 | 0.00048 ± 6.3e-20 | 0.006196 ± 0.0014 |
| fracción normal del error | 0.8312 ± 0.038 | 0.9606 ± 0.017 | 0.9659 ± 0.03 |
| fracción tangencial del error | 0.1688 ± 0.038 | 0.03943 ± 0.017 | 0.03407 ± 0.03 |

> ⚠ **La fila de `ΔL` no es una comparación.** Con `λ_ΔL = 0` —igual
> para las tres variantes— la rotación no está supervisada. v3 predice
> `ΔL = 0` exactamente por construcción, así que su cifra es la norma
> del target; GNS tiene una salida rotacional libre que deriva sin
> señal. Ninguno de los dos números mide capacidad rotacional.

### Estabilidad de rollout

| Métrica | v3-C | v3-I | GNS controlado |
|---|---|---|---|
| RMSE `q` a h=1 | 0.005209 ± 3.3e-06 | 0.005234 ± 4.8e-06 | 0.005272 ± 1.9e-05 |
| RMSE `q` a h=5 | 0.001728 ± 0.00031 | 0.005133 ± 0.00014 | 0.01269 ± 0.0021 |
| RMSE `q` a h=10 | 0.004686 ± 0.00057 | 0.01351 ± 0.00057 | 0.03639 ± 0.0075 |
| RMSE `q` a h=25 | 0.02608 ± 0.0027 | 0.07053 ± 0.0023 | 0.1886 ± 0.037 |
| RMSE `v` a h=25 | 0.01904 ± 0.0014 | 0.04977 ± 0.00092 | 0.1486 ± 0.032 |
| pasos hasta NaN (H+1 = nunca falló) | 26 ± 0 | 26 ± 0 | 26 ± 0 |

> Con una o dos partículas la curva por horizonte **no es monótona**:
> el error a un horizonte dado lo domina si en ese instante hay o no
> un contacto activo, y con tan pocos eventos la media sobre puntos de
> partida no lo suaviza. La curva de `gravity60_small`, con 60
> partículas, sí es monótona y es la que conviene leer como
> estabilidad.

### Coste

| Métrica | v3-C | v3-I | GNS controlado |
|---|---|---|---|
| parámetros | 1.769e+05 ± 0 | 1.769e+05 ± 0 | 1.715e+05 ± 0 |
| segundos de entrenamiento | 21.05 ± 0.86 | 24.31 ± 0.44 | 7.377 ± 0.16 |
| segundos por paso de inferencia | 0.01081 ± 0.00042 | 0.009057 ± 0.00099 | 0.003953 ± 0.00014 |
| mejor pérdida de validación | 1.765e-05 ± 2.7e-06 | 0.0001716 ± 1.7e-05 | 0.0003395 ± 3.7e-05 |

### Extrapolación (evaluada una sola vez, al cerrar la fase)

Caso reservado: `4x` — escala de velocidad 4× la de entrenamiento.

| Métrica | v3-C | v3-I | GNS controlado |
|---|---|---|---|
| RMSE `Δp` global | 0.01277 ± 0.0011 | 0.01801 ± 7.1e-05 | 0.02075 ± 0.00056 |
| RMSE `Δp` partícula–partícula | 0.0385 ± 0.0032 | 0.05343 ± 0.00035 | 0.06014 ± 0.0025 |
| RMSE `q` a h=1 | 0.01103 ± 3.1e-05 | 0.01092 ± 3.7e-06 | 0.01097 ± 5.1e-05 |
| RMSE `q` a h=25 | 0.05307 ± 0.0027 | 0.1042 ± 0.0033 | 0.2911 ± 0.048 |
| precisión de contacto | 1 ± 0 | 1 ± 0 | 1 ± 0 |
| penetración máxima | 0.4898 ± 0.0017 | 0.505 ± 0.00098 | 0.5152 ± 0.0014 |

## 3.6 Experimento `two_spheres_dL`

Semillas: `[0, 1, 2]`. Selección de lr por validación con presupuesto idéntico por familia.

| Variante | lr elegido | rejilla (pérdida de validación) |
|---|---|---|
| v3-C | 0.001 | lr=0.003: 2.191e-05, lr=0.001: 2.138e-05, lr=0.0003: 2.316e-05, lr=0.0001: 3.045e-05 |
| v3-I | 0.001 | lr=0.003: 0.0003858, lr=0.001: 0.0001529, lr=0.0003: 0.0001887, lr=0.0001: 0.0009512 |
| GNS controlado | 0.0003 | lr=0.003: 0.0006216, lr=0.001: 0.0005075, lr=0.0003: 0.0004838, lr=0.0001: 0.000486 |

### Generalización de un paso (validación)

| Métrica | v3-C | v3-I | GNS controlado |
|---|---|---|---|
| RMSE `Δp` global | 0.003511 ± 0.00043 | 0.008699 ± 0.00018 | 0.01189 ± 0.0011 |
| RMSE `Δp` vuelo libre | 0.0002553 ± 1.1e-05 | 0.001491 ± 0.00016 | 0.007166 ± 0.0017 |
| RMSE `Δp` partícula–partícula | 0.01056 ± 0.0013 | 0.02588 ± 0.00062 | 0.02931 ± 0.0015 |
| MAE `Δp` global | 0.000771 ± 8.7e-05 | 0.001845 ± 3.1e-05 | 0.006926 ± 0.0013 |
| RMSE `ΔL` global (⚠ ver nota) | 0.00048 ± 6.3e-20 | 0.00048 ± 6.3e-20 | 0.005627 ± 0.0028 |
| fracción normal del error | 0.8312 ± 0.038 | 0.9606 ± 0.017 | 0.9731 ± 0.0061 |
| fracción tangencial del error | 0.1688 ± 0.038 | 0.03943 ± 0.017 | 0.0269 ± 0.0061 |

> ⚠ **La fila de `ΔL` no es una comparación.** Con `λ_ΔL = 0` —igual
> para las tres variantes— la rotación no está supervisada. v3 predice
> `ΔL = 0` exactamente por construcción, así que su cifra es la norma
> del target; GNS tiene una salida rotacional libre que deriva sin
> señal. Ninguno de los dos números mide capacidad rotacional.

### Estabilidad de rollout

| Métrica | v3-C | v3-I | GNS controlado |
|---|---|---|---|
| RMSE `q` a h=1 | 0.005209 ± 3.3e-06 | 0.005234 ± 4.8e-06 | 0.00528 ± 3.3e-05 |
| RMSE `q` a h=5 | 0.001728 ± 0.00031 | 0.005133 ± 0.00014 | 0.0146 ± 0.002 |
| RMSE `q` a h=10 | 0.004686 ± 0.00057 | 0.01351 ± 0.00057 | 0.04159 ± 0.0072 |
| RMSE `q` a h=25 | 0.02608 ± 0.0027 | 0.07053 ± 0.0023 | 0.2034 ± 0.049 |
| RMSE `v` a h=25 | 0.01904 ± 0.0014 | 0.04977 ± 0.00092 | 0.1522 ± 0.035 |
| fracción de rollouts sin NaN | 1 ± 0 | 1 ± 0 | 1 ± 0 |
| pasos hasta NaN (H+1 = nunca falló) | 26 ± 0 | 26 ± 0 | 26 ± 0 |

### Coste

| Métrica | v3-C | v3-I | GNS controlado |
|---|---|---|---|
| parámetros | 1.769e+05 ± 0 | 1.769e+05 ± 0 | 1.715e+05 ± 0 |
| segundos de entrenamiento | 14.08 ± 0.28 | 23.16 ± 0.26 | 6.692 ± 0.11 |
| segundos por paso de inferencia | 0.0075 ± 0.00069 | 0.008768 ± 0.0012 | 0.003793 ± 0.00016 |
| mejor pérdida de validación | 1.832e-05 ± 2.7e-06 | 0.0001723 ± 1.7e-05 | 0.000473 ± 0.00013 |

### Extrapolación (evaluada una sola vez, al cerrar la fase)

Caso reservado: two_spheres_dL.

| Métrica | v3-C | v3-I | GNS controlado |
|---|---|---|---|
| RMSE `Δp` global | 0.01277 ± 0.0011 | 0.01801 ± 7.1e-05 | 0.02057 ± 0.00069 |
| RMSE `Δp` partícula–partícula | 0.0385 ± 0.0032 | 0.05343 ± 0.00035 | 0.05797 ± 0.0014 |
| RMSE `q` a h=1 | 0.01103 ± 3.1e-05 | 0.01092 ± 3.7e-06 | 0.01098 ± 7.7e-05 |
| RMSE `q` a h=25 | 0.05307 ± 0.0027 | 0.1042 ± 0.0033 | 0.3024 ± 0.045 |
| precisión de contacto | 1 ± 0 | 1 ± 0 | 1 ± 0 |
| penetración máxima | 0.4898 ± 0.0017 | 0.505 ± 0.00098 | 0.5139 ± 0.0012 |

---

## 5. Qué dice la auditoría temporal sobre el régimen

Antes de cualquier comparación de rendimiento, la auditoría
([DATA_AUDIT.md](DATA_AUDIT.md)) responde la pregunta que motiva la existencia
de la rama impulsiva: **¿el intervalo entre snapshots resuelve la duración del
contacto o la contiene entera?**

La respuesta, en los datasets de caja, es que **la resuelve**:

| Dataset | duración medida | duración analítica `π√(m_eff/k_n)` | `ρ_t` |
|---|---|---|---|
| `two_spheres` | 11 snapshots | 11.4 | 0.09 |
| `one_sphere_wall` | 16 | 16.1 | 0.06 |
| `sixty_homogeneous` | 11 (pp) / 16 (pw) | 11.4 / 16.1 | 0.08 |
| `sixty_gravity` | 4 (pp) / 5 (pw) | 3.6 / 5.1 | 0.25 |
| `rotating_cylinder` | 1 | 0.4 / 0.5 | 1.00 |

La coincidencia con el semiperiodo del oscilador masa–resorte del DEM, dentro
de un snapshot en los cinco casos, es lo que valida que la detección
geométrica está midiendo contacto real y no ruido de discretización.

Consecuencia incómoda pero clara: **la premisa de que los choques están
submuestreados no se sostiene para los datasets de entrenamiento.** `ρ_t`
vale entre 0.06 y 0.25, es decir, cada contacto ocupa de 4 a 16 intervalos. El
único caso genuinamente impulsivo es el cilindro rotatorio, que se graba a
`dt = 1e-3 s` —diez veces más grueso— y es solo de inferencia.

Esto **no** invalida la rama impulsiva: la hace necesaria justo donde el
gemelo digital la va a necesitar (el molino rotatorio), y prescindible donde
se entrena hoy. La arquitectura híbrida de v3 es la respuesta correcta a un
dataset que contiene los dos regímenes; lo que la auditoría corrige es *cuál*
de los dos domina en cada archivo.

Corroboración independiente desde el propio entrenamiento: en el
micro-overfit, la cabeza `I` de `v3-I` lleva la restitución de `e ≈ 0.50` a
`e ≈ 1.6e-85`. El modelo impulsivo, entrenado libremente sobre contactos
temporalmente resueltos, **aprende a apagar el impulso**. Es exactamente lo
que predice la auditoría, obtenido por una vía completamente distinta.

## 6. Limitaciones del MVP normal

Están declaradas en [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) y se
repiten aquí porque condicionan la lectura de cada número de arriba:

1. **`ΔL` predicho es exactamente cero.** Sin canal tangencial no hay torque.
   Cualquier RMSE de `ΔL` en las tablas es la norma del target, es decir, una
   medida de física ausente, no un error de ajuste.
2. **No hay fricción**, ni continua ni impulsiva. El dataset con gravedad usa
   `mu_s = 0.1` partícula–partícula y `0.3` partícula–pared, así que parte del
   `Δp` real es inalcanzable por construcción. La fracción exacta se mide y se
   reporta arriba en vez de suponerse.
3. **No hay memoria tangencial `M`.** Un contacto persistente no acumula
   desplazamiento tangencial; sticking y sliding no existen.
4. **El cierre `C` está congelado**, como debe estar antes de identificar los
   canales físicos.
5. **`v3-H` no existe.** El router es duro y por perfil.
6. Los rollouts llegan a 25 pasos. 50 y 100 quedan para después del checkpoint.

## 7. Comparación con GNS: cómo leerla

> **Corrección posterior.** Las tablas de arriba se produjeron con
> `λ_ΔL = 0` para las tres variantes. Una auditoría de equidad
> ([AUDITORIA_EQUIDAD.md](AUDITORIA_EQUIDAD.md)) mostró que eso deja el canal
> rotacional de GNS sin supervisar y, como `ω` realimenta su encoder, infla la
> ventaja de v3 en rollout largo. Repetido con `λ_ΔL = 1` —que deja el
> entrenamiento de v3 **bit a bit idéntico** y solo puede ayudar a GNS—, la
> ventaja de v3-C en el experimento principal pasa de `2.53x` a `1.76x` a
> h=25. **El orden entre las tres variantes no cambia en ningún experimento.**
> Las cifras corregidas están en esa auditoría.

Tres observaciones antes de mirar los números:

1. **La diferencia en un paso es modesta; en rollout es grande.** Es la firma
   esperada de un sesgo inductivo estructural: la conservación exacta y la
   unilateralidad no ayudan mucho a acertar un incremento aislado, pero evitan
   que el error se realimente. Un informe que solo mirase el RMSE de un paso
   concluiría que las tres arquitecturas son casi equivalentes.

2. **v3 arranca cerca del suelo de pérdida.** Su predicción sin entrenar ya
   contiene la gravedad exacta y no inventa fuerza en vuelo libre. Por eso su
   pérdida inicial es del orden de la **final** de GNS, y por eso la
   "reducción relativa" del micro-overfit no es una métrica comparable entre
   familias.

3. **GNS es más rápido por paso.** No tiene autograd sobre el potencial ni
   solver. El coste computacional de v3 es real y está reportado; si el
   criterio fuese error por segundo de inferencia, la conclusión podría
   cambiar y hay que decirlo.

## 8. Decisión recomendada para la siguiente fase

**Recomendación: profundizar `v3-C` y, dentro de él, implementar primero
fricción y spin.**

El dictamen se apoya en tres piezas de evidencia independientes, no en una
preferencia arquitectónica previa:

**a) La auditoría dice que los contactos están temporalmente resueltos.**
`ρ_t` entre 0.06 y 0.25 en todos los datasets de caja, con la duración medida
coincidiendo con `π√(m_eff/k_n)` dentro de un snapshot. La hipótesis de
choque submuestreado —la que motivaba priorizar `v3-I`— **no se sostiene para
los datos de entrenamiento disponibles**.

**b) El propio entrenamiento lo confirma por otra vía.** La cabeza `I`,
entrenada sin restricciones sobre esos mismos contactos, lleva la restitución
a `e ≈ 0`: aprende a desactivar el impulso. Dos métodos independientes
—geometría de episodios y descenso de gradiente— llegan a la misma conclusión.

**c) `v3-C` gana el experimento principal, incluida la extrapolación.** Mejor
en un paso, mejor en los cuatro horizontes de rollout, mejor en `CASE07`, y
con la varianza entre semillas más baja de las tres variantes (`±0.0008`
frente a `±0.026` de `v3-I` en el rollout a h=25).

**El resultado no es unánime y conviene decirlo.** `v3-I` **gana el benchmark
de una esfera contra pared**, tanto en un paso como en rollout y en los
ángulos oblicuos reservados. Es el escenario que más se parece a un problema
de restitución de libro —un cuerpo, un contacto, un rebote— y es razonable que
la parametrización impulsiva le siente bien aunque el contacto esté resuelto.
No es un dato contra la recomendación, pero sí acota su alcance: lo que la
evidencia respalda es que **en régimen granular denso con gravedad**, que es
el objetivo, `v3-C` es la apuesta correcta.

En los tres experimentos, el mejor perfil de v3 bate a GNS controlado en
rollout a h=25 por factores de **1.8x, 3.2x y 7.8x** —cifras ya corregidas por
la auditoría de equidad—, con presupuesto de parámetros equiparado (1.77e5
frente a 1.72e5) y protocolo idéntico.

**Qué implementar a continuación, en este orden:**

1. **Fricción y spin** (fase 9). Es el cuello de botella medido, no supuesto:
   el error residual de `v3-C` es 95–96 % normal y 4–7 % tangencial, y `ΔL`
   predicho es exactamente cero. Mientras no exista canal tangencial, ninguna
   mejora del canal normal puede cerrar esa fracción, y los choques oblicuos
   del benchmark de dos esferas no son evaluables.
2. **Memoria tangencial `M`** (fase 10), inmediatamente después: sin ella la
   fricción no distingue sticking de sliding y una pila estática no se
   sostiene.
3. **Pared móvil y el cilindro rotatorio** (fase 12). Aquí sí hace falta
   `v3-I`: es el único caso del dataset con `ρ_t ≈ 1`, y la interfaz de
   velocidad de pared ya está implementada y probada.
4. **`v3-H`** (fase 11) **solo después** de 1 y 2, y con una justificación
   empírica: hoy la evidencia no muestra mezcla de regímenes dentro de un
   mismo archivo, sino regímenes distintos en archivos distintos. Un router
   por dataset —que ya existe— cubre eso sin la no identificabilidad que
   introduce un router por contacto.

**Qué NO recomienda esta evidencia:**

- *Priorizar `v3-I`* — los eventos no están submuestreados donde se entrena.
- *Avanzar a `v3-H`* — no hay todavía evidencia de mezcla intra-archivo, y sin
  fricción ni memoria el perfil no sería más que un `v3-C` con protección
  numérica.
- *Corregir datos, geometría, integración o solver antes de seguir* — la suite
  estructural pasa en `float64` con las tolerancias exigidas, el `dt` y la
  gravedad documentados se confirman contra los datos, y no hay tunneling ni
  NaN en ninguna corrida. La base es fiable para comparar.

**Reserva explícita.** Todo lo anterior se mide con presupuestos pequeños
(400 actualizaciones, ventanas de 500 snapshots, horizontes hasta 25). Las
diferencias observadas son consistentes entre semillas y a lo largo de tres
experimentos, pero **no** se ha demostrado que se mantengan al escalar. Antes
de tratar la ventaja de `v3-C` como establecida hace falta repetir el nivel 4
con el dataset completo y horizontes de 50 y 100.

