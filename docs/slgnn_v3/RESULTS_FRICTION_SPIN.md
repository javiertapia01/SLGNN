# RESULTS_FRICTION_SPIN — primer gate de `Psi_tau`

Resultado de validación de la primera extensión posterior al MVP normal.
Fuente ejecutable:
`results/slgnn_v3/compare/comparison_two_spheres_friction_spin.json`.

## Protocolo

- Dataset: `two_spheres`.
- Entrenamiento: `1x`; validación: `2x`.
- Extrapolación `4x`: **no evaluada**.
- Presupuesto: 400 updates, batch 4, tres semillas.
- Selección de `lr`: rejilla común `{3e-3, 1e-3, 3e-4, 1e-4}` por validación.
- Objetivo compartido: `lambda_delta_p = 1`, `lambda_delta_L = 1`.
- Comparación: v3-C normal, v3-C + `Psi_tau`, GNS controlado.

La tasa elegida para ambos perfiles v3 fue `1e-3`; para GNS, `3e-4`.

## Resultado principal

| Métrica de validación | v3-C normal | v3-C + `Psi_tau` | Cambio |
|---|---:|---:|---:|
| RMSE `Delta L` global | 4.8001e-4 | 4.3324e-4 | -9.7% |
| RMSE `Delta L` en contacto pp | 1.4461e-3 | 1.3059e-3 | -9.7% |
| RMSE `omega`, rollout h=25 | 4.6783e-2 | 3.1267e-2 | -33.2% |
| RMSE `q`, rollout h=25 | 2.6081e-2 | 2.2059e-2 | -15.4% |
| RMSE `Delta p` global | 3.5108e-3 | 3.4728e-3 | -1.1% |
| Parámetros | 176,877 | 181,679 | +2.7% |
| Tiempo de entrenamiento | 16.40 s | 16.75 s | +2.1% |

El resultado cumple el criterio del gate: el canal reduce el error rotacional
de un paso y de rollout, no deteriora la traslación y conserva un coste
incremental pequeño. Las tres semillas producen el mismo orden.

GNS controlado obtuvo RMSE `Delta L` global `5.627e-3`, RMSE `omega` h=25
`0.9887` y RMSE `q` h=25 `0.2034`. Esas cifras sirven como control de la
corrida, no como conclusión de extrapolación.

## Alcance del dictamen

Este gate valida **disipación tangencial continua y spin por `J^T`**. No valida:

- fricción impulsiva ni cono de Coulomb;
- sticking/sliding;
- memoria persistente `M`;
- pared móvil;
- extrapolación `4x` o rollouts mayores que 25.

La siguiente implementación es la proyección de Coulomb y el estado
tangencial persistente, comenzando por contactos compliant antes de habilitar
el router híbrido.
