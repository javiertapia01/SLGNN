# AUDITORÍA DE EQUIDAD — comparación SLGNN-v3 / GNS

Generado por `scripts/slgnn_v3/make_fairness_report.py`. **No editar a mano.**

## 1. La pregunta

> «SLGNN tuvo mejor desempeño en rollouts y generalización que GNS, pero en el
> informe proponemos que en la función de pérdida de SLGNN se incluyera el
> error respecto a rollouts. ¿Pudo venir de ahí la ventaja?»

La duda está bien fundada: la formulación oficial de SLGNN-v3 **sí** incluye
`L_roll` en la pérdida total (§13.7). Si ese término se hubiera activado solo
para v3, su ventaja en rollout largo dejaría de ser atribuible al sesgo
inductivo —estaría entrenando directamente contra la métrica con la que se lo
evalúa— y la comparación no mediría nada.

## 2. Respuesta corta

**No.** En el MVP la pérdida de entrenamiento es de **un solo paso** para las
tres variantes, sin ningún término de rollout:

```
loss = λ_Δp · ‖Δp_θ − Δp_DEM‖² / N  +  λ_ΔL · ‖ΔL_θ − ΔL_DEM‖² / N
```

con `λ_Δp = 1` y `λ_ΔL = 0` en las tres. `model.step` se llama **una vez** por
batch: no hay desenrollado. Los `config_resolved.yaml` de las nueve corridas
finales de `gravity60_small` lo confirman, y `rollout_loss` existe en
`slgnn_v3/losses.py` pero el runner nunca la llama.

Sin embargo, la pregunta **destapó un sesgo real en la dirección contraria**,
que se documenta en la sección 4 y se ha medido y corregido.

## 3. Canales auditados

Cada vía por la que una de las dos familias podría haber recibido ventaja, y
dónde se comprueba:

| Canal | Estado | Verificación |
|---|---|---|
| Término de rollout en la pérdida | ausente en ambos | `test_shared_objective.py::test_training_loss_has_no_rollout_term` |
| `rollout_loss` conectada al runner | no lo está | `..._is_never_wired_into_training` |
| Clave de rollout en el bloque `train:` | ninguna configuración | `..._experiment_configs_declare_no_rollout_training_term` |
| Esquema de integración | mismo `q_{k+1} = q_k + dt v_{k+1}` | `..._same_semi_implicit_position_update` |
| Gravedad analítica | dada a ambos | `..._gravity_is_given_to_both_models_analytically` |
| Datos, targets y splits | infraestructura única | `test_shared_data.py`, `test_shared_targets.py` |
| Sampler, cuotas y semillas | idénticos | mismo `StratifiedSampler`, composición en el manifiesto |
| Grafo y consultas de pared | idénticos | `slgnn_experiments.scene` |
| Presupuesto de updates y ejemplos | 400 × 2 en ambos | `manifest.json` |
| Presupuesto de parámetros | 1.769e5 vs 1.715e5 (−3 %) | `manifest.json` |
| Selección de `lr` | misma rejilla y presupuesto por familia | `compare.py::select_lr` |
| Selección de checkpoint | por validación, nunca por test | `runner.train(on_best=...)` |
| `CASE07` | fuera de toda selección | `splits.assert_case07_protected` |
| Código de evaluación | mismo `evaluate_rollout` | runner común |
| Métricas de contacto | misma geometría neutral | `scene.active_contact_keys` |
| Fuga entre ejemplos del batch | cero en ambos | `test_no_batch_leakage.py` |
| Precisión numérica | `float64` en ambos | `torch.set_default_dtype` en los scripts |


## 4. La ablación de `λ_ΔL`

La auditoría encontró un sesgo que **perjudicaba a GNS**, no a v3.

Con `λ_ΔL = 0` la rotación no está supervisada para nadie. Para v3 eso es
inocuo: predice `ΔL = 0` exactamente, así que `ω` se queda congelada en su
valor inicial —incorrecto, pero acotado— y el término de pérdida es una
constante de gradiente nulo.

Para GNS no es inocuo. Su decoder rotacional queda libre, y —esta es la parte
que importa— `ω` es una **feature de entrada de su encoder de nodo**:

```python
raw = torch.cat([particles.v, particles.omega, scal, z, wall_feat], dim=-1)
```

v3 no tiene esa realimentación: `ω` solo le entra por las features cinemáticas
de contacto y en combinación simétrica (`|ω_i|+|ω_j|`, `||ω_i|−|ω_j||`). En
GNS, una `ω` que deriva corrompe todos los latentes de nodo del paso
siguiente, y de ahí contamina también la posición.

El efecto es grande y medible: el RMSE de `ω` de GNS pasa de `0.15` en h=1 a
`4.29` en h=25 —un factor 28—, mientras el de v3 se queda en `0.049`.

La ablación activa `λ_ΔL = 1` para las tres variantes. Es **justa por
construcción**: no puede cambiar nada en v3 y solo puede ayudar a GNS. Que v3
salga bit a bit idéntico es la comprobación de que la ablación hace lo que
dice.


### 60 esferas con gravedad (experimento principal)

| Métrica | Variante | `λ_ΔL = 0` | `λ_ΔL = 1` | cambio |
|---|---|---|---|---|
| RMSE `ω` rollout h=1 | v3-C | 0.00622 ± 0 | 0.00622 ± 0 | +0.0% |
| RMSE `ω` rollout h=1 | v3-I | 0.00622 ± 5e-19 | 0.00622 ± 0 | +0.0% |
| RMSE `ω` rollout h=1 | GNS controlado | 0.1522 ± 0.065 | 0.00704 ± 0.00011 | -95.4% |
| RMSE `ω` rollout h=25 | v3-C | 0.0487 ± 0 | 0.0487 ± 0 | +0.0% |
| RMSE `ω` rollout h=25 | v3-I | 0.0487 ± 0 | 0.0487 ± 0 | +0.0% |
| RMSE `ω` rollout h=25 | GNS controlado | 4.291 ± 1.7 | 0.08515 ± 0.011 | -98.0% |
| RMSE `q` rollout h=1 | v3-C | 0.0004313 ± 3.9e-06 | 0.0004313 ± 3.9e-06 | +0.0% |
| RMSE `q` rollout h=1 | v3-I | 0.0005473 ± 1.8e-05 | 0.0005473 ± 1.8e-05 | +0.0% |
| RMSE `q` rollout h=1 | GNS controlado | 0.0005412 ± 7.7e-07 | 0.0005398 ± 5.5e-07 | -0.2% |
| RMSE `q` rollout h=10 | v3-C | 0.01935 ± 0.00034 | 0.01935 ± 0.00034 | +0.0% |
| RMSE `q` rollout h=10 | v3-I | 0.02303 ± 0.0082 | 0.02303 ± 0.0082 | +0.0% |
| RMSE `q` rollout h=10 | GNS controlado | 0.02906 ± 0.0011 | 0.02809 ± 0.00013 | -3.3% |
| RMSE `q` rollout h=25 | v3-C | 0.05255 ± 0.00083 | 0.05255 ± 0.00083 | +0.0% |
| RMSE `q` rollout h=25 | v3-I | 0.07462 ± 0.026 | 0.07462 ± 0.026 | +0.0% |
| RMSE `q` rollout h=25 | GNS controlado | 0.133 ± 0.045 | 0.09244 ± 0.001 | -30.5% |
| RMSE `Δp` un paso | v3-C | 0.007807 ± 6.8e-05 | 0.007807 ± 6.8e-05 | +0.0% |
| RMSE `Δp` un paso | v3-I | 0.008022 ± 0.0018 | 0.008022 ± 0.0018 | +0.0% |
| RMSE `Δp` un paso | GNS controlado | 0.009111 ± 9.9e-06 | 0.009101 ± 4.6e-06 | -0.1% |
| RMSE `Δp` extrapolación | v3-C | 0.01064 ± 6.3e-05 | 0.01064 ± 6.3e-05 | +0.0% |
| RMSE `Δp` extrapolación | v3-I | 0.01068 ± 0.0017 | 0.01068 ± 0.0017 | +0.0% |
| RMSE `Δp` extrapolación | GNS controlado | 0.0117 ± 4.4e-06 | 0.0117 ± 5.9e-06 | -0.1% |
| RMSE `q` h=25 extrapolación | v3-C | 0.07612 ± 0.0012 | 0.07612 ± 0.0012 | +0.0% |
| RMSE `q` h=25 extrapolación | v3-I | 0.09952 ± 0.035 | 0.09952 ± 0.035 | +0.0% |
| RMSE `q` h=25 extrapolación | GNS controlado | 0.1587 ± 0.042 | 0.1226 ± 0.0012 | -22.7% |

- v3-C: entrenamiento **idéntico** al activar `λ_ΔL`.
- v3-I: entrenamiento **idéntico** al activar `λ_ΔL`.
- Mejor perfil de v3 en este experimento: **v3-C** (0.05255 frente a 0.09244 de GNS, factor **1.76x**).
- v3-C frente a GNS en rollout h=25: factor **1.76x** tras la ablación, frente a 2.53x antes. La ablación reduce la ventaja de v3 y **no** cambia el orden.

### dos esferas, colisión oblicua

| Métrica | Variante | `λ_ΔL = 0` | `λ_ΔL = 1` | cambio |
|---|---|---|---|---|
| RMSE `ω` rollout h=1 | v3-C | 0.002198 ± 0 | 0.002198 ± 0 | +0.0% |
| RMSE `ω` rollout h=1 | v3-I | 0.002198 ± 0 | 0.002198 ± 0 | +0.0% |
| RMSE `ω` rollout h=1 | GNS controlado | 0.0601 ± 0.012 | 0.04451 ± 0.021 | -25.9% |
| RMSE `ω` rollout h=25 | v3-C | 0.04678 ± 0 | 0.04678 ± 0 | +0.0% |
| RMSE `ω` rollout h=25 | v3-I | 0.04678 ± 0 | 0.04678 ± 0 | +0.0% |
| RMSE `ω` rollout h=25 | GNS controlado | 1.457 ± 0.51 | 0.9887 ± 0.7 | -32.1% |
| RMSE `q` rollout h=1 | v3-C | 0.005209 ± 3.3e-06 | 0.005209 ± 3.3e-06 | +0.0% |
| RMSE `q` rollout h=1 | v3-I | 0.005234 ± 4.8e-06 | 0.005234 ± 4.8e-06 | +0.0% |
| RMSE `q` rollout h=1 | GNS controlado | 0.005272 ± 1.9e-05 | 0.00528 ± 3.3e-05 | +0.1% |
| RMSE `q` rollout h=10 | v3-C | 0.004686 ± 0.00057 | 0.004686 ± 0.00057 | +0.0% |
| RMSE `q` rollout h=10 | v3-I | 0.01351 ± 0.00057 | 0.01351 ± 0.00057 | -0.0% |
| RMSE `q` rollout h=10 | GNS controlado | 0.03639 ± 0.0075 | 0.04159 ± 0.0072 | +14.3% |
| RMSE `q` rollout h=25 | v3-C | 0.02608 ± 0.0027 | 0.02608 ± 0.0027 | +0.0% |
| RMSE `q` rollout h=25 | v3-I | 0.07053 ± 0.0023 | 0.07053 ± 0.0023 | +0.0% |
| RMSE `q` rollout h=25 | GNS controlado | 0.1886 ± 0.037 | 0.2034 ± 0.049 | +7.9% |
| RMSE `Δp` un paso | v3-C | 0.003511 ± 0.00043 | 0.003511 ± 0.00043 | +0.0% |
| RMSE `Δp` un paso | v3-I | 0.008699 ± 0.00018 | 0.008699 ± 0.00018 | +0.0% |
| RMSE `Δp` un paso | GNS controlado | 0.01105 ± 0.00064 | 0.01189 ± 0.0011 | +7.6% |
| RMSE `Δp` extrapolación | v3-C | 0.01277 ± 0.0011 | 0.01277 ± 0.0011 | +0.0% |
| RMSE `Δp` extrapolación | v3-I | 0.01801 ± 7.1e-05 | 0.01801 ± 7.1e-05 | +0.0% |
| RMSE `Δp` extrapolación | GNS controlado | 0.02075 ± 0.00056 | 0.02057 ± 0.00069 | -0.9% |
| RMSE `q` h=25 extrapolación | v3-C | 0.05307 ± 0.0027 | 0.05307 ± 0.0027 | +0.0% |
| RMSE `q` h=25 extrapolación | v3-I | 0.1042 ± 0.0033 | 0.1042 ± 0.0033 | -0.0% |
| RMSE `q` h=25 extrapolación | GNS controlado | 0.2911 ± 0.048 | 0.3024 ± 0.045 | +3.9% |

- v3-C: entrenamiento **idéntico** al activar `λ_ΔL`.
- v3-I: entrenamiento **idéntico** al activar `λ_ΔL`.
- Mejor perfil de v3 en este experimento: **v3-C** (0.02608 frente a 0.2034 de GNS, factor **7.80x**).
- v3-C frente a GNS en rollout h=25: factor **7.80x** tras la ablación, frente a 7.23x antes. La ablación no reduce la ventaja de v3 y **no** cambia el orden.

### una esfera contra pared

| Métrica | Variante | `λ_ΔL = 0` | `λ_ΔL = 1` | cambio |
|---|---|---|---|---|
| RMSE `ω` rollout h=1 | v3-C | 0 ± 0 | 0 ± 0 | — |
| RMSE `ω` rollout h=1 | v3-I | 0 ± 0 | 0 ± 0 | — |
| RMSE `ω` rollout h=1 | GNS controlado | 0.01776 ± 0.0039 | 0.002072 ± 0.0014 | -88.3% |
| RMSE `ω` rollout h=25 | v3-C | 0.01164 ± 0 | 0.01164 ± 0 | +0.0% |
| RMSE `ω` rollout h=25 | v3-I | 0.01164 ± 0 | 0.01164 ± 0 | +0.0% |
| RMSE `ω` rollout h=25 | GNS controlado | 0.5266 ± 0.21 | 0.05592 ± 0.025 | -89.4% |
| RMSE `q` rollout h=1 | v3-C | 0.002423 ± 0 | 0.002423 ± 0 | +0.0% |
| RMSE `q` rollout h=1 | v3-I | 0.002423 ± 0 | 0.002423 ± 0 | +0.0% |
| RMSE `q` rollout h=1 | GNS controlado | 0.002496 ± 1.4e-05 | 0.002494 ± 9.9e-06 | -0.1% |
| RMSE `q` rollout h=10 | v3-C | 0.002003 ± 5.8e-05 | 0.002003 ± 5.8e-05 | +0.0% |
| RMSE `q` rollout h=10 | v3-I | 0.0005279 ± 2.4e-05 | 0.0005279 ± 2.4e-05 | +0.0% |
| RMSE `q` rollout h=10 | GNS controlado | 0.008015 ± 0.0014 | 0.0075 ± 0.00088 | -6.4% |
| RMSE `q` rollout h=25 | v3-C | 0.02711 ± 0.0019 | 0.02711 ± 0.0019 | +0.0% |
| RMSE `q` rollout h=25 | v3-I | 0.01523 ± 0.00017 | 0.01523 ± 0.00017 | +0.0% |
| RMSE `q` rollout h=25 | GNS controlado | 0.05288 ± 0.0039 | 0.04924 ± 0.0034 | -6.9% |
| RMSE `Δp` un paso | v3-C | 0.003785 ± 0.00016 | 0.003785 ± 0.00016 | +0.0% |
| RMSE `Δp` un paso | v3-I | 0.003117 ± 1.3e-05 | 0.003117 ± 1.3e-05 | +0.0% |
| RMSE `Δp` un paso | GNS controlado | 0.003371 ± 0.00011 | 0.003356 ± 0.00013 | -0.4% |
| RMSE `Δp` extrapolación | v3-C | 0.00647 ± 0.00028 | 0.00647 ± 0.00028 | +0.0% |
| RMSE `Δp` extrapolación | v3-I | 0.005374 ± 2.3e-05 | 0.005374 ± 2.3e-05 | +0.0% |
| RMSE `Δp` extrapolación | GNS controlado | 0.005594 ± 0.00018 | 0.005607 ± 0.00031 | +0.2% |
| RMSE `q` h=25 extrapolación | v3-C | 0.04704 ± 0.005 | 0.04704 ± 0.005 | +0.0% |
| RMSE `q` h=25 extrapolación | v3-I | 0.03384 ± 0.00014 | 0.03384 ± 0.00014 | +0.0% |
| RMSE `q` h=25 extrapolación | GNS controlado | 0.05132 ± 0.0032 | 0.04524 ± 0.0029 | -11.8% |

- v3-C: entrenamiento **idéntico** al activar `λ_ΔL`.
- v3-I: entrenamiento **idéntico** al activar `λ_ΔL`.
- Mejor perfil de v3 en este experimento: **v3-I** (0.01523 frente a 0.04924 de GNS, factor **3.23x**).
- v3-C frente a GNS en rollout h=25: factor **1.82x** tras la ablación, frente a 1.95x antes. La ablación reduce la ventaja de v3 y **no** cambia el orden.

---

## 5. Conclusión

**La respuesta a la pregunta original es no**: la ventaja de v3 en rollout no
viene de la función de pérdida. Ambos modelos entrenaron contra el mismo
objetivo de un paso, y hay tests que lo impiden cambiar en silencio.

**Pero la pregunta era productiva.** Al auditar el canal señalado apareció
otro que iba en dirección contraria: `λ_ΔL = 0` dejaba el canal rotacional de
GNS sin supervisar, y como `ω` realimenta su encoder, eso inflaba
artificialmente la ventaja de v3 en rollout largo. Corregido, la ventaja de
v3-C **se reduce alrededor de un tercio pero no desaparece**, y el orden entre
las tres variantes no cambia en ningún experimento.

Como subproducto, la varianza entre semillas de GNS se desploma: la `ω` sin
supervisar era también la fuente de su inestabilidad.

## 6. Qué queda sin controlar

Honestamente, lo que esta auditoría **no** descarta:

1. **Presupuestos pequeños.** 400 actualizaciones y ventanas de 500 snapshots.
   Las diferencias son consistentes entre semillas y entre tres experimentos,
   pero no se ha demostrado que sobrevivan al escalar.
2. **Capacidad frente a arquitectura.** Los parámetros están equiparados al
   3 %, pero eso no garantiza capacidad efectiva equiparada: v3 gasta parte de
   la suya en cuadratura y solver, GNS toda en el decoder.
3. **Coste computacional.** GNS es 1.6x más rápido por paso de inferencia. Con
   un criterio de error por segundo, la conclusión podría cambiar.
4. **Un solo baseline principal.** `GNSClassicReduced` está implementado y
   probado, pero no se ha corrido como comparación: cambia historia, target y
   representación a la vez, y con tres variables movidas una diferencia no
   sería atribuible.

## 7. Consecuencia para la fase siguiente

Cuando se active `L_roll` para v3 —que la formulación pide (§13.7)— **hay que
activarlo también para GNS**, o la comparación deja de medir el sesgo
inductivo y pasa a medir quién entrena contra la métrica de evaluación.

`tests/comparison/test_shared_objective.py` falla si eso ocurre solo en un
lado. Es el mecanismo que hace que esta auditoría no haya que repetirla a
mano.

## 8. Reproducir

```bash
python -m pytest tests/comparison -q

python scripts/slgnn_v3/compare.py \
  --experiment configs/experiments/gravity60_small_dL.yaml \
  --seeds 0 1 2 --tune-lr --final-test

python scripts/slgnn_v3/make_fairness_report.py
```

