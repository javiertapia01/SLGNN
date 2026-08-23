# EXPERIMENT_PROTOCOL — SLGNN-v3 MVP

Protocolo de los experimentos del MVP. Está escrito para que el resultado sea
interpretable **antes** de conocerlo: si v3 pierde contra GNS bajo estas
reglas, es un resultado; si gana, también.

---

## 1. Reglas de equidad

Implementadas en código, no confiadas al operador:

| Regla | Dónde se garantiza |
|---|---|
| Mismos splits y transiciones | `slgnn_experiments.splits`, un solo `Split` por experimento |
| Mismos targets, construidos una vez | `slgnn_experiments.targets`, consumido por ambos |
| Mismo sampler y mismas cuotas | `slgnn_experiments.sampling`, misma semilla |
| Mismo grafo y mismas consultas de pared | `slgnn_experiments.scene` |
| Mismo bucle de entrenamiento | `slgnn_experiments.runner.train` |
| Mismo presupuesto de updates y ejemplos | `TrainConfig`, registrado en el manifiesto |
| Mejor checkpoint por **validación** | `runner.train(on_best=...)`, nunca por test |
| `CASE07` fuera de toda selección | `splits.assert_case07_protected` levanta `ValueError` |
| Presupuesto de parámetros comparable | reportado en cada manifiesto y en el informe |
| Tiempos de entrenamiento e inferencia | `train_seconds`, `seconds_per_step` |

### La pérdida de entrenamiento es de un paso, para las tres variantes

`λ_Δp = 1`, `λ_ΔL = 0`, y **ningún término de rollout**. Importa decirlo
explícitamente porque la formulación oficial sí contempla `L_roll` en la
pérdida total (§13.7): si ese término se activara solo para v3, su ventaja en
rollout largo dejaría de ser atribuible al sesgo inductivo —estaría
entrenando contra la métrica con la que se lo evalúa— y la comparación no
mediría nada.

`rollout_horizons` en los YAML de experimento es una clave de **evaluación**,
vive en la raíz y nunca entra en `TrainConfig`. `TrainConfig.lambda_rollout` y
`rollout_horizon` existen reservados para una fase futura y valen cero; el
runner no llama a `rollout_loss`.

`tests/comparison/test_shared_objective.py` lo ancla: falla si aparece un
término de rollout en la pérdida, si alguna configuración lo mete en el bloque
`train:`, si los modelos dejan de compartir el mismo update semiimplícito de
posición, o si dejan de recibir la gravedad analítica por igual.

### Selección de hiperparámetros

Una única rejilla de `lr` `{3e-3, 1e-3, 3e-4, 1e-4}`, la misma para las tres
variantes, con la misma semilla y el mismo número de corridas por familia.
Se elige por pérdida de validación.

Esto **no** es tuning asimétrico: es lo contrario. Un `lr` único para todos
penalizaría a una de las dos familias por una razón ajena a la arquitectura.
v3 arranca cerca del suelo de pérdida —tiene la gravedad y la unilateralidad
incorporadas— y tolera mal un paso grande; GNS parte de cero y lo necesita.
La rejilla, el presupuesto y el criterio son idénticos.

---

## 2. Jerarquía de escenarios

### Nivel 0 — pruebas analíticas, sin aprendizaje

Ejecutadas como tests, antes de entrenar nada
([tests/slgnn_v3/test_integrator.py](../../tests/slgnn_v3/test_integrator.py)):

1. partícula libre sin gravedad → `Delta p = 0` exacto;
2. partícula libre con gravedad → `Delta p = dt m g` exacto;
3. esfera contra pared fija, impacto normal;
4. dos masas, choque frontal → recupera `-(1+e) u_n^-/(1/m_i + 1/m_j)`;
5. cadena de tres partículas con dos contactos simultáneos → un solo solve
   acoplado de tamaño 2;
6. rotación y traslación global del sistema → equivarianza `SE(3)`;
7. cambio de velocidad de pared con SDF axisimétrica constante;
8. batch de dos sistemas geométricamente superpuestos → cero aristas cruzadas
   y resultado idéntico al de correrlos por separado.

**Gate:** los 105 tests de `tests/slgnn_v3` en `float64`.

### Nivel 1 — micro-overfit

Conjunto fijo y versionado: `sixty_gravity/CASE01`, ventana `[185, 265)`,
elegida **con la auditoría temporal**, no a ojo (el primer contacto del caso
aparece cerca del snapshot 190; un prefijo `[0, 120)` no contiene ninguno).

Criterios: la pérdida cae, los gradientes llegan a `V`, `Psi` e `I` cuando
corresponde, `e` y `kappa` se mueven, y ninguna cabeza recibe gradiente en el
régimen donde está desactivada.

> **El objetivo del 90–95 % de reducción no aplica a v3 tal cual.** v3 arranca
> con la gravedad analítica y sin fuerza espuria en vuelo libre, así que su
> pérdida inicial ya es ~40 veces menor que la **final** de GNS. La reducción
> relativa mide cuánta ignorancia tenía el modelo al empezar, no cuánto
> aprende. Se reportan las tres cifras —inicial, final y reducción— y se
> compara la **final**.

### Nivel 2 — una esfera contra pared

Prioridad al caso de 90 grados (impacto normal puro). Los ángulos oblicuos se
cargan para **diagnosticar** física pendiente: el error tangencial que
produzcan no cuenta como fallo del canal normal, porque la fricción llega en
la fase 9.

### Nivel 3 — dos esferas

`1x` entrenamiento, `2x` validación, `4x` extrapolación. Sistema aislado: el
test más limpio de conservación. El choque oblicuo **no** se declara resuelto.

### Nivel 4 — 60 esferas con gravedad, reducido

`CASE01–05` entrenamiento, `CASE06` validación, `CASE07` test. Ventana
`[150, 650)`, batch 2, 400 updates, horizontes 1/5/10/25, tres semillas.

---

## 3. Atribución del error: normal frente a tangencial

Todo modelo puramente normal solo puede aplicar a la partícula `i` una fuerza
en `span{n_alpha : alpha incidente a i}`. La componente de `Delta p`
ortogonal a ese subespacio es **inalcanzable para cualquier canal normal**,
por bien entrenado que esté.

`slgnn_experiments.metrics.reachable_decomposition` calcula esa proyección y
el runner la reporta como `err_normal_fraction` / `err_tangential_fraction`,
**medida igual para v3 y para GNS** (`scene.active_contact_normals`), para que
la separación no dependa de lo que cada modelo crea que es un contacto.

Es lo que distingue "no se ha aprendido" de "no se puede representar", y es la
única forma honesta de evaluar un MVP normal contra datos con fricción.

---

## 4. Métricas reportadas

| Categoría | Métricas |
|---|---|
| Estado | RMSE/MAE de `Delta p` y `Delta L`, por régimen |
| Régimen | `free`, `pp`, `pw`, `mixed` por separado; nunca solo el global |
| Contacto | precision, recall, F1, `g_min`, penetración máxima, tunneling |
| Impacto | `e` efectiva, `Lambda_n`, restitución aplicada |
| Física | error de momento lineal y angular **interno** (pared aparte) |
| Solver | residuo primal, complementariedad, iteraciones, condicionamiento |
| Rollout | curva completa a horizontes 1/5/10/25 |
| Estabilidad | pasos hasta NaN |
| Cómputo | parámetros, ms/paso, segundos de entrenamiento |

Los agregados incluyen media, desviación estándar **y** el valor por semilla
(`metrics.aggregate_seeds`).

---

## 5. Reproducibilidad

Cada corrida escribe `results/slgnn_v3/**/<run_id>/` con `config_resolved.yaml`,
`manifest.json`, `metrics.jsonl`, `summary.json`, `best.pt`, `last.pt`,
`environment.txt`.

El manifiesto incluye SHA de git y estado sucio, semilla, casos, hash de la
lista de índices, escalas, conteo de parámetros, presupuesto, composición real
del sampler, backend y tolerancias del solver, versiones y duración.

Los checkpoints llevan versión de esquema; cargar uno incompatible **falla con
explicación** en vez de meter pesos en otro modelo
(`checkpointing.load_checkpoint`).

---

## 6. Comandos

```bash
python -m pytest -q
python -m pytest tests/slgnn_v3 -q
python -m pytest tests/baselines tests/comparison -q

python scripts/slgnn_v3/audit_temporal.py --config configs/experiments/audit_temporal.yaml
python scripts/slgnn_v3/run_micro_overfit.py
python scripts/slgnn_v3/run_all_benchmarks.py

python scripts/slgnn_v3/train.py --model slgnn_v3 --config configs/slgnn_v3/mvp_c.yaml \
    --experiment configs/experiments/micro_overfit.yaml --smoke

python scripts/slgnn_v3/compare.py --experiment configs/experiments/gravity60_small.yaml \
    --models v3_c v3_i gns_controlled --seeds 0 1 2 --tune-lr --final-test
```

`--smoke` toca forward, backward, checkpoint y evaluación en segundos, y el
manifiesto lo marca como `SMOKE`: no es un resultado científico.
