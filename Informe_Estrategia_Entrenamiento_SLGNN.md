# Informe: Estrategia de entrenamiento de SLGNN-v2

**Proyecto:** SLGNN (SDF–Lagrangian Graph Neural Network) — MAT2320
**Alcance de este documento:** cómo entrenar la implementación ya existente
en `src/slgnn/`, con qué datos, en qué orden, con qué pérdidas y pesos, y qué
esperar de cómputo en CPU. No repite la arquitectura (ver `SLGNN_versión2.txt`)
ni el esquema de datos (ver `data/DATA_NOTES.md`); parte de ambos como dados.

---

## 1. Punto de partida

Hoy existe: el modelo completo (`slgnn.SLGNN`), 31 tests que verifican las
garantías de construcción (§37: equivariancia SO(3), conservación de momento,
disipación no positiva, doble backward entrenable), y un smoke de
entrenamiento (`scripts/mini_train.py`) que hace overfit del benchmark de 2
esferas. Ese smoke demuestra que el mecanismo de entrenamiento *funciona*
(pérdida cae, rollout de 100 pasos atraviesa la colisión con RMSE razonable),
pero no es todavía un entrenamiento real: usa una sola trayectoria, sin
normalización de datos, sin curriculum de canales, sin pesos de pérdida
ajustados, y sin el canal de Rayleigh ni el histórico puestos a prueba con
datos reales de disipación.

Este informe describe cómo pasar de ese smoke a un entrenamiento real
siguiendo el plan de etapas que la propia arquitectura prescribe (§35 de
`SLGNN_versión2.txt`), adaptado a lo que el código ya ofrece y a lo que falta
por agregar.

---

## 2. Datos disponibles y su rol en el entrenamiento

| Dataset | Rol de entrenamiento | Régimen físico |
|---|---|---|
| `Benchmark_2Spheres_Oblique_Collision` (1x/2x/4x) | Validación aislada del canal V_pp + Rayleigh pp (sin pared) | sin gravedad, sin pared |
| `Benchmark_1Sphere_Multiple_Wall_Collision` (10°-90°) | Validación aislada del canal V_pW + Rayleigh pW (sin partícula-partícula) | sin gravedad, pared plana |
| `60Spheres_Homogeneous...` CASE01-05 | **Entrenamiento principal, Etapa I-III** | sin gravedad, caja estática, contacto homogéneo |
| `60Spheres_Homogeneous...` CASE06 | Validación (misma distribución) | ídem |
| `60Spheres_Homogeneous...` CASE07 | Test de extrapolación (energía ~3x) | ídem |
| `60Spheres_Homogeneous...` CASE08-09 | No documentados por el PDF fuente — **excluir del split oficial** hasta confirmar que son del mismo experimento (ver `DATA_NOTES.md` pendiente) | esquema distinto (sin `Diameter`) |
| `60Spheres_Gravity...` CASE01-07 | Segunda fase de entrenamiento, con gravedad y fricción heterogénea pp vs pW | con gravedad, kn/μ/e distintos por canal |
| `Extrapolation_2073Spheres...` (cilindro) | **Solo inferencia, nunca entrenamiento** — evaluación de generalización a frontera móvil | con gravedad, pared rotatoria |

Los dos benchmarks pequeños no son para entrenar los pesos principales del
modelo en el sentido de "muchas épocas" — son demasiado cortos y demasiado
simples (2 y 1 partícula) para que el modelo aprenda algo generalizable de
ellos solo. Su rol correcto es el que ya cumple `mini_train.py`: un **test de
sobreajuste rápido** que confirma que el mecanismo (doble backward, canal
correspondiente) funciona antes de pasar a datos reales de 60 esferas, donde
el entrenamiento de verdad ocurre.

---

## 3. Decisión previa: ¿un modelo o dos regímenes?

`60Spheres_Homogeneous` (sin gravedad) y `60Spheres_Gravity` (con gravedad,
fricción heterogénea pp≠pW) tienen **parámetros de contacto DEM distintos**
(k_n, μ_s, e, η_t/η_n — ver tabla en `DATA_NOTES.md`). El modelo no recibe
esos parámetros como entrada hoy: `Particles.props` está vacío
(`n_props=0`) en el smoke actual. Esto significa que si se mezclan ambos
datasets en un solo entrenamiento sin más, el modelo no tiene forma de saber
en qué régimen de fricción está y promediará ambos comportamientos de forma
incorrecta.

Hay dos caminos, y recomiendo el primero para esta fase:

- **(a) Dos modelos separados**, uno por archivo/régimen. Es el camino simple,
  ya soportado por el código sin cambios: basta entrenar dos instancias de
  `SLGNN` con `n_props=0`, una por dataset. Es la ruta recomendada para
  llegar rápido a resultados comparables con el GNS baseline del paper de
  origen (que hace lo mismo: entrena sobre 60 esferas homogéneo).
- **(b) Un solo modelo con propiedades materiales como entrada**, extendiendo
  `Particles.props` para incluir (k_n, μ_s, e, η_t/η_n) por partícula o por
  caso, y subiendo `SLGNNConfig.n_props` en consecuencia. Es más fiel al
  espíritu de "un solo operador dinámico" de la arquitectura, pero es trabajo
  adicional (hay que decidir si son propiedades por partícula o un embedding
  global de "caso") y no aporta nada al objetivo inmediato de M6-M7.

Empezar por (a) y dejar (b) para cuando haya que unificar con el cilindro
rotatorio (que comparte parámetros con `60Spheres_Gravity`, así que en
principio el modelo entrenado en gravedad ya sirve de base para esa
extrapolación sin necesitar (b)).

---

## 4. Estrategia por etapas (curriculum)

Sigue el orden de §35 de `SLGNN_versión2.txt`, con el mapeo concreto a los
datasets de la sección 2. Cada etapa parte de los pesos de la anterior
(`torch.save`/`torch.load` del `state_dict`).

### Etapa 0 — ya completada
Smoke sobre `Benchmark_2Spheres_Oblique_Collision`. Sirve como test de
regresión rápido (< 2 min) antes de cada cambio de arquitectura, no como
entrenamiento real.

### Etapa I — canal conservativo
Entrenar solo con pérdida de aceleración (`losses.acceleration_loss` +
`angular_acceleration_loss`) sobre `60Spheres_Homogeneous` CASE01-05,
usando pasos individuales (no rollout). Aquí es donde conviene la
restricción a "estados con baja disipación" que menciona §35: los primeros
pasos de cada trayectoria (antes de que las colisiones inelásticas hayan
disipado energía) son los que mejor aíslan la parte conservativa.

**Gap de código a resolver primero:** el modelo actual entrena los tres
canales (V, R, H) simultáneamente desde la inicialización — no hay forma de
"apagar" Rayleigh o el canal V durante esta etapa. El canal H ya nace en cero
(`nets.zero_init_last`, correcto para Etapa I-III), pero Rayleigh no. Antes
de esta etapa conviene agregar un flag simple (p. ej. `SLGNNConfig.active =
{"V", "R", "H"}` o similar) que permita anular la contribución de un canal
en `model.py` sin tocar sus pesos, para poder desactivar R durante la Etapa I
tal como pide la especificación. Es un cambio pequeño y localizado.

### Etapa II — Rayleigh
Activar el canal R (dejar de anularlo), manteniendo `f_H = 0`. Mismo dataset,
misma pérdida de aceleración de un paso. Los coeficientes de Rayleigh parten
de `rayleigh_bias_init=-2.0` (softplus(-2)≈0.13), un valor de disipación
inicial pequeño pero no nulo — razonable como punto de partida para que el
gradiente no sea idénticamente cero al empezar esta etapa.

### Etapa III — rollout corto
Introducir `losses.rollout_loss` con horizonte H=2-5 pasos
(`integrator.rollout(..., keep_graph=True)`), más ruido gaussiano pequeño en
posición/velocidad de entrada (aún no implementado — agregar como una función
simple `q + sigma_noise * torch.randn_like(q)` antes de cada forward de
entrenamiento). El ruido es importante: en inferencia el modelo se alimenta
de sus propias predicciones imperfectas, y sin ruido de entrenamiento el
modelo nunca ve ese tipo de error.

### Etapa IV — canal histórico
Reactivar el canal H (dejar de forzarlo a cero — hoy nace en cero por
`zero_init_last`, así que basta con seguir entrenando: sus gradientes ya no
están bloqueados, solo parten de cero). Agregar `residual_regularization`
(L_res) y `passivity_loss` (L_pass) a la pérdida total con pesos altos al
principio, para que el modelo siga prefiriendo explicar la dinámica con V y R
antes que con H.

### Etapa V — rollout largo
Incrementar H progresivamente (curriculum: 5 → 10 → 20 → ... pasos),
agregando `penetration_loss` (L_pen). Extender de `60Spheres_Homogeneous` a
`60Spheres_Gravity` en esta etapa o en una etapa V.5 separada, repitiendo I-IV
si el régimen de gravedad lo requiere (recomiendo repetir el curriculum
completo para el segundo modelo, no solo saltar a rollout largo).

---

## 5. Función de pérdida y pesos por etapa

| Etapa | Pérdidas activas | Pesos sugeridos (punto de partida) |
|---|---|---|
| I | L_a, L_alpha | λ_a=1, λ_alpha=1 |
| II | + (R activo, misma pérdida) | igual |
| III | + L_roll | λ_roll=1, λ_v=1 (peso relativo v/q dentro de L_roll) |
| IV | + L_res, L_pass | λ_res=0.1 inicial, decreciente; λ_pass=1 (alto, es una restricción dura) |
| V | + L_pen | λ_pen=0.1, subir si aparecen penetraciones en rollout de validación |

Estos son puntos de partida razonables, no valores calibrados — no existe
aún una corrida real que los valide. La señal para ajustar cada λ es
observar el diagnóstico (`out.diagnostics`) de cada corrida: si `R_pp`/`R_pW`
se disparan sin bajar la pérdida de aceleración, subir λ_pass; si `f_H` no
baja de la magnitud de `f_cons`+`f_R`, subir λ_res.

---

## 6. Normalización y escalas

`slgnn.data.default_scales()` ya adimensionaliza longitud (diámetro de
partícula), tiempo (0.01 s) y masa (masa de una partícula) — usado en
`mini_train.py`. Falta el segundo nivel de normalización que pide §34.1-34.2:
las `sigma_a`, `sigma_alpha` (y `sigma_q`, `sigma_v` de L_roll) están hoy
fijas en 1.0 por defecto en `losses.py`. Antes de Etapa I, calcular la
desviación estándar real de `a_ref`/`alpha_ref` sobre CASE01-05 (ya
adimensionalizadas) y usarla como `sigma`, para que ninguna componente de la
pérdida domine por una diferencia de escala incidental.

---

## 7. Optimización

- **Optimizador:** Adam, como en el smoke. lr inicial 3e-3 para Etapa I-II
  (pocos parámetros activos importan), bajar a 1e-3 o menos al activar
  rollout (Etapa III en adelante), porque el horizonte multi-paso amplifica
  el ruido de gradiente del doble backward.
- **Gradient clipping:** no está en el smoke actual y conviene agregarlo
  (`torch.nn.utils.clip_grad_norm_`) desde Etapa III — las ventanas C² y la
  softplus con `beta=20` cerca del contacto pueden producir gradientes
  puntualmente grandes.
- **Batching:** `mini_train.py` hoy hace un loop Python sobre un minibatch de
  timesteps, cada uno con su propio forward (sin batching real entre
  grafos). Para acelerar en CPU sin reintroducir `torch_geometric.Batch`
  (que arrastraría la dependencia de `torch_cluster`/`torch_scatter` que se
  evitó deliberadamente, ver comentario en `requirements.txt`), la opción
  práctica es batchear "a mano": concatenar N sistemas trasladando cada uno
  lejos de los demás (más allá de `r_list`) para que `neighbor_pairs` no
  cree aristas espurias entre sistemas, y correr un solo forward sobre el
  conjunto. No es necesario para Etapa I-II con datasets chicos, pero vale
  la pena antes de Etapa III-V si el loop Python resulta ser el cuello de
  botella (medir primero, no asumir).

---

## 8. Cómputo en CPU: qué esperar

El único dato empírico disponible es el smoke: 300 iteraciones con
minibatch=8 sobre 2 partículas (1 arista) tardaron ~80 s (~33 ms por
forward+backward). Escalar a 60 esferas cambia el costo dominante: con
`r_off=1.5` diámetros y 60 esferas confinadas en una caja de 6 diámetros de
lado, el número de aristas será del orden de cientos, no de 1 — el costo por
paso crecerá bastante más que linealmente respecto del smoke (más aristas
implica más llamadas a las MLP de arista y mensajes, además de que el doble
backward escala con el tamaño del grafo computacional).

**No estimo aquí una cifra de horas** porque sería una extrapolación
demasiado especulativa desde un solo punto de datos. La recomendación
concreta es: antes de comprometer un presupuesto de entrenamiento, correr 5-10
iteraciones reales sobre CASE01 de `60Spheres_Homogeneous` y medir el tiempo
por iteración directamente — con eso se puede presupuestar el resto con
confianza. Si resulta demasiado lento para CPU, las palancas son (en orden de
impacto esperado): reducir `hidden`/`layers` en `SLGNNConfig`, subsamplear
partículas (entrenar con un subconjunto de las 60 esferas por paso), reducir
`r_list`/`r_off` si la densidad de contactos lo permite, y el batching manual
de la sección 7.

---

## 9. Validación y criterio de "listo"

Usar CASE06 (validación, misma distribución) y CASE07 (extrapolación
energética) de cada dataset como en el paper de origen. Métricas:

- RMSE de posición/velocidad a horizonte fijo (§16.1).
- Pasos de rollout antes de divergencia o penetración sostenida — usar
  `out.diagnostics["phi"]` para detectar penetración (φ < R).
  numérica.
- Diagnóstico energético: `V_g + V_pp + V_pW + T` no debería crecer sin
  límite en ausencia de pared móvil (aquí no hay inyección de energía
  externa salvo gravedad, que es conservativa) — un crecimiento sostenido en
  rollout largo es señal de que R o H están inyectando energía por error de
  entrenamiento, contradiciendo la garantía de disipación no positiva
  verificada en los tests (que solo se verifica para pesos aleatorios, no
  post-entrenamiento).
- Comparación contra el baseline GNS caracterizado en el paper de origen
  (que ya reporta que GNS "caja negra" se desestabiliza en la frontera
  rotatoria) — pendiente hasta implementar el baseline (fuera del alcance de
  este informe).

"Listo" para pasar de una etapa a la siguiente: pérdida de la etapa actual
estabilizada (no necesariamente mínima) y sin señales de energía espuria en
un rollout corto de validación.

---

## 10. Reproducibilidad

Fijar semilla (`torch.manual_seed`) y guardar por corrida: config (YAML desde
`SLGNNConfig` — ya es un dataclass, trivial de serializar), sigmas de
normalización calculadas, split de CASEs usado, y el `state_dict` al final de
cada etapa (no solo al final del entrenamiento completo, para poder retomar
sin repetir etapas si algo falla más adelante).

---

## 11. Riesgos específicos de esta arquitectura

- **Costo del doble backward en rollouts largos:** mantener el grafo de
  autograd a través de muchos pasos de rollout (`keep_graph=True`) crece
  linealmente en memoria con el número de pasos. Si Etapa V se vuelve
  pesada, la mitigación estándar es backprop-through-time truncado
  (`detach().requires_grad_(True)` cada K pasos), a costa de perder
  gradiente exacto sobre el horizonte completo.
- **Discontinuidades de contacto:** las ventanas C² ayudan pero no eliminan
  el problema; vigilar picos de gradiente cerca de $d \to r_{on}$ y $g \to
  0$ (razón adicional para el clipping de la sección 7).
- **Ambigüedad de ω(t) del cilindro:** irrelevante para el entrenamiento
  (que no toca el cilindro), pero relevante para cuando este mismo modelo se
  use en inferencia sobre la extrapolación — queda documentado en
  `DATA_NOTES.md` como pendiente de verificación antes de esa fase.
- **CASE08/09 no documentados:** no incluir en el split de entrenamiento
  hasta resolver la duda de la sección 2 — mezclar datos de esquema/origen
  incierto en el entrenamiento principal complicaría cualquier diagnóstico
  posterior.

---

## 12. Próximos hitos concretos

- **M6:** agregar el flag de canales activos (sección 4, Etapa I) y las
  `sigma` calculadas desde datos (sección 6). Entregable: `train.py` +
  `configs/etapa1.yaml`.
- **M7:** correr Etapas I-III sobre `60Spheres_Homogeneous`, con la medición
  de tiempo por iteración de la sección 8 hecha *antes* de comprometer el
  presupuesto completo. Entregable: checkpoint + curva de pérdida + rollout
  de validación en CASE06.
- **M8:** Etapas IV-V sobre el mismo dataset; repetir I-V para
  `60Spheres_Gravity` (modelo separado, sección 3).
- **M9:** baseline GNS (adaptando `geoelements/gns` como estaba previsto en
  el informe de sesión original) para comparación cuantitativa.
- **M10:** inferencia (solo) del modelo entrenado en gravedad sobre el
  cilindro rotatorio, resolviendo antes la ambigüedad de ω(t).
