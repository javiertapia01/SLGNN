# Informe de preparación para S04-N60-G-RP

**Proyecto:** entrenamiento en serie de SLGNN-v2  
**Fecha de corte:** 30 de julio de 2026  
**Objetivo:** establecer qué debe completarse antes de iniciar el entrenamiento formal
de SLGNN con 60 esferas y gravedad, conservando la trazabilidad y las competencias
adquiridas en PP2O, PW1 y N60 sin gravedad.

## 1. Dictamen ejecutivo

El benchmark gravitacional y la ruta física de inferencia están disponibles, pero
**todavía no corresponde iniciar S04 como una continuación promovida de la serie**.
El bloqueo no es la presencia de gravedad: es que S03-N60-ZG-RP terminó sin un
checkpoint promovido debido a olvido de PP2O y PW1.

El orden correcto es:

1. recuperar la retención perdida en S03 y producir un checkpoint promovido;
2. implementar la etapa S04 con carga estricta, replay y métricas apropiadas para gravedad;
3. certificar por completo los datos gravitacionales y sus unidades;
4. registrar el baseline y verificar que cada cabeza responde al canal físico correcto;
5. congelar el protocolo, las semillas y los gates antes de ejecutar el entrenamiento.

Sólo después de aprobar las cinco tareas se debe lanzar S04-N60-G-RP.

## 2. Estado comprobado

### 2.1 Lineage existente

```text
S00-PP2O
→ S01-PW1-RP
→ S02-JOINT-PP2O-PW1
→ S02-R2-preN60
→ S03-N60-ZG-RP
```

El último checkpoint formalmente promovido es:

```text
checkpoints/slgnn_v2/curriculum/S02_R2_preN60/
slgnn-v2__S00-PP2O__S01-PW1-RP__S02-JOINT-PP2O-PW1__S02-R2-preN60__seeds-20260741-43/
promoted-to-S03-N60-ZG-RP.pt
```

SHA-256 verificado:

```text
8e938285c9247190e3993a139e9603d30660720b02ec63c5a7c8102e269c359b
```

S03 produjo `best-joint.pt`, pero no produjo `promoted-to-next-stage.pt`.
El SHA-256 actual de `best-joint.pt` es:

```text
b8562e538b54dd3bdcb2cbc72c89d85a8bc3e443eebfc51c2fb89cc8adabcbb2
```

Este archivo puede utilizarse como candidato de diagnóstico y recuperación, pero
**no como padre promovido de S04**.

### 2.2 Razón exacta del bloqueo

| Gate | `best-joint` S03 | Requisito |
|---|---:|---:|
| Mediana PW de RMSE de velocidad | 0.6480 | ≤ 0.5289 |
| PP-1x / especialista | 1.399× | ≤ 1.10× |
| PP-2x / especialista | 2.973× | ≤ 1.10× |
| PP-4x / especialista | 3.435× | ≤ 1.10× |
| PP-4x absoluto | 5.590 | ≤ aproximadamente 1.790 |
| N60-ZG CASE06, 1500 pasos | completado | obligatorio |
| N60-ZG CASE07, 1500 pasos | no completado | auditoría final, no selección |

S03 sí aprendió dinámica multibody útil: en CASE06 sin gravedad, `best-joint`
completó los 1500 pasos y superó al control scratch. Sin embargo, la mejora N60
no compensa el olvido de los benchmarks anteriores.

### 2.3 Evidencia preliminar del benchmark con gravedad

La ruta disponible es:

```text
data/extracted/60Spheres_Gravity_Inside_Cuboidal_Enclosure/CASE01..CASE07
```

La inspección preliminar confirma:

- 7 casos;
- 1501 archivos CSV por caso;
- 60 partículas por snapshot;
- `Particle_ID` presente;
- densidad uniforme de `4000 kg/m³`;
- ausencia de la columna `Diameter`;
- diámetro documentado y usado por el loader: `0.005 m`;
- caja estática `[0,0.03]³ m`;
- `dt = 10⁻⁴ s`;
- gravedad en el eje `−y`;
- parámetros de contacto distintos para PP y PW.

El zero-shot actual de `best-joint` sobre CASE06 con gravedad completó los 1500
pasos. A H100 obtuvo RMSE `q/v/ω = 0.1122/0.8802/0.8662`; su penetración máxima
durante el rollout completo fue `1.089 d_p`. Esto demuestra que el camino de
inferencia funciona, pero no sustituye el baseline del futuro checkpoint promovido.

## 3. Principio físico que debe gobernar S04

La gravedad no debe ser aprendida por las cabezas de contacto. Es un término físico
conocido que se entrega explícitamente como:

```text
g_vec(SI)        = [0, -9.81, 0] m/s²
g_vec(adimensional) = [0, -0.1962, 0]
```

La distribución de responsabilidades debe ser:

| Componente | Responsabilidad |
|---|---|
| término externo `g_vec` | aceleración gravitacional conocida |
| cabezas PP, canales V/R/H | interacción partícula–partícula |
| cabezas PW, canales V/R/H | interacción partícula–pared |
| procesadores compartidos | representación geométrica y dinámica común |
| material encoder | masa, radio y tipo/material disponibles en la arquitectura |

En este dataset las partículas tienen el mismo diámetro y densidad. La
heterogeneidad relevante está en las leyes de contacto por canal:

- PP: `μs=0.1`, `kn=10000 N/m`, `ηt/ηn=0.3`, `e=0.95`;
- PW: `μs=0.3`, `kn=10000 N/m`, `ηt/ηn=0.5`, `e=0.9`.

La separación existente entre cabezas PP y PW es, por tanto, la estructura que debe
aprender esas diferencias. No se debe aumentar `hidden`, cambiar `n_props` ni
introducir otra arquitectura dentro de esta etapa, porque se rompería la carga
estricta y dejaría de ser una transferencia controlada.

## Tarea 1 — Recuperar S03 y obtener un checkpoint promovido

### Objetivo

Recuperar PP2O y PW1 sin destruir la competencia N60-ZG, y producir:

```text
promoted-to-S04-N60-G-RP.pt
```

La subetapa propuesta es:

```text
S03-N60-ZG-RP → S03-R1-RETENTION
```

### Instrucciones

1. Crear:

   ```text
   configs/slgnn_v2/curriculum/S03_R1_retention.yaml
   scripts/slgnn_v2/recover_S03_retention.py
   ```

2. Tratar como fuentes inmutables:

   - el checkpoint promovido S02-R2, que conserva PP/PW;
   - `best-joint.pt` de S03, que contiene la competencia N60-ZG.

3. Verificar antes de cualquier operación:

   - SHA-256 de ambos archivos;
   - igualdad exacta de claves y shapes del `state_dict`;
   - arquitectura `hidden=16`, `layers=2`, `use_history=true`;
   - carga `strict=True`;
   - 24 754 parámetros.

4. Ejecutar primero un diagnóstico sin entrenamiento:

   - interpolación en espacio de pesos entre S02-R2 y S03 `best-joint`;
   - malla inicial recomendada `α ∈ {0.00, 0.25, 0.50, 0.75, 1.00}`;
   - auditoría de cada interpolación en N60-ZG CASE06, PP2O y PW1;
   - trasplante diagnóstico separado de cabezas PP y PW para localizar si el
     olvido reside en las cabezas, en los procesadores compartidos o en ambos.

   Los modelos híbridos de diagnóstico no se promueven automáticamente. Sirven para
   escoger una inicialización verificable.

5. Afinar los candidatos Pareto con una consolidación de replay intensivo.
   Como punto de partida pre-registrado para cada batch lógico:

   - 40% N60-ZG;
   - 30% PP2O;
   - 30% PW1.

   Dentro de PP2O se mantienen `1x`, `2x`, `4x`, con sobremuestreo moderado de
   `4x`. Dentro de PW1 se balancean `10°`, `30°`, `45°`, `60°` y `90°`.

6. Entrenar todos los módulos, pero con tasas conservadoras y discriminativas.
   La búsqueda debe centrarse en las cabezas antes de mover agresivamente los
   procesadores. Un punto de partida coherente con S02-R2 es:

   - cabezas PP: `3e-5`;
   - cabezas PW: `1e-5` a `3e-5`;
   - procesadores V/R/H: `1e-5`;
   - material encoder: `2e-6`;
   - AdamW nuevo, `weight_decay=1e-6`, clipping `3.0`;
   - LR de rollout igual a la mitad;
   - L2-SP respecto al checkpoint S02-R2.

   Estos valores deben quedar fijados en la configuración antes de lanzar las
   semillas. Cualquier cambio posterior sólo puede usar CASE06 y los benchmarks de
   retención, nunca CASE07.

7. Usar al menos tres semillas de recuperación y guardar checkpoints independientes
   por fase. Una consolidación corta debe incluir one-step y rollouts
   H4→H8→H16→H32; H64 se usa para confirmar la competencia multibody.

### Criterios de aceptación

S03-R1 queda aprobado únicamente si un mismo checkpoint:

- tiene PW mediana ≤ `0.5289`;
- deja cada caso PP ≤ `1.10×` su especialista;
- deja PP-4x ≤ aproximadamente `1.790`;
- completa N60-ZG CASE06 a H100, H250, H500 y 1500 sin NaN ni divergencia;
- cumple la estabilidad pre-registrada de H500, incluida penetración máxima
  ≤ `0.75 d_p`;
- no fue seleccionado observando CASE07;
- carga estrictamente al terminar;
- conserva inalterados los dos checkpoints fuente.

### Artefactos requeridos

```text
checkpoints/slgnn_v2/curriculum/S03_R1_retention/<lineage>/best-N60-ZG.pt
checkpoints/slgnn_v2/curriculum/S03_R1_retention/<lineage>/best-retention.pt
checkpoints/slgnn_v2/curriculum/S03_R1_retention/<lineage>/best-joint.pt
checkpoints/slgnn_v2/curriculum/S03_R1_retention/<lineage>/promoted-to-S04-N60-G-RP.pt
results/slgnn_v2/curriculum/S03_R1_retention/<lineage>/manifest.json
results/slgnn_v2/curriculum/S03_R1_retention/<lineage>/interpolation_audit.json
results/slgnn_v2/curriculum/S03_R1_retention/<lineage>/head_swap_diagnostics.json
results/slgnn_v2/curriculum/S03_R1_retention/<lineage>/retention_PP2O_PW1.json
results/slgnn_v2/curriculum/S03_R1_retention/<lineage>/N60_ZG_CASE06_metrics.json
results/slgnn_v2/curriculum/S03_R1_retention/<lineage>/RESULTADOS.md
```

**Condición de salida:** si no aparece `promoted-to-S04-N60-G-RP.pt`, no se
continúa a entrenamiento formal de S04.

## Tarea 2 — Implementar la etapa S04 específica para gravedad

### Objetivo

Crear un entrenador de transferencia reproducible que no herede los defectos de
`scripts/slgnn_v2/train.py`.

### Instrucciones

1. Crear:

   ```text
   configs/slgnn_v2/curriculum/S04_N60_G_RP.yaml
   scripts/slgnn_v2/train_S04_N60_G_RP.py
   ```

2. La lineage debe ser completa:

   ```text
   S00-PP2O
   → S01-PW1-RP
   → S02-JOINT-PP2O-PW1
   → S02-R2-preN60
   → S03-N60-ZG-RP
   → S03-R1-RETENTION
   → S04-N60-G-RP
   ```

3. El padre de S04 debe ser exclusivamente
   `promoted-to-S04-N60-G-RP.pt`. Su ruta y SHA-256 se escriben en la
   configuración después de aprobar la Tarea 1.

4. Reconstruir exactamente:

   ```yaml
   hidden: 16
   layers: 2
   use_history: true
   n_types: 1
   n_props: 0
   ```

   La carga debe ser `strict=True`. Se prohíbe la carga parcial.

5. Crear un optimizador AdamW nuevo. Verificar que:

   - el estado inicial del optimizador esté vacío;
   - todos los parámetros sean entrenables;
   - cada parámetro pertenezca exactamente a un grupo;
   - existan grupos separados para procesadores, cabezas PP, cabezas PW y material
     encoder.

6. El nuevo entrenador debe incluir:

   - transferencia desde el checkpoint promovido;
   - replay N60-ZG, PP2O y PW1;
   - L2-SP respecto al padre;
   - muestreo equilibrado por caso y actividad;
   - curriculum de rollout y TBPTT;
   - validación multi-horizonte;
   - tres semillas transferidas y control scratch con el mismo presupuesto;
   - selección multiobjetivo;
   - manifest, hashes y verificación final de checkpoints.

7. Corregir la evaluación energética para gravedad. No basta con vigilar sólo
   energía cinética. Registrar al menos:

   ```text
   E_mech = K_lineal + K_angular + V_g + V_PP + V_PW
   ```

   y comprobar el balance compatible con disipación y pasividad.

8. Añadir pruebas de responsabilidad de canal:

   - estado libre de contactos: aceleración de contacto aproximadamente cero y
     aceleración total aproximadamente igual a `g_vec`;
   - contacto PP sin pared: contribución PW aproximadamente cero;
   - contacto PW sin par PP: contribución PP aproximadamente cero;
   - estado mixto: ambas ramas activas, sin duplicar gravedad.

### Criterios de aceptación

- `--preflight` termina sin crear un optimizador ni alterar el padre;
- `--smoke` ejecuta carga→batch→backward→validación→guardado→recarga;
- la arquitectura y el número de parámetros coinciden con el checkpoint;
- todos los grupos del optimizador tienen parámetros y LR explícito;
- los tests de aislamiento de cabezas pasan;
- el script no consulta CASE07 durante selección;
- el hash del padre es idéntico antes y después del smoke test.

## Tarea 3 — Certificar datos, split, unidades y condiciones físicas

### Objetivo

Convertir la inspección preliminar del benchmark gravitacional en un pre-flight
auditable antes de utilizarlo para gradientes.

### Instrucciones

1. Usar el split oficial:

   - CASE01–CASE05: entrenamiento;
   - CASE06: validación y selección;
   - CASE07: extrapolación final sellada.

2. Auditar los 1501 snapshots de cada caso, no sólo el primer archivo:

   - timesteps enteros contiguos `0..1500`;
   - exactamente 60 filas por snapshot;
   - `Particle_ID` único y conjunto constante por frame;
   - ordenamiento por el valor numérico del timestep, nunca lexicográfico;
   - ausencia de NaN o infinito;
   - continuidad temporal razonable de `q`, `v` y `ω`;
   - densidad constante `4000 kg/m³`.

3. Leer siempre por nombre de columna. Como falta `Diameter`, fijar
   **explícitamente** `default_diameter=0.005` y registrar esta decisión en el
   manifest. No se permite que sea un supuesto silencioso.

4. Ejecutar la auditoría con `cache=False` y comparar después con los caches
   existentes. Un cache previo no debe ocultar un error del CSV ni una configuración
   distinta.

5. Verificar las conversiones:

   | Magnitud | SI | Adimensional |
   |---|---:|---:|
   | diámetro | `0.005 m` | `1.0` |
   | radio | `0.0025 m` | `0.5` |
   | lado de caja | `0.03 m` | `6.0` |
   | paso temporal | `1e-4 s` | `0.01` |
   | masa | `4000·π/6·0.005³ kg` | `1.0` |
   | gravedad `y` | `−9.81 m/s²` | `−0.1962` |

6. Comprobar BoxSDF estática `[0,0,0]→[6,6,6]` en unidades adimensionales.

7. Registrar en el pre-flight los parámetros PP y PW documentados y verificar que
   el modelo usa cabezas separadas para ambos canales.

### Artefacto requerido

```text
results/slgnn_v2/curriculum/S04_N60_G_RP/<campaign>/preflight_data.json
```

Debe contener shapes, IDs, rangos, unidades, escalas, parámetros físicos, hashes
o inventario de archivos y resultado de la comparación cache/CSV.

### Criterios de aceptación

Los siete casos deben pasar todas las comprobaciones. Cualquier timestep faltante,
ID duplicado, unidad ambigua, cache inconsistente o gravedad en otro eje detiene la
etapa.

## Tarea 4 — Registrar baseline y diagnóstico de cabezas

### Objetivo

Medir el punto de partida real de S04 antes de producir el primer gradiente.

### Instrucciones

1. Usar únicamente el checkpoint promovido por S03-R1. El zero-shot ya ejecutado
   con `best-joint` queda como evidencia exploratoria, no como baseline formal de S04.

2. Registrar zero-shot en:

   - N60-G CASE06 a H100, H250, H500 y 1500;
   - N60-ZG CASE06 a los mismos horizontes;
   - PP2O completo: `1x`, `2x`, `4x`;
   - PW1 completo: `10°`, `30°`, `45°`, `60°`, `90°`.

3. Registrar por separado:

   - RMSE de `q`, `v` y `ω`;
   - aceleración lineal y angular one-step;
   - penetración máxima;
   - energía mecánica y pasividad;
   - NaN, divergencia y paso alcanzado;
   - métricas por régimen `free`, `pp_only`, `pw_only` y `mixed`;
   - magnitud de las contribuciones V/R/H de PP y PW.

4. Ejecutar los tests de responsabilidad de cabeza de la Tarea 2 y guardar sus
   resultados junto al baseline.

5. Ejecutar un smoke test completo con una copia de salida separada. El smoke no
   puede sobrescribir checkpoints oficiales.

#### Regla sobre CASE07

CASE07 no debe verse antes de fijar el checkpoint de S04. Para mantener el split
realmente limpio, **no se exige un zero-shot visible de CASE07 como precondición**.
Se debe verificar su existencia, integridad y hash en la Tarea 3, pero sus métricas
se calculan sólo después de seleccionar con CASE06.

Esta regla corrige cualquier interpretación anterior que propusiera mirar CASE07
durante el baseline: hacerlo abriría una vía de ajuste humano sobre la extrapolación.

### Artefactos requeridos

```text
results/slgnn_v2/curriculum/S04_N60_G_RP/<campaign>/preflight_baseline.json
results/slgnn_v2/curriculum/S04_N60_G_RP/<campaign>/head_responsibility.json
results/slgnn_v2/curriculum/S04_N60_G_RP/<campaign>/smoke_test.json
```

### Criterios de aceptación

- todas las suites terminan sin incompatibilidad de arquitectura;
- las métricas son finitas o registran explícitamente el paso de fallo;
- la gravedad aparece sólo una vez como término externo;
- las cabezas inactivas no aportan fuerzas espurias importantes;
- PP2O y PW1 siguen pasando sus gates al inicio de S04;
- CASE07 no figura entre los datos usados para decidir hiperparámetros.

## Tarea 5 — Congelar el protocolo de entrenamiento y promoción de S04

### Objetivo

Evitar cambios oportunistas de presupuesto, datos, horizonte o gates después de
observar resultados.

### Instrucciones

1. Fijar antes de entrenar un reparto inicial de batches. Propuesta:

   - 60% N60-G, tarea principal;
   - 20% N60-ZG, retención multibody sin gravedad;
   - 10% PP2O;
   - 10% PW1.

   CASE01–CASE05 deben muestrearse equilibradamente en ambas tareas N60. Dentro de
   cada dominio se favorecen contactos o aceleración significativa, conservando
   aproximadamente 20% de ventanas de baja actividad.

2. Fijar LR discriminativos que permitan adaptar las cabezas sin reescribir
   rápidamente los procesadores compartidos. Propuesta para warmup:

   - cabezas PP: `5e-5`;
   - cabezas PW: `5e-5`;
   - procesadores V/R/H: `2.5e-5`;
   - material encoder: `5e-6`;
   - AdamW nuevo, `weight_decay=1e-6`, clipping `3.0`;
   - multiplicador de rollout `0.5`.

3. Mantener:

   - warmup one-step de 500–1000 iteraciones;
   - H4→H8→H16→H32→H64;
   - H128 sólo si H64 es estable en todas las semillas transferidas;
   - TBPTT de 4–8;
   - ruido pequeño en `q/v`;
   - validación frecuente en CASE06;
   - checkpoints independientes por horizonte.

4. Incluir pérdidas de:

   - aceleración lineal y angular;
   - posición, velocidad y `ω` en rollout;
   - penetración;
   - pasividad;
   - regularización histórica;
   - replay N60-ZG, PP2O y PW1;
   - L2-SP respecto al padre promovido.

5. Ejecutar como mínimo:

   - tres semillas transferidas;
   - una semilla scratch con arquitectura, datos y presupuesto idénticos;
   - idealmente tres semillas scratch si el presupuesto lo permite.

6. Seleccionar exclusivamente con CASE06 mediante score multiobjetivo. Los gates
   de promoción deben quedar escritos en YAML antes de comenzar:

   - N60-G CASE06 completa H100, H250, H500 y 1500;
   - sin NaN ni divergencia;
   - penetración H500 ≤ `0.75 d_p`;
   - balance energético y pasividad dentro de tolerancias pre-registradas;
   - N60-ZG CASE06 completa 1500 y su score no empeora más de 10% respecto al
     padre promovido;
   - PW mediana ≤ `0.5289`;
   - cada PP ≤ `1.10×` su especialista;
   - PP-4x ≤ aproximadamente `1.790`;
   - tests de responsabilidad de cabeza aprobados.

7. Evaluar CASE07 gravitacional sólo después de copiar el checkpoint seleccionado
   por CASE06 a un nombre inmutable. CASE07 se informa, pero no se usa para volver
   atrás y modificar hiperparámetros.

8. Guardar:

   ```text
   best-N60-G.pt
   best-retention.pt
   best-joint.pt
   final.pt
   promoted-to-next-stage.pt  # sólo si todos los gates pasan
   manifest.json
   métricas por fase y semilla
   curvas de entrenamiento/validación
   métricas CASE06 y CASE07
   PP2O/PW1 completos
   N60-ZG de retención
   comparación transfer vs scratch
   RESULTADOS.md
   GIF/APNG DEM vs SLGNN
   ```

### Criterios de aceptación

El protocolo queda listo cuando configuración, semillas, presupuesto, score y gates
están congelados y el smoke test los recorre sin errores. No se permite reducir
horizontes, semillas, replay o gates para obtener una promoción.

## 4. Definición final de “listo para entrenar S04”

S04 puede comenzar únicamente cuando todas las casillas sean verdaderas:

- [ ] existe `promoted-to-S04-N60-G-RP.pt`;
- [ ] su SHA-256 está registrado y se verificó carga estricta;
- [ ] S03-R1 pasa simultáneamente N60-ZG, PP2O y PW1;
- [ ] existen `configs/slgnn_v2/curriculum/S04_N60_G_RP.yaml` y
  `scripts/slgnn_v2/train_S04_N60_G_RP.py`;
- [ ] el dataset gravitacional completo pasó el pre-flight;
- [ ] el baseline formal fue generado desde el checkpoint promovido;
- [ ] los tests de responsabilidad PP/PW/gravedad pasan;
- [ ] el smoke test extremo a extremo pasa;
- [ ] el protocolo y los gates están congelados;
- [ ] CASE07 permanece fuera de la selección.

## 5. Orden de ejecución recomendado

Los siguientes comandos son la interfaz que deben ofrecer los scripts una vez
implementados; no deben ejecutarse hasta crear y verificar dichos archivos:

```powershell
# 1. Recuperación de S03
python scripts/slgnn_v2/recover_S03_retention.py --config configs/slgnn_v2/curriculum/S03_R1_retention.yaml --preflight
python scripts/slgnn_v2/recover_S03_retention.py --config configs/slgnn_v2/curriculum/S03_R1_retention.yaml --smoke
python scripts/slgnn_v2/recover_S03_retention.py --config configs/slgnn_v2/curriculum/S03_R1_retention.yaml

# 2. Pre-flight de S04, sin entrenamiento
python scripts/slgnn_v2/train_S04_N60_G_RP.py --config configs/slgnn_v2/curriculum/S04_N60_G_RP.yaml --preflight

# 3. Smoke test aislado
python scripts/slgnn_v2/train_S04_N60_G_RP.py --config configs/slgnn_v2/curriculum/S04_N60_G_RP.yaml --smoke

# 4. Entrenamiento formal, sólo tras aprobar los pasos anteriores
python scripts/slgnn_v2/train_S04_N60_G_RP.py --config configs/slgnn_v2/curriculum/S04_N60_G_RP.yaml
```

Después de cada campaña:

```powershell
python -m pytest
git diff --check
```

El resultado debe informarse con honestidad. Si S03-R1 o S04 no pasan sus gates,
se conservan métricas y checkpoints diagnósticos, pero no se crea ni se simula una
promoción.
