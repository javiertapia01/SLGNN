# DATA_NOTES — Dynami-CAL GraphNet DEM dataset

**Fuente:** Sharma, V. & Fink, O. *6 DoF Dynamics: DEM Simulation Dataset for
Learning GNN Surrogate Model.* Zenodo, DOI
[10.5281/zenodo.17589419](https://doi.org/10.5281/zenodo.17589419), CC-BY-4.0.
Generado con el solver DEM **MFiX**. Documentación fuente:
`data/raw/Detailed_Information_on_Data_Structure.pdf`.

Este documento cumple el entregable M0: describe exactamente el contenido de
cada archivo y la representación del movimiento del cilindro rotatorio, para
que `load_dataset.py` (M2) se escriba contra un esquema ya verificado en vez
de uno supuesto.

## Checksums (MD5), verificados contra el registro Zenodo

| Archivo | MD5 | Estado |
|---|---|---|
| `Detailed_Information_on_Data_Structure.pdf` | `d2571c4bf6e5b6581ee74f2e9ae6b3ee` | ✅ verificado |
| `Benchmark_2Spheres_Oblique_Collision.zip` | `0936433f3aae20a6fd36886fbc07e743` | ✅ verificado |
| `Benchmark_1Sphere_Multiple_Wall_Collision.zip` | `01571f270b3d7d60c479811f04805b89` | ✅ verificado |
| `60Spheres_Gravity_Inside_Cuboidal_Enclosure.zip` | `8dc3af6f650788ffca99f4d62f5debe9` | ✅ verificado |
| `60Spheres_Homogeneous_Interaction_Inside_Cuboidal_Enclosure.zip` | `e622e162dcb688c5e1e11ad1fa1e7f6d` | ✅ verificado |
| `Extrapolation_2073Spheres_Gravity_Inside_Rotating_Cylinder.zip` | `db8613ce97ec42e5732766a1faf472c3` | ✅ verificado |

Los seis archivos están descomprimidos en `data/extracted/`.

## Esquema CSV: 3 variantes reales (el PDF documenta solo 1)

⚠️ **Hallazgo no documentado en el PDF fuente, confirmado por inspección directa de los CSV extraídos.** El PDF describe un único esquema de 15 columnas con `Diameter` y `Particle_ID`. En la práctica hay **tres variantes de cabecera distintas** repartidas de forma no uniforme entre archivos, incluso entre carpetas `CASE0X` de un mismo zip:

| Variante | Columnas | `Diameter` | `Particle_ID` | Dónde aparece |
|---|---|---|---|---|
| A (completa, 15 cols) | Diameter, Velocity:0-2, Angular_velocity:0-2, Orientation:0-2, Density, Particle_ID, coordinates:0-2 | sí | sí | `Benchmark_2Spheres_Oblique_Collision`, Homogéneo `CASE01-07` |
| B (sin Diameter, 14 cols) | Velocity:0-2, Angular_velocity:0-2, Orientation:0-2, Density, Particle_ID, coordinates:0-2 | **no** | sí | Homogéneo `CASE08-09`, **todo** `60Spheres_Gravity` (`CASE01-07`), `Benchmark_1Sphere_Multiple_Wall_Collision` |
| C (sin Particle_ID, 14 cols) | Diameter, Velocity:0-2, Angular_velocity:0-2, Orientation:0-2, Density, coordinates:0-2 | sí | **no** | `Extrapolation_2073Spheres` (`CASE08`) |

**Implicación crítica para el loader (M2):** el parser **no puede asumir posiciones de columna fijas ni un número de columnas fijo**. Debe leer siempre por nombre de cabecera (`csv.DictReader` o `pandas` con header), detectar qué columnas existen en cada archivo, y:
- si falta `Diameter`, usar el valor conocido del dataset (0.005 m en los seis archivos, verificado — ver más abajo);
- si falta `Particle_ID`, usar el **orden de fila** como identidad de partícula. Verificado empíricamente en `Extrapolation_2073Spheres/CASE08`: las posiciones y velocidades de las primeras filas evolucionan de forma continua entre `timestep_000` y `timestep_001` (mismo orden de partícula por fila), por lo que asumir `Particle_ID = índice de fila` es seguro **siempre que no se reordenen filas al leer**.

Columnas comunes a las tres variantes:

```
[Diameter?], Velocity:0, Velocity:1, Velocity:2,
Angular_velocity:0, Angular_velocity:1, Angular_velocity:2,
Orientation:0, Orientation:1, Orientation:2,
Density, [Particle_ID?],
coordinates:0, coordinates:1, coordinates:2
```

- **Diameter** [m] = 0.005 constante, **Density** [kg/m³] = 4000 constante, verificado en todos los archivos inspeccionados (homogéneo, gravedad, extrapolación) — no varía entre partículas ni entre casos.
- **Velocity**, **Angular_velocity**: m/s y rad/s — corresponden directamente a $\mathbf v_i$ y $\boldsymbol\omega_i$ de SLGNN-v2.
- **coordinates**: posición del centro $\mathbf q_i$, en metros.
- **Orientation:0-2**: **ignorar**. El PDF advierte explícitamente que el método de cómputo de estos campos en MFiX no está documentado y no corresponde necesariamente a un cuaternión o Euler estándar. No se usan en SLGNN ni en el paper de origen.
- No hay columna de aceleración ni de fuerzas/contactos explícitos: las aceleraciones se derivan por diferencias finitas de velocidad, como ya estaba previsto en el informe de sesión.

## Inventario de archivos y estructura interna

### 1. Benchmark 2 esferas — colisión oblicua (`Benchmark_2Spheres_Oblique_Collision.zip`)
- **Rol:** test unitario del canal partícula–partícula (conservación de momento lineal y angular).
- Sin gravedad, sin paredes, sistema aislado. 100 timesteps, $\Delta t = 10^{-4}$ s.
- Subcarpetas **`1x/`, `2x/`, `4x/`**: variantes de escala (misma condición física, distinta escala espacial/velocidad) — 3 trayectorias independientes, cada una con 100 CSVs de 2 filas.
- Confirmado por extracción: `1x/data_at_timestep_000.csv` a `..._099.csv`, esquema como se describe arriba.

### 2. Benchmark 1 esfera — colisión con pared angulada (`Benchmark_1Sphere_Multiple_Wall_Collision.zip`)
- **Rol:** test unitario del canal partícula–pared vía SDF (restitución, fricción, sin acoplar con partícula–partícula).
- Sin gravedad. Pared plana en $z=0$. Esfera parte de $\mathbf r_0=(0,0,0.005)$ m con velocidad inicial que determina el ángulo de impacto.
- Subcarpetas **`10/, 30/, 45/, 60/, 90/`** (ángulo de impacto en grados). 200 timesteps cada una, $\Delta t = 10^{-4}$ s, 1 fila por CSV.

### 3. 60 esferas homogéneo, sin gravedad (`60Spheres_Homogeneous_Interaction_Inside_Cuboidal_Enclosure.zip`)
- **Rol:** primer entrenamiento real, material homogéneo.
- Caja estática $[0,0.03]^3$ m. Sin gravedad. $d_p=0.005$ m, $\rho_p=4000$ kg/m³, $\mu_s=0.1$, $k_n=1000$ N/m, $k_t/k_n=2/7$, $\eta_t/\eta_n=0.5$, $e=0.9$ (mismos parámetros partícula-partícula y partícula-pared).
- Ruta real tras descomprimir: `60Spheres_Homogeneous_Interaction_Inside_Cuboidal_Enclosure/DATA/CASE01..CASE09/` (con wrapper `DATA/`, a diferencia del resto de archivos).
- **9 casos, no 7 como documenta el PDF.** CASE01–07 coinciden con la descripción del PDF (entrenamiento/validación/extrapolación) y usan el esquema CSV **variante A** (con `Diameter` y `Particle_ID`). **CASE08 y CASE09 son adicionales, no documentados, y usan la variante B** (sin `Diameter`) — ver sección de esquema arriba.
- Cada carpeta CASE tiene **1501 CSVs** (`data_at_timestep_000.csv` a `_1500.csv`), 60 filas cada uno, $\Delta t=10^{-4}$ s ($\delta t$ interno del solver $10^{-7}$ s). El PDF dice "1500 pasos"; en la práctica son 1501 snapshots (estado inicial + 1500 pasos).
  - CASE01–05: entrenamiento (velocidades iniciales aleatorias distintas).
  - CASE06: validación (magnitud de velocidad dentro del rango de entrenamiento, no vista).
  - CASE07: extrapolación (energía cinética inicial ≈3× el caso de entrenamiento más energético).
  - CASE08–09: sin documentar; mismo esquema físico presumiblemente, tratar como entrenamiento/validación adicional no oficial.

### 4. 60 esferas con gravedad, heterogéneo (`60Spheres_Gravity_Inside_Cuboidal_Enclosure.zip`)
- **Rol:** entrenamiento con gravedad y contactos heterogéneos partícula-partícula vs partícula-pared.
- Misma caja $[0,0.03]^3$ m. $g=9.81$ m/s². Parámetros **distintos** por canal:
  - Partícula-partícula: $\mu_s=0.1$, $k_n=10000$ N/m, $\eta_t/\eta_n=0.3$, $e=0.95$.
  - Partícula-pared: $\mu_s=0.3$, $k_n=10000$ N/m, $\eta_t/\eta_n=0.5$, $e=0.9$.
- Ruta real: `60Spheres_Gravity_Inside_Cuboidal_Enclosure/CASE01..CASE07/` — **sin wrapper `DATA/`** (a diferencia del archivo homogéneo). 7 casos, coincide con el PDF en cantidad.
- **Todo este archivo usa la variante B de esquema (sin columna `Diameter`)**, a diferencia de lo que el PDF sugiere (que documenta la variante A como si fuera universal).
- 1501 CSVs × 60 filas por carpeta, $\Delta t=10^{-4}$ s.

### 5. Extrapolación 2073 esferas, cilindro rotatorio (`Extrapolation_2073Spheres_Gravity_Inside_Rotating_Cylinder.zip`)
- **Rol:** test de frontera móvil / extrapolación — **solo inferencia, nunca entrenamiento**.
- Mismos parámetros de contacto que el dataset con gravedad (ítem 4).
- **Geometría:** cilindro vertical, eje alineado con el eje global $z$, radio $r=0.05$ m, altura $h=0.1$ m, centro geométrico en $(0.0, 0.002, 0.05)$ m. Definido por $(x,y,z)\in[-0.05,0.05]\times[-0.048,0.052]\times[0,0.1]$ m. Pared curva rota; tapas superior e inferior fijas.
- **Cinemática de la pared — clave para la SDF dependiente del tiempo de SLGNN:**
  $$\omega(t) = \begin{cases} 4t & 0 \le t < 0.5 \\ 4 - 4t & 0.5 \le t \le 1.5 \\ 0 & t > 1.5 \end{cases} \quad \text{(unidades normalizadas)} \times 2\pi \; [\text{rad/s}]$$
  Es decir: rampa de aceleración angular (0→0.5s), rampa de desaceleración simétrica (0.5→1.5s), luego reposo. **No es velocidad angular constante** — el eje de rotación es fijo ($z$), pero $\Omega_W(t)$ varía en el tiempo, y hay que reconstruir $\omega(t)$ exactamente con esta función para computar $\mathbf v_W(\mathbf x,t) = \mathbf V_W + \boldsymbol\Omega_W(t)\times(\mathbf x-\mathbf c_W)$ del formalismo v2.
- $t$ aquí corresponde a tiempo real en segundos (coincide con los pasos grabados a $\Delta t_{\text{record}}=10^{-3}$ s → trayectoria de 2 s total, consistente con el dominio de la rampa 0–1.5s + cola de reposo).
- **Discrepancia de resolución temporal, importante para el rollout de inferencia:** los datos están grabados a $\Delta t_{\text{record}} = 10^{-3}$ s, pero el modelo (según el paper de origen, y por consistencia debe aplicar también a SLGNN) opera a $\Delta t_{\text{model}} = 10^{-4}$ s. Reproducir la trayectoria completa requiere un **rollout de 20 000 pasos** del modelo para sincronizar con los 2000 snapshots grabados (10 pasos de modelo por snapshot). Esto hay que replicarlo en nuestro integrador de inferencia.
- Ruta real: `Extrapolation_2073Spheres_Gravity_Inside_Rotating_Cylinder/CASE08/` — sin wrapper `DATA/`. **2001 CSVs** (`_000.csv` a `_1999.csv`, sin padding a 4 dígitos: `_1000.csv`, `_1999.csv`, etc.), 2073 filas cada uno.
- Usa la **variante C de esquema** (tiene `Diameter`, no tiene `Particle_ID`). Verificado: el orden de fila es estable entre timesteps consecutivos (identidad de partícula = índice de fila).

## Notas para el loader (M2)

- **El parser debe leer por nombre de columna (header), nunca por índice fijo.** Las 3 variantes de esquema (ver arriba) conviven incluso dentro de un mismo archivo `.zip` (homogéneo: CASE01-07 variante A, CASE08-09 variante B). Un parser posicional produciría columnas desalineadas silenciosamente en CASE08/09 y en todo el archivo de gravedad.
- Si falta `Diameter` en el CSV, usar 0.005 m (constante verificada en los 6 archivos). Si falta `Particle_ID`, usar el índice de fila como ID, preservando el orden de lectura del CSV.
- **Aceleraciones:** no vienen en el CSV. Calcular por diferencias finitas centradas o hacia adelante sobre `Velocity` con el $\Delta t$ correspondiente a cada archivo (no es el mismo para todos: $10^{-4}$s para benchmarks/60-esferas, $10^{-3}$s para el cilindro).
- **Descartar `Orientation:0-2`** en el loader — confirmado no fiable por el propio PDF de origen.
- **Rotación propia ($\omega_i$):** disponible en todas las variantes (`Angular_velocity` presente siempre), por lo que SLGNN-v2-R (con rotación) es viable desde M2 sin esperar un dataset adicional — contrario a lo que se anticipaba como incierto en el informe de sesión.
- Para el cilindro rotatorio, la SDF analítica debe modelarse como un cilindro de radio 0.05 m centrado en $(0.0, 0.002, \cdot)$ con eje $z$, tapas planas en $z=0$ y $z=0.1$, y $\Omega_W(t)$ dado por la función piecewise de arriba (no constante).
- El número de archivos por caso es siempre (pasos documentados + 1) — el CSV `_000` es el estado inicial. Al indexar transiciones $(k, k{+}1)$, el rango válido de $k$ es `[0, n_files - 2]`.
- Nombrado de archivos: padding a 3 dígitos solo hasta 999 (`_000".."_999`); de ahí en adelante sin padding fijo (`_1000`, `_1999`, ...). No asumir un ancho de padding constante al generar rutas — usar el contador entero directamente en el f-string.

## Pendiente

- [x] Verificar checksums de los 6 archivos.
- [x] Descomprimir los 6 archivos a `data/extracted/`.
- [x] Confirmar que la identidad de partícula por orden de fila es estable entre timesteps consecutivos (verificado en `Extrapolation_2073Spheres/CASE08`, que carece de `Particle_ID`).
- [ ] Decidir si CASE08/CASE09 del archivo homogéneo (no documentados) se incluyen en el split de entrenamiento o se excluyen por prudencia.
- [ ] Escribir `load_dataset.py` (M2) contra las 3 variantes de esquema descritas arriba, con un test que falle si aparece una cabecera no reconocida.
