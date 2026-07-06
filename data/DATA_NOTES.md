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
| `60Spheres_Gravity_Inside_Cuboidal_Enclosure.zip` | `8dc3af6f650788ffca99f4d62f5debe9` | pendiente |
| `60Spheres_Homogeneous_Interaction_Inside_Cuboidal_Enclosure.zip` | `e622e162dcb688c5e1e11ad1fa1e7f6d` | pendiente |
| `Extrapolation_2073Spheres_Gravity_Inside_Rotating_Cylinder.zip` | `db8613ce97ec42e5732766a1faf472c3` | pendiente |

## Esquema CSV común (todas las variantes)

Cada `data_at_timestep_XXX.csv` tiene una fila por partícula, columnas:

```
Diameter,
Velocity:0, Velocity:1, Velocity:2,
Angular_velocity:0, Angular_velocity:1, Angular_velocity:2,
Orientation:0, Orientation:1, Orientation:2,
Density, Particle_ID,
coordinates:0, coordinates:1, coordinates:2
```

- **Diameter** [m], **Density** [kg/m³]: constantes por partícula en cada dataset (esferas idénticas dentro de un mismo archivo).
- **Velocity**, **Angular_velocity**: m/s y rad/s — corresponden directamente a $\mathbf v_i$ y $\boldsymbol\omega_i$ de SLGNN-v2.
- **coordinates**: posición del centro $\mathbf q_i$, en metros.
- **Particle_ID**: entero 1..N, consistente entre timesteps (permite construir trayectorias por ID).
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
- `DATA/CASE01..CASE07/`, 1500 CSVs de 60 filas cada carpeta, $\Delta t=10^{-4}$ s ($\delta t$ interno del solver $10^{-7}$ s).
  - CASE01–05: entrenamiento (velocidades iniciales aleatorias distintas).
  - CASE06: validación (magnitud de velocidad dentro del rango de entrenamiento, no vista).
  - CASE07: extrapolación (energía cinética inicial ≈3× el caso de entrenamiento más energético).

### 4. 60 esferas con gravedad, heterogéneo (`60Spheres_Gravity_Inside_Cuboidal_Enclosure.zip`)
- **Rol:** entrenamiento con gravedad y contactos heterogéneos partícula-partícula vs partícula-pared.
- Misma caja $[0,0.03]^3$ m. $g=9.81$ m/s². Parámetros **distintos** por canal:
  - Partícula-partícula: $\mu_s=0.1$, $k_n=10000$ N/m, $\eta_t/\eta_n=0.3$, $e=0.95$.
  - Partícula-pared: $\mu_s=0.3$, $k_n=10000$ N/m, $\eta_t/\eta_n=0.5$, $e=0.9$.
- Misma estructura `DATA/CASE01..CASE07/`, 1500 CSVs × 60 filas, $\Delta t=10^{-4}$ s.

### 5. Extrapolación 2073 esferas, cilindro rotatorio (`Extrapolation_2073Spheres_Gravity_Inside_Rotating_Cylinder.zip`)
- **Rol:** test de frontera móvil / extrapolación — **solo inferencia, nunca entrenamiento**.
- Mismos parámetros de contacto que el dataset con gravedad (ítem 4).
- **Geometría:** cilindro vertical, eje alineado con el eje global $z$, radio $r=0.05$ m, altura $h=0.1$ m, centro geométrico en $(0.0, 0.002, 0.05)$ m. Definido por $(x,y,z)\in[-0.05,0.05]\times[-0.048,0.052]\times[0,0.1]$ m. Pared curva rota; tapas superior e inferior fijas.
- **Cinemática de la pared — clave para la SDF dependiente del tiempo de SLGNN:**
  $$\omega(t) = \begin{cases} 4t & 0 \le t < 0.5 \\ 4 - 4t & 0.5 \le t \le 1.5 \\ 0 & t > 1.5 \end{cases} \quad \text{(unidades normalizadas)} \times 2\pi \; [\text{rad/s}]$$
  Es decir: rampa de aceleración angular (0→0.5s), rampa de desaceleración simétrica (0.5→1.5s), luego reposo. **No es velocidad angular constante** — el eje de rotación es fijo ($z$), pero $\Omega_W(t)$ varía en el tiempo, y hay que reconstruir $\omega(t)$ exactamente con esta función para computar $\mathbf v_W(\mathbf x,t) = \mathbf V_W + \boldsymbol\Omega_W(t)\times(\mathbf x-\mathbf c_W)$ del formalismo v2.
- $t$ aquí corresponde a tiempo real en segundos (coincide con los 2000 pasos grabados a $\Delta t_{\text{record}}=10^{-3}$ s → trayectoria de 2 s total, consistente con el dominio de la rampa 0–1.5s + cola de reposo).
- **Discrepancia de resolución temporal, importante para el rollout de inferencia:** los datos están grabados a $\Delta t_{\text{record}} = 10^{-3}$ s (2000 pasos), pero el modelo (según el paper de origen, y por consistencia debe aplicar también a SLGNN) opera a $\Delta t_{\text{model}} = 10^{-4}$ s. Reproducir la trayectoria completa requiere un **rollout de 20 000 pasos** del modelo para sincronizar con los 2000 snapshots grabados (10 pasos de modelo por snapshot). Esto hay que replicarlo en nuestro integrador de inferencia.
- `DATA/CASE08/`, 2000 CSVs de 2073 filas.

## Notas para el loader (M2)

- Todas las variantes comparten el mismo esquema CSV → un único parser sirve para todo (`load_dataset.py`), diferenciando solo N (filas por CSV) y presencia/ausencia de gravedad y de pared móvil por metadata del caso, no por parsing distinto.
- **Aceleraciones:** no vienen en el CSV. Calcular por diferencias finitas centradas o hacia adelante sobre `Velocity` con el $\Delta t$ correspondiente a cada archivo (ver tabla de $\Delta t$ arriba — no es el mismo para todos).
- **Descartar `Orientation:0-2`** en el loader — confirmado no fiable por el propio PDF de origen.
- **Rotación propia ($\omega_i$):** si viene disponible en todos los datasets (sí, `Angular_velocity` está en todas las variantes), por lo que SLGNN-v2-R (con rotación) es viable desde M2 sin esperar un dataset adicional — contrario a lo que se anticipaba como incierto en el informe de sesión.
- Para el cilindro rotatorio, la SDF analítica debe modelarse como un cilindro de radio 0.05 m centrado en $(0.0, 0.002, \cdot)$ con eje $z$, tapas planas en $z=0$ y $z=0.1$, y $\Omega_W(t)$ dado por la función piecewise de arriba (no constante).

## Pendiente

- [ ] Verificar checksums de los 3 archivos grandes una vez terminada la descarga.
- [ ] Descomprimir 60-esferas (homogéneo y gravedad) y cilindro rotatorio a `data/extracted/`.
- [ ] Confirmar en código que `Particle_ID` es estable entre timesteps consecutivos dentro de una misma carpeta CASE (asumido, no verificado exhaustivamente).
