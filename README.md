# SLGNN — SDF–Lagrangian Graph Neural Network

Modelo sustituto neuronal para dinámica granular en molinos SAG, informado por
física: grafo dinámico + SDF de pared móvil + lagrangiano aprendido +
disipación de Rayleigh estructurada.

Trabajo final MAT2320. La formulación completa vive en
`Presentación formal del proyecto SLGNN.txt` y `SLGNN_versión2.txt`; el plan
de datos e implementación en `Informe_Sesion_SLGNN_Datos_y_Plan.md`.

## Estructura

```
data/
  raw/         # descargas sin procesar (no versionado)
  extracted/   # datos descomprimidos (no versionado)
  DATA_NOTES.md
src/twin/      # banco de pruebas del esqueleto del gemelo digital (ver su README)
src/slgnn/     # paquete principal
  config.py      # hiperparámetros (unidades adimensionales)
  cutoff.py      # ventana quintic C², compresión softplus, parte negativa suave
  sdf.py         # BoxSDF, RotatingCylinderSDF con cinemática de pared explícita
  graph.py       # lista de vecinos, geometría de pares, punto común de contacto
  nets.py        # encoder material, processor escalar simétrico (solo invariantes)
  model.py       # SLGNN: energías → Euler-Lagrange (autograd) + Rayleigh + canal H
  integrator.py  # Euler semiimplícito + rollout
  losses.py      # L_a, L_alpha, L_roll, L_res, L_pass, L_pen
  data.py        # loader robusto a las 3 variantes de esquema + adimensionalización
tests/         # garantías por construcción (§37): equivariancia SO(3),
               # conservación de momento, disipación, acción-reacción
scripts/       # mini_train.py: overfit del benchmark de 2 esferas
requirements.txt
```

## Entorno

Stack: PyTorch (CPU) + PyTorch Geometric.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Instalar el paquete y correr los tests:

```
pip install -e . --no-deps
pytest -q
```

Mini-entrenamiento de demostración (CPU, ~1-2 min):

```
python scripts/mini_train.py
```

Entrenamiento serio sobre 60 esferas con gravedad, rollout largo y todas las
herramientas del modelo (canales V/R/H, rotación, pared SDF, las 6 pérdidas):

```
python scripts/train.py --config configs/gravity_rollout.yaml --smoke   # verificación rápida
python scripts/train.py --config configs/gravity_rollout.yaml           # corrida completa
```

Los checkpoints (modelo + optimizador + config + sigmas de normalización) se
guardan por fase en `checkpoints/`. La estrategia detrás del curriculum está
en `Informe_Estrategia_Entrenamiento_SLGNN.md`.

## Dataset

**6 DoF Dynamics: DEM Simulation Dataset for Learning GNN Surrogate Model**
(Sharma & Fink, EPFL). Zenodo DOI
[10.5281/zenodo.17589419](https://doi.org/10.5281/zenodo.17589419), CC-BY-4.0.

### Descripción

Dataset de simulaciones DEM (Discrete Element Method) generadas con **MFiX**, que contiene
trayectorias de múltiples partículas esféricas en diversos escenarios de dinámica.
Incluye 6 grados de libertad (posición + orientación) con velocidades lineales y
angulares.

### Contenido

El dataset comprende 6 archivos principales:

1. **Benchmark 2 Spheres** — colisión oblicua sin gravedad (test de conservación de momento)
   - 3 subcasos con escalas distintas (`1x`, `2x`, `4x`)
   - ~100 timesteps por subcaso

2. **Benchmark 1 Sphere** — colisión con pared plana a distintos ángulos
   - 5 ángulos de impacto: 10°, 30°, 45°, 60°, 90°
   - ~200 timesteps por ángulo

3. **60 Spheres Homogeneous** — 60 esferas sin gravedad en caja rígida
   - 9 casos: CASE01-07 (entrenamiento/validación/extrapolación), CASE08-09 (adicionales)
   - 1501 timesteps cada uno

4. **60 Spheres Gravity** — 60 esferas con gravedad en caja rígida
   - Parámetros de contacto distintos para partícula–partícula vs partícula–pared
   - 7 casos (CASE01-07): entrenamiento, validación y extrapolación
   - 1501 timesteps cada uno

5. **Extrapolation 2073 Spheres** — cilindro rotatorio con pared móvil
   - Escenario de frontera dinámica (solo para inferencia)
   - 2073 esferas, 2001 timesteps
   - Pared rotatoria con cinemática time-dependent (rampa aceleración → desaceleración)

6. **Detalles técnicos** — ver `data/DATA_NOTES.md` para:
   - Esquema CSV (3 variantes de cabecera)
   - Parámetros físicos (diámetro, densidad, coeficientes de contacto)
   - Resolución temporal y unidades
   - Estructura de directorios

## Banco de pruebas del esqueleto del gemelo digital (`src/twin/`)

Prototipo del esqueleto de la propuesta de gemelo digital SAG, construido sobre
la arquitectura SLGNN. **No es un gemelo digital**: no hay conexión con planta,
la carga es monodispersa y el sistema es un tambor de laboratorio. Ver
[src/twin/README.md](src/twin/README.md) para la lista completa de lo que este
artefacto no sostiene.

```
trayectorias (MFiX o rollout SLGNN) ──► C_φ ──► espectro E^coll(ω)
                                                     │
                              biblioteca de regímenes ┤
                                                     ▼
                        PBM energético (6 clases) ──► MPC restringido
```

El contrato central es que `C_φ` **no sabe de dónde vienen las trayectorias**:
la fuente es intercambiable sin tocar el operador.

```
python experiments/exp_H_hysteresis.py              # ¿ω ↦ E^coll es una función?
python experiments/exp_E1_value_of_information.py   # ¿el espectro cambia decisiones?
python experiments/exp_E2_spectrum_estimators.py    # necesita checkpoint de SLGNN
```

Parámetros en `configs/twin_toy.yaml`. Resultados en `results/twin/`.

Tres resultados sobre datos reales, con sus cifras en el README del paquete:

- **ω(t) del cilindro:** los datos **refutan** el perfil que implementa
  `slgnn.sdf.dynamical_cylinder_omega`. El tambor invierte el sentido de giro
  entre t = 1.0 y 1.5 s. Ver `data/DATA_NOTES.md` §5.
- **Experimento H:** `ω ↦ E^coll` **depende del camino**, y la dependencia se
  concentra a ω baja (hasta 2.4 décadas de Wasserstein entre ramas), donde la
  rama ascendente arrastra el transitorio de arranque. A ω alta las ramas
  coinciden.
- **Disipación:** `(1−e²)·E_impacto` recupera solo ~1/3 de la pérdida real de
  energía mecánica, de forma sistemática. Se calibra contra la caja estática,
  donde el balance es exacto.

## Plan de hitos

Ver `Informe_Sesion_SLGNN_Datos_y_Plan.md`, sección 4 (M0–M5).
