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

*6 DoF Dynamics: DEM Simulation Dataset for Learning GNN Surrogate Model*
(Sharma & Fink, EPFL). Zenodo DOI
[10.5281/zenodo.17589419](https://doi.org/10.5281/zenodo.17589419),
CC-BY-4.0. Ver `data/DATA_NOTES.md` para el esquema una vez descargado.

## Plan de hitos

Ver `Informe_Sesion_SLGNN_Datos_y_Plan.md`, sección 4 (M0–M5).
