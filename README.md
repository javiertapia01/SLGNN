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
tests/         # arnés de pruebas por canal
notebooks/     # exploración
requirements.txt
```

## Entorno

Stack: PyTorch (CPU) + PyTorch Geometric.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Smoke test:

```
python -c "import torch, torch_geometric; print(torch.__version__, torch_geometric.__version__)"
```

## Dataset

*6 DoF Dynamics: DEM Simulation Dataset for Learning GNN Surrogate Model*
(Sharma & Fink, EPFL). Zenodo DOI
[10.5281/zenodo.17589419](https://doi.org/10.5281/zenodo.17589419),
CC-BY-4.0. Ver `data/DATA_NOTES.md` para el esquema una vez descargado.

## Plan de hitos

Ver `Informe_Sesion_SLGNN_Datos_y_Plan.md`, sección 4 (M0–M5).
