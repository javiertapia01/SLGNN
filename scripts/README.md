# Interfaces de ejecución

- `slgnn_v2/`: entrenamiento, evaluación, currículo, benchmarks y diagnósticos
  de la implementación original.
- `slgnn_v3/`: auditoría, entrenamiento, evaluación y comparación de v3.
- `twin/`: interfaz gráfica del banco de pruebas del gemelo digital.

Ejemplos:

```powershell
python scripts/slgnn_v2/train.py --config configs/slgnn_v2/gravity_rollout.yaml --smoke
python scripts/slgnn_v3/run_micro_overfit.py
python scripts/twin/gui.py --config configs/twin/twin_toy.yaml
```

Todos los comandos se ejecutan desde la raíz del repositorio.
