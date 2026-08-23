# Sistema de organización del repositorio

Este documento define dónde vive cada tipo de contenido. La regla central es
separar **código importable**, **interfaces de ejecución**, **configuración**,
**documentación**, **datos** y **artefactos generados**. Dentro de cada capa se
repite, cuando aplica, la misma clasificación por dominio: `slgnn_v2`,
`slgnn_v3`, `twin`, `benchmarks` y `comparison`.

## Mapa principal

| Ruta | Responsabilidad | Se versiona |
|---|---|---|
| `src/` | Paquetes Python importables y reutilizables | Sí |
| `tests/` | Pruebas agrupadas por el paquete o contrato que validan | Sí |
| `scripts/` | Entradas CLI para entrenar, evaluar, diagnosticar o generar informes | Sí |
| `configs/` | YAML declarativos; nunca código ni resultados | Sí |
| `experiments/` | Experimentos científicos de alto nivel | Sí |
| `docs/` | Formulación, decisiones, planes e informes | Sí |
| `data/raw/` | Descargas originales e inmutables | No |
| `data/extracted/` | Datos descomprimidos o preparados desde `raw/` | No |
| `checkpoints/` | Estados de modelo y optimizador | No, salvo su índice |
| `results/` | Métricas, figuras, manifiestos e informes generados | Solo resúmenes explícitos |
| `notebooks/` | Exploración interactiva; no debe contener lógica de producción | Sí |

La raíz queda reservada a metadatos necesarios para comprender, instalar y
ejecutar el proyecto: `README.md`, `pyproject.toml`, `requirements.txt`,
`.gitignore` y herramientas ocultas del entorno.

## Clasificación por dominio

```text
configs/                    scripts/                    tests/
├── benchmarks/             ├── slgnn_v2/               ├── slgnn_v2/
├── experiments/            ├── slgnn_v3/               ├── slgnn_v3/
├── gns/                    └── twin/                    ├── twin/
├── slgnn_v2/                                           ├── baselines/
│   └── curriculum/                                     └── comparison/
├── slgnn_v3/
└── twin/
```

- `slgnn_v2` contiene la implementación original y su currículo S02–S04.
- `slgnn_v3` contiene la arquitectura energético-disipativa/impulsiva.
- `twin` contiene el banco de pruebas del gemelo digital.
- `benchmarks` contiene escenarios pequeños usados para validar o preentrenar
  SLGNN-v2.
- `comparison`, `experiments` y `gns` contienen la infraestructura neutral de
  comparación entre arquitecturas y sus baselines.

## Artefactos generados

Los checkpoints y resultados reflejan el dominio que los produce:

```text
checkpoints/slgnn_v2/
├── benchmarks/
├── curriculum/
└── gravity_rollout/

results/
├── slgnn_v2/
│   ├── benchmarks/
│   ├── curriculum/
│   ├── diagnostics/
│   └── evaluations/
├── slgnn_v3/
└── twin/
```

Una corrida nueva debe escribir dentro de un directorio propio e incluir, si
el flujo lo permite, configuración resuelta, semilla, checkpoint de origen,
estado de Git, métricas y fecha. No se escriben logs directamente en la raíz de
`results/`.

## Convenciones para contenido nuevo

1. El código reutilizable va en `src/`; un archivo de `scripts/` debe ser una
   interfaz delgada o una orquestación, no una segunda implementación.
2. Cada script recibe rutas relativas a la raíz del repositorio y debe calcular
   `REPO_ROOT` desde `__file__`, sin depender del directorio de trabajo actual.
3. Una configuración nueva se coloca junto a las de su mismo dominio. Las
   etapas del currículo conservan el prefijo canónico `SNN_` para mantener el
   orden temporal.
4. Los documentos activos usan enlaces a las rutas actuales. Copias históricas
   o duplicadas van en `docs/archive/` y no se consideran instrucciones vigentes.
5. `data/raw/` no se modifica. Toda transformación reproducible escribe en
   `data/extracted/`, una caché o un directorio nuevo documentado en
   `data/DATA_NOTES.md`.
6. No se versionan `.pt`, cachés, logs ni resultados voluminosos. Solo se
   reincluyen en `.gitignore` los resúmenes pequeños que respaldan un informe.
7. Antes de cerrar una reorganización o una nueva etapa se ejecuta
   `python -m pytest -q` y se comprueban las rutas mencionadas en README y YAML.

## Migración aplicada el 10 de agosto de 2026

La reorganización preservó todos los archivos y agrupó:

- los documentos sueltos de la raíz bajo `docs/slgnn_v2/`, `docs/slgnn_v3/`
  y `docs/twin/`;
- scripts, configuraciones y pruebas bajo dominios paralelos;
- checkpoints y resultados de SLGNN-v2 bajo `slgnn_v2/`;
- `results/v3_mvp/` bajo el nombre estable `results/slgnn_v3/`;
- los logs sueltos de S03-R1 bajo el directorio `logs/` de esa etapa.

Los dos informes S04 encontrados eran idénticos byte a byte. La copia con
nombre canónico quedó en `docs/slgnn_v2/training/`; la segunda se conservó en
`docs/archive/duplicates/` para no eliminar información del usuario.
