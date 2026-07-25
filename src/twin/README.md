# `twin` — banco de pruebas del esqueleto del gemelo digital SAG

El nombre importa. Esto es un **banco de pruebas del esqueleto**, no un gemelo
digital *toy*, para que el nombre no prometa lo que el artefacto no sostiene.

## Lo que este banco de pruebas NO promete

- **No valida SLGNN** como surrogate de DEM industrial.
- El PBM tiene parámetros **plausibles, no calibrados**: sus PSD no son
  predicciones.
- La carga es **monodispersa y sin medios de molienda**. La retroalimentación
  PSD → espectro es un **andamio declarado** (`coupling.psd_feedback_factor`,
  que emite `AndamioWarning`), no un resultado.
- El sistema es un **tambor de laboratorio**, no un molino SAG. No hay
  escalamiento a unidades industriales, y el objetivo económico está definido en
  unidades del tambor: masa bajo tamaño objetivo por unidad de energía.
- No hay pulpa, desgaste, parrilla, *pebbles* ni circuito.
- **No hay conexión con planta y por tanto no es un gemelo digital** en el
  sentido operacional que el informe de propuesta define con cuidado.
- Las cifras de aceleración no son transferibles.

## Estatus de cada arco del esqueleto

| Arco | Estado |
|---|---|
| `omega → espectro` | **real, con datos** |
| `espectro → tasas de rotura → PSD` | real como modelo, parámetros plausibles no calibrados |
| `PSD → espectro` | **andamio sintético declarado**, no verificable con estos datos |
| llenado y carga de bolas → espectro | no verificable |
| estimación de estado | andamio |
| planta real | ausente |

Consecuencia directa para el control: **la única acción con respaldo
microdinámico es `omega`.** `F_feed` entra únicamente por el balance
macroscópico. `ControlResult.meta` arrastra esa asimetría (`omega_has_micro_support`
vs `feed_has_micro_support`) hasta el reporte.

## Módulos

| Módulo | Rol |
|---|---|
| `units.py` | frontera de unidades: `C_phi` recibe adimensional y entrega SI |
| `events.py` | detector de colisiones sobre trayectorias (p–p y p–pared) |
| `coarse.py` | `C_phi`: espectros, energías, disipación derivada, cierre |
| `cylinder.py` | cinemática **verificada** del cilindro rotatorio |
| `library.py` | biblioteca de regímenes `E^coll(omega)` + histéresis |
| `macro.py` | PBM energético de 6 clases |
| `coupling.py` | operador `S`, políticas de refresco, andamio PSD |
| `control.py` | MPC restringido + baselines |
| `confidence.py` | score OOD e incertidumbre desde la histéresis |
| `harness.py` | métricas (Wasserstein), tablas, reportes |
| `pipeline.py` | carga de datos y armado, compartido por los experimentos |

## Cómo correrlo

```bash
python experiments/exp_H_hysteresis.py              # T1: ¿omega ↦ E^coll es función?
python experiments/exp_E1_value_of_information.py   # T2: ¿el espectro cambia decisiones?
python experiments/exp_E2_spectrum_estimators.py    # T3: necesita checkpoint de SLGNN
pytest tests/test_events.py tests/test_coarse.py tests/test_macro.py
```

Todo parámetro vive en `configs/twin_toy.yaml`.

## Tres desviaciones respecto de la especificación, todas medidas

Están documentadas en detalle en el docstring de cada módulo; en resumen:

1. **Detección por onset de contacto, no `k_pre` fijo.** La especificación
   anticipa *aliasing*; en este dataset ocurre lo contrario — el contacto abarca
   ~11 snapshots. Con `k_pre = 1` la velocidad normal ya viene frenada por el
   resorte de contacto (−0.375 frente a −1.450 reales en `2Spheres/1x`: factor
   15 en energía). El detector camina hacia atrás hasta el último snapshot de
   vuelo libre, y recupera la velocidad exacta en los 8 subcasos de los dos
   micro-benchmarks.

2. **La disipación necesita un factor de calibración.** `(1−e²)·E_impacto`
   recupera solo ~1/3 de la pérdida real de energía mecánica, de forma
   sistemática (κ ≈ 3.1–3.3, dispersión entre ventanas < 8 %). Se calibra contra
   la caja estática, donde `p_in ≡ 0` es exacto. `p_diss` es **derivado y
   calibrado**, nunca medido.

3. **`E_normal` no es monótona en la etiqueta de ángulo del benchmark de
   pared.** Las etiquetas 10–60 son el ángulo respecto de la normal
   (`v0 = 2·(sin θ, 0, −cos θ)`), de modo que la energía normal *decrece* con la
   etiqueta; la etiqueta 90 es un impacto frontal y por eso sí es el máximo.

## Hallazgo que afecta a `src/slgnn/`

`slgnn.sdf.dynamical_cylinder_omega` implementa un perfil de `omega(t)`
**refutado por los datos**. Ver `twin/cylinder.py` y el paso 0 de
`exp_H_hysteresis.py`: el tambor **invierte el sentido de giro** entre t = 1.0 y
t = 1.5 s. Dos señales independientes lo confirman (signo de la rotación del
lecho y el segundo pico del spin propio). El perfil triangular predice reposo en
ese tramo.

Consecuencia: `v_W(x, t)` —el canal que la v2 introdujo precisamente para este
caso— queda equivocado para t > 1 s, lo que afecta a cualquier rollout de
extrapolación. No se corrigió en `src/slgnn/` para no cambiar el comportamiento
del modelo sin decisión explícita del equipo.
