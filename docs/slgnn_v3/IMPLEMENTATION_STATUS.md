# IMPLEMENTATION_STATUS — SLGNN-v3

Estado de cada componente de la formulación oficial. Tres valores posibles:

- **implementado** — existe, está probado y está activo;
- **contrato** — la interfaz existe y **falla explícitamente** si se usa;
- **ausente** — ni siquiera hay interfaz.

Nada está marcado como implementado sin un test que lo respalde. La regla que
gobierna este documento: *un canal que no existe se declara, no se simula*.

---

## Resumen

| Componente | Estado | Notas |
|---|---|---|
| Estado `(q, v, omega)` y `M` conocida | implementado | `state.py` |
| SDF diferenciable, multi-superficie, `v_W` | implementado | `surfaces.py` |
| Grafo no ordenado, batch-safe, CCD | implementado | `graph.py` |
| Cinemática, punto común, `J`/`J^T` | implementado | `contact_kinematics.py`, `contact_operator.py` |
| Gravedad analítica, contada una vez | implementado | `integrator.py` |
| Potencial normal repulsivo `V` | implementado | `potential.py` |
| Disipación normal convexa `Psi` | implementado | `dissipation.py` |
| Disipación tangencial `Psi_tau` y spin por `J^T` | implementado | opcional en `v3_c_tangential.yaml` |
| Cabeza impulsiva `I` (`e`, `kappa`) | implementado | `impact.py` |
| Solver normal acoplado | implementado | `solver.py` |
| Router fijo `v3-C` / `v3-I` | implementado | `router.py` |
| Integración fuerza–impulso | implementado | `integrator.py`, `model.py` |
| Pérdidas `Delta p`, `Delta L`, rollout | implementado | `losses.py` |
| Diagnósticos por cabeza y régimen | implementado | `diagnostics.py` |
| Tests estructurales en `float64` | implementado | 130 tests en `tests/slgnn_v3` |
| GNS controlado | implementado | `gns_baseline/model.py` |
| GNS clásico reducido | implementado | `gns_baseline/classic.py` |
| **Fricción impulsiva `mu`** | **contrato** | `ImpactHead` devuelve `mu = None` |
| **`Psi` rotacional directa (rolling/twist)** | **contrato** | falla explícitamente si se activa |
| **Memoria tangencial `M`, estado `xi`** | **contrato** | `memory.py`, todo `NotImplementedError` |
| **Cierre residual `C`** | **contrato** | `closure.py`, congelado |
| **Perfil híbrido `v3-H`** | **contrato** | `build_router` falla nombrando lo que falta |
| Pared móvil en entrenamiento | contrato | interfaz y tests existen; no se ha entrenado con ella |
| Rollouts largos (50, 100) | ausente | el MVP llega a 25 |

---

## Consecuencias medibles de lo que falta

### `Delta L` era exactamente cero en el MVP normal

Con `configs/slgnn_v3/mvp_c.yaml` y `mvp_i.yaml`, que preservan el MVP normal
histórico, no hay torque y el modelo predice `Delta L = 0` con igualdad exacta.
Hay un test que lo afirma
(`test_symmetry.py::test_mvp_produces_exactly_zero_angular_momentum`). Por eso
`lambda_delta_L = 0` en las configuraciones: incluirlo en la pérdida no
enseñaría nada, solo añadiría una constante.

La configuración `v3_c_tangential.yaml` activa `Psi_tau`: aplica la fuerza
tangencial mediante `J^T`, produce `Delta L != 0` y recibe gradiente desde la
pérdida rotacional. El primer gate de tres semillas reduce 9.7 % el RMSE de
`Delta L` y 33.2 % el RMSE de `omega` a h=25 sin degradar `Delta p`; ver
[RESULTS_FRICTION_SPIN.md](RESULTS_FRICTION_SPIN.md). Aún no representa
sticking/sliding ni memoria.

Cualquier RMSE de `Delta L` que aparezca en un informe es **la norma del
target**, es decir, una medida de física ausente. No es un error de ajuste.

### La componente tangencial de `Delta p` es parcialmente inalcanzable

Un canal puramente normal solo puede aplicar fuerza en el span de las normales
de contacto activas. Medido sobre la ventana de micro-overfit
(`sixty_gravity/CASE01[185:265]`), **el 95.4 % de `Delta p` (descontada la
gravedad) sí es alcanzable** y el 4.6 % no lo es.

Ese número importa: significa que el residual de v3-C en este dataset **no**
está dominado por la fricción ausente. Un residual mayor que el 4.6 % es
limitación de aprendizaje, no de representabilidad.

### `beta > 0` puede hacer `D_impact < 0`

La estabilización de penetración de Baumgarte inyecta energía a propósito para
separar contactos ya solapados. `D_impact >= 0` está garantizado solo con
`beta = 0`. La magnitud del término se expone como
`impact.stabilization_term_max` para que el aporte sea atribuible.

---

## Desviaciones respecto de la estructura pedida en §2.3

| Desviación | Razón |
|---|---|
| `src/slgnn_v3/smoothing.py` (no listado) | Separa el suavizado unilateral (`p_eps`) del corte de grafo (`quintic_window`). §16.2 de la formulación exige explícitamente no confundirlos; tenerlos en un módulo propio lo hace visible. |
| `src/slgnn_experiments/scene.py` (no listado) | Punto único de construcción de escena compartida. Necesario para que v3 y GNS reciban literalmente los mismos objetos. Ver D-016. |
| `scripts/slgnn_v3/_common.py`, `run_all_benchmarks.py` | Utilidades de CLI; no cambian ningún contrato. |
| `slgnn_experiments` importa `slgnn_v3.{state,surfaces,graph}` | Esos módulos son contenedores y geometría, sin física aprendida. `gns_baseline` **no** importa `slgnn_v3`, y `tests/comparison/test_shared_data.py` lo verifica sobre el código fuente. |

---

## Legacy

`src/slgnn/` **no se ha modificado**. Los 85 tests legacy pasaban antes de
empezar y siguen pasando. `tests/comparison/test_shared_data.py::
test_legacy_package_still_imports` lo comprueba en cada corrida.

Una discrepancia conocida del legacy queda documentada pero **no corregida**,
por estar fuera del alcance: `slgnn.sdf.dynamical_cylinder_omega` implementa el
perfil triangular, que los datos refutan
(`data/DATA_NOTES.md` §5). v3 usa la fórmula literal en
`surfaces.dynamical_cylinder_omega_literal`, con su propio test.

---

## Orden de las fases siguientes

Tal como fija §14.6 de la formulación oficial, y solo después del checkpoint
científico del MVP:

1. validar `Psi_tau` y spin oblicuo en entrenamiento (implementación lista);
2. fricción impulsiva y cono de Coulomb;
3. memoria tangencial `M`, sticking/sliding y lifecycle completo;
4. router híbrido `v3-H`;
5. pared móvil y extrapolación al cilindro rotatorio;
6. rollouts largos (50, 100);
7. cierre `C`, solo si quedan residuales reproducibles.
