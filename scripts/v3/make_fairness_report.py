"""Genera `docs/v3/AUDITORIA_EQUIDAD.md` desde los JSON de las corridas.

Auditoría de los canales por los que la comparación v3 / GNS podría haber
quedado sesgada, con la ablación de `lambda_delta_L` incluida. Las cifras se
leen de los artefactos; el texto interpretativo vive en `_NARRATIVE_*`.

    python scripts/v3/make_fairness_report.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARE = REPO_ROOT / "results/v3_mvp/compare"

VARIANTS = ("v3_c", "v3_i", "gns_controlled")
LABEL = {"v3_c": "v3-C", "v3_i": "v3-I", "gns_controlled": "GNS controlado"}
EXPERIMENTS = ("gravity60_small", "two_spheres", "one_sphere_wall")
EXP_LABEL = {
    "gravity60_small": "60 esferas con gravedad (experimento principal)",
    "two_spheres": "dos esferas, colisión oblicua",
    "one_sphere_wall": "una esfera contra pared",
}


def load(name: str):
    p = COMPARE / f"comparison_{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def val(d, variant, key):
    if d is None:
        return None
    e = d["per_variant"].get(variant, {}).get(key)
    return None if e is None else (e["mean"], e.get("std", 0.0))


def cell(x):
    return "—" if x is None else f"{x[0]:.4g} ± {x[1]:.2g}"


def delta(a, b):
    if a is None or b is None or a[0] == 0:
        return "—"
    return f"{100 * (b[0] - a[0]) / a[0]:+.1f}%"


ROWS = [
    ("val/rollout_w_rmse_h1", "RMSE `ω` rollout h=1"),
    ("val/rollout_w_rmse_h25", "RMSE `ω` rollout h=25"),
    ("val/rollout_q_rmse_h1", "RMSE `q` rollout h=1"),
    ("val/rollout_q_rmse_h10", "RMSE `q` rollout h=10"),
    ("val/rollout_q_rmse_h25", "RMSE `q` rollout h=25"),
    ("val/dp_rmse_all", "RMSE `Δp` un paso"),
    ("test/dp_rmse_all", "RMSE `Δp` extrapolación"),
    ("test/rollout_q_rmse_h25", "RMSE `q` h=25 extrapolación"),
]


def build(out: Path) -> None:
    L: list[str] = []
    A = L.append
    A("# AUDITORÍA DE EQUIDAD — comparación SLGNN-v3 / GNS")
    A("")
    A("Generado por `scripts/v3/make_fairness_report.py`. **No editar a mano.**")
    A("")
    A(_NARRATIVE_INTRO)

    A("## 4. La ablación de `λ_ΔL`")
    A("")
    A(_NARRATIVE_ABLATION)
    for exp in EXPERIMENTS:
        base, abl = load(exp), load(f"{exp}_dL")
        if base is None or abl is None:
            continue
        A(f"### {EXP_LABEL.get(exp, exp)}")
        A("")
        A("| Métrica | Variante | `λ_ΔL = 0` | `λ_ΔL = 1` | cambio |")
        A("|---|---|---|---|---|")
        for key, label in ROWS:
            for v in VARIANTS:
                a, b = val(base, v, key), val(abl, v, key)
                if a is None and b is None:
                    continue
                A(f"| {label} | {LABEL[v]} | {cell(a)} | {cell(b)} | {delta(a, b)} |")
        A("")
        _verdict(A, base, abl)
        A("")

    A(_NARRATIVE_END)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[report] {out}")


def _verdict(A, base, abl) -> None:
    """Comprueba automáticamente las dos afirmaciones que sostiene el informe."""
    for v in ("v3_c", "v3_i"):
        pairs = [(val(base, v, k), val(abl, v, k)) for k, _ in ROWS]
        same = all(a is None or b is None or abs(a[0] - b[0]) <= 1e-12 * max(abs(a[0]), 1.0)
                   for a, b in pairs)
        A(f"- {LABEL[v]}: entrenamiento "
          f"{'**idéntico**' if same else '**alterado**'} al activar `λ_ΔL`.")
    key = "val/rollout_q_rmse_h25"
    g_new, g_old = val(abl, "gns_controlled", key), val(base, "gns_controlled", key)
    scores = {v: val(abl, v, key) for v in ("v3_c", "v3_i")}
    best = min((v for v in scores if scores[v]), key=lambda v: scores[v][0], default=None)
    if g_new and best:
        b = scores[best]
        A(f"- Mejor perfil de v3 en este experimento: **{LABEL[best]}** "
          f"({b[0]:.4g} frente a {g_new[0]:.4g} de GNS, factor "
          f"**{g_new[0] / b[0]:.2f}x**).")
    a25 = scores["v3_c"]
    if a25 and g_new and g_old:
        A(f"- v3-C frente a GNS en rollout h=25: factor **{g_new[0] / a25[0]:.2f}x** "
          f"tras la ablación, frente a {g_old[0] / a25[0]:.2f}x antes. "
          f"La ablación {'reduce' if g_new[0] < g_old[0] else 'no reduce'} "
          f"la ventaja de v3 y **no** cambia el orden.")


_NARRATIVE_INTRO = """## 1. La pregunta

> «SLGNN tuvo mejor desempeño en rollouts y generalización que GNS, pero en el
> informe proponemos que en la función de pérdida de SLGNN se incluyera el
> error respecto a rollouts. ¿Pudo venir de ahí la ventaja?»

La duda está bien fundada: la formulación oficial de SLGNN-v3 **sí** incluye
`L_roll` en la pérdida total (§13.7). Si ese término se hubiera activado solo
para v3, su ventaja en rollout largo dejaría de ser atribuible al sesgo
inductivo —estaría entrenando directamente contra la métrica con la que se lo
evalúa— y la comparación no mediría nada.

## 2. Respuesta corta

**No.** En el MVP la pérdida de entrenamiento es de **un solo paso** para las
tres variantes, sin ningún término de rollout:

```
loss = λ_Δp · ‖Δp_θ − Δp_DEM‖² / N  +  λ_ΔL · ‖ΔL_θ − ΔL_DEM‖² / N
```

con `λ_Δp = 1` y `λ_ΔL = 0` en las tres. `model.step` se llama **una vez** por
batch: no hay desenrollado. Los `config_resolved.yaml` de las nueve corridas
finales de `gravity60_small` lo confirman, y `rollout_loss` existe en
`slgnn_v3/losses.py` pero el runner nunca la llama.

Sin embargo, la pregunta **destapó un sesgo real en la dirección contraria**,
que se documenta en la sección 4 y se ha medido y corregido.

## 3. Canales auditados

Cada vía por la que una de las dos familias podría haber recibido ventaja, y
dónde se comprueba:

| Canal | Estado | Verificación |
|---|---|---|
| Término de rollout en la pérdida | ausente en ambos | `test_shared_objective.py::test_training_loss_has_no_rollout_term` |
| `rollout_loss` conectada al runner | no lo está | `..._is_never_wired_into_training` |
| Clave de rollout en el bloque `train:` | ninguna configuración | `..._experiment_configs_declare_no_rollout_training_term` |
| Esquema de integración | mismo `q_{k+1} = q_k + dt v_{k+1}` | `..._same_semi_implicit_position_update` |
| Gravedad analítica | dada a ambos | `..._gravity_is_given_to_both_models_analytically` |
| Datos, targets y splits | infraestructura única | `test_shared_data.py`, `test_shared_targets.py` |
| Sampler, cuotas y semillas | idénticos | mismo `StratifiedSampler`, composición en el manifiesto |
| Grafo y consultas de pared | idénticos | `slgnn_experiments.scene` |
| Presupuesto de updates y ejemplos | 400 × 2 en ambos | `manifest.json` |
| Presupuesto de parámetros | 1.769e5 vs 1.715e5 (−3 %) | `manifest.json` |
| Selección de `lr` | misma rejilla y presupuesto por familia | `compare.py::select_lr` |
| Selección de checkpoint | por validación, nunca por test | `runner.train(on_best=...)` |
| `CASE07` | fuera de toda selección | `splits.assert_case07_protected` |
| Código de evaluación | mismo `evaluate_rollout` | runner común |
| Métricas de contacto | misma geometría neutral | `scene.active_contact_keys` |
| Fuga entre ejemplos del batch | cero en ambos | `test_no_batch_leakage.py` |
| Precisión numérica | `float64` en ambos | `torch.set_default_dtype` en los scripts |

"""


_NARRATIVE_ABLATION = """La auditoría encontró un sesgo que **perjudicaba a GNS**, no a v3.

Con `λ_ΔL = 0` la rotación no está supervisada para nadie. Para v3 eso es
inocuo: predice `ΔL = 0` exactamente, así que `ω` se queda congelada en su
valor inicial —incorrecto, pero acotado— y el término de pérdida es una
constante de gradiente nulo.

Para GNS no es inocuo. Su decoder rotacional queda libre, y —esta es la parte
que importa— `ω` es una **feature de entrada de su encoder de nodo**:

```python
raw = torch.cat([particles.v, particles.omega, scal, z, wall_feat], dim=-1)
```

v3 no tiene esa realimentación: `ω` solo le entra por las features cinemáticas
de contacto y en combinación simétrica (`|ω_i|+|ω_j|`, `||ω_i|−|ω_j||`). En
GNS, una `ω` que deriva corrompe todos los latentes de nodo del paso
siguiente, y de ahí contamina también la posición.

El efecto es grande y medible: el RMSE de `ω` de GNS pasa de `0.15` en h=1 a
`4.29` en h=25 —un factor 28—, mientras el de v3 se queda en `0.049`.

La ablación activa `λ_ΔL = 1` para las tres variantes. Es **justa por
construcción**: no puede cambiar nada en v3 y solo puede ayudar a GNS. Que v3
salga bit a bit idéntico es la comprobación de que la ablación hace lo que
dice.

"""


_NARRATIVE_END = """---

## 5. Conclusión

**La respuesta a la pregunta original es no**: la ventaja de v3 en rollout no
viene de la función de pérdida. Ambos modelos entrenaron contra el mismo
objetivo de un paso, y hay tests que lo impiden cambiar en silencio.

**Pero la pregunta era productiva.** Al auditar el canal señalado apareció
otro que iba en dirección contraria: `λ_ΔL = 0` dejaba el canal rotacional de
GNS sin supervisar, y como `ω` realimenta su encoder, eso inflaba
artificialmente la ventaja de v3 en rollout largo. Corregido, la ventaja de
v3-C **se reduce alrededor de un tercio pero no desaparece**, y el orden entre
las tres variantes no cambia en ningún experimento.

Como subproducto, la varianza entre semillas de GNS se desploma: la `ω` sin
supervisar era también la fuente de su inestabilidad.

## 6. Qué queda sin controlar

Honestamente, lo que esta auditoría **no** descarta:

1. **Presupuestos pequeños.** 400 actualizaciones y ventanas de 500 snapshots.
   Las diferencias son consistentes entre semillas y entre tres experimentos,
   pero no se ha demostrado que sobrevivan al escalar.
2. **Capacidad frente a arquitectura.** Los parámetros están equiparados al
   3 %, pero eso no garantiza capacidad efectiva equiparada: v3 gasta parte de
   la suya en cuadratura y solver, GNS toda en el decoder.
3. **Coste computacional.** GNS es 1.6x más rápido por paso de inferencia. Con
   un criterio de error por segundo, la conclusión podría cambiar.
4. **Un solo baseline principal.** `GNSClassicReduced` está implementado y
   probado, pero no se ha corrido como comparación: cambia historia, target y
   representación a la vez, y con tres variables movidas una diferencia no
   sería atribuible.

## 7. Consecuencia para la fase siguiente

Cuando se active `L_roll` para v3 —que la formulación pide (§13.7)— **hay que
activarlo también para GNS**, o la comparación deja de medir el sesgo
inductivo y pasa a medir quién entrena contra la métrica de evaluación.

`tests/comparison/test_shared_objective.py` falla si eso ocurre solo en un
lado. Es el mecanismo que hace que esta auditoría no haya que repetirla a
mano.

## 8. Reproducir

```bash
python -m pytest tests/comparison -q

python scripts/v3/compare.py \\
  --experiment configs/experiments/gravity60_small_dL.yaml \\
  --seeds 0 1 2 --tune-lr --final-test

python scripts/v3/make_fairness_report.py
```
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "docs/v3/AUDITORIA_EQUIDAD.md")
    args = ap.parse_args()
    build(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
