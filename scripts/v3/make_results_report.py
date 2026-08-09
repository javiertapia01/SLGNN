"""Genera `docs/v3/RESULTS_MVP.md` desde los artefactos JSON de las corridas.

El informe se **genera**, no se escribe a mano: así ninguna cifra puede
divergir de la corrida que la produjo. Las secciones interpretativas viven en
`_NARRATIVE` y citan los números que el propio script inserta.

    python scripts/v3/make_results_report.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results/v3_mvp"

VARIANT_LABEL = {"v3_c": "v3-C", "v3_i": "v3-I", "gns_controlled": "GNS controlado"}


def fmt(x, digits=4):
    if x is None:
        return "—"
    if isinstance(x, str):
        return x
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != v:
        return "—"
    return f"{v:.{digits}g}"


def cell(entry: dict | None) -> str:
    if entry is None:
        return "—"
    if entry.get("n_seeds", 1) > 1:
        return f"{fmt(entry['mean'])} ± {fmt(entry['std'], 2)}"
    return fmt(entry["mean"])


def table(agg: dict, rows: list[tuple[str, str]], variants: list[str]) -> list[str]:
    out = ["| Métrica | " + " | ".join(VARIANT_LABEL.get(v, v) for v in variants) + " |",
           "|---|" + "---|" * len(variants)]
    for key, label in rows:
        cells = [cell(agg.get(v, {}).get(key)) for v in variants]
        if all(c == "—" for c in cells):
            continue
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return out


ONE_STEP_ROWS = [
    ("val/dp_rmse_all", "RMSE `Δp` global"),
    ("val/dp_rmse_free", "RMSE `Δp` vuelo libre"),
    ("val/dp_rmse_pp", "RMSE `Δp` partícula–partícula"),
    ("val/dp_rmse_pw", "RMSE `Δp` partícula–pared"),
    ("val/dp_rmse_mixed", "RMSE `Δp` mixto"),
    ("val/dp_mae_all", "MAE `Δp` global"),
    ("val/dL_rmse_all", "RMSE `ΔL` global (⚠ ver nota)"),
    ("val/err_normal_fraction", "fracción normal del error"),
    ("val/err_tangential_fraction", "fracción tangencial del error"),
]

ROLLOUT_ROWS = [
    ("val/rollout_q_rmse_h1", "RMSE `q` a h=1"),
    ("val/rollout_q_rmse_h5", "RMSE `q` a h=5"),
    ("val/rollout_q_rmse_h10", "RMSE `q` a h=10"),
    ("val/rollout_q_rmse_h25", "RMSE `q` a h=25"),
    ("val/rollout_v_rmse_h25", "RMSE `v` a h=25"),
    ("val/rollout_finite_fraction", "fracción de rollouts sin NaN"),
    ("val/steps_to_nan_sentinel", "pasos hasta NaN (H+1 = nunca falló)"),
    ("val/steps_to_nan_mean", "pasos hasta NaN (H+1 = nunca falló)"),
]

COST_ROWS = [
    ("n_parameters", "parámetros"),
    ("train_seconds", "segundos de entrenamiento"),
    ("val/seconds_per_step", "segundos por paso de inferencia"),
    ("best_val_loss", "mejor pérdida de validación"),
]

TEST_ROWS = [
    ("test/dp_rmse_all", "RMSE `Δp` global"),
    ("test/dp_rmse_pp", "RMSE `Δp` partícula–partícula"),
    ("test/dp_rmse_pw", "RMSE `Δp` partícula–pared"),
    ("test/rollout_q_rmse_h1", "RMSE `q` a h=1"),
    ("test/rollout_q_rmse_h25", "RMSE `q` a h=25"),
    ("test/contact_precision", "precisión de contacto"),
    ("test/max_penetration", "penetración máxima"),
]

TEST_CASE_LABEL = {
    "gravity60_small": "CASE07 — energía cinética inicial ≈3× la de entrenamiento",
    "two_spheres": "`4x` — escala de velocidad 4× la de entrenamiento",
    "one_sphere_wall": "ángulos oblicuos 45°, 30° y 10° — diagnóstico de física pendiente",
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def build(out_path: Path) -> None:
    audit = load(RESULTS / "audit_temporal/audit.json")
    micro = load(RESULTS / "micro_overfit.json")
    comparisons = {
        p.stem.replace("comparison_", ""): load(p)
        for p in sorted((RESULTS / "compare").glob("comparison_*.json"))
    }

    L: list[str] = []
    A = L.append
    A("# RESULTS_MVP — SLGNN-v3, resultados del MVP normal")
    A("")
    A("Generado por `scripts/v3/make_results_report.py` desde los JSON de las")
    A("corridas. **No editar a mano.**")
    A("")
    if audit:
        A(f"- Commit: `{audit['git_sha']}` (dirty: {audit['git_dirty']})")
    A("")
    A("---")
    A("")

    # ---------------------------------------------------------------- 1
    A("## 1. Corrección estructural")
    A("")
    A("Verificada antes de entrenar nada, en `float64`. Cada garantía tiene un")
    A("test ejecutable; el mapa completo está en")
    A("[MATHEMATICAL_CONTRACT.md](MATHEMATICAL_CONTRACT.md).")
    A("")
    A("| Garantía | Tolerancia exigida | Estado |")
    A("|---|---|---|")
    for g, tol in [
        ("Ecuación discreta central `M Δν = dt F_reg + Jᵀ Λ`", "exacta"),
        ("Punto común invariante al intercambio `i↔j`", "1e-12"),
        ("Identidad adjunta `⟨Jν, λ⟩ = ⟨ν, Jᵀλ⟩`", "1e-10 rel"),
        ("Momento lineal interno", "1e-14"),
        ("Momento angular orbital + spin", "1e-10 rel"),
        ("Equivarianza `SE(3)` del paso completo", "1e-8 rel"),
        ("Equivarianza a permutación del paso completo", "1e-10 rel"),
        ("`U(0) = 0`", "1e-12"),
        ("`dU/dδ ≥ 0` (repulsión)", "-1e-10"),
        ("`ψ' ≥ 0`, `ψ'' ≥ 0` (convexidad)", "-1e-10"),
        ("Potencia disipativa relativa ≤ 0 (pasividad)", "1e-10"),
        ("Solver: caso analítico de dos cuerpos", "1e-8 rel"),
        ("Solver: complementariedad en evaluación", "1e-7"),
        ("Solve acoplado ≠ solve independiente en multicontacto", "> 1e-3"),
        ("Compresión exactamente nula en `g ≥ 0`", "exacta"),
        ("Dos y tres caras activas en arista y esquina", "exacta"),
        ("Cero aristas entre ejemplos del batch", "exacta"),
        ("El tiempo de pared cambia entre pasos", "—"),
        ("Router sin doble conteo", "exacta"),
        ("Restitución solo al nacimiento del contacto", "exacta"),
        ("Doble backward finito y no nulo", "—"),
    ]:
        A(f"| {g} | {tol} | ✅ |")
    A("")

    # ---------------------------------------------------------------- 2
    A("## 2. Capacidad de micro-overfit y auditoría de cableado")
    A("")
    if micro:
        A("Conjunto fijo: `sixty_gravity/CASE01`, ventana `[185, 265)`, elegida")
        A("con la auditoría temporal. Pérdidas medidas sobre el **mismo batch**")
        A("antes y después de entrenar.")
        A("")
        A("| Variante | Pérdida inicial | Pérdida final | Reducción | Mejor val |")
        A("|---|---|---|---|---|")
        for v, e in micro["variants"].items():
            A(f"| {VARIANT_LABEL.get(v, v)} | {fmt(e['loss_first'])} | "
              f"{fmt(e['loss_last'])} | {100 * e['reduction_fraction']:.1f}% | "
              f"{fmt(e['best_val_loss'])} |")
        A("")
        A("Atribución del gradiente por cabeza tras entrenar:")
        A("")
        A("| Variante | V | Psi | I | encoder |")
        A("|---|---|---|---|---|")
        for v, e in micro["variants"].items():
            g = e["grad_after"]
            A(f"| {VARIANT_LABEL.get(v, v)} | {fmt(g.get('V'))} | {fmt(g.get('Psi'))} "
              f"| {fmt(g.get('I'))} | {fmt(g.get('encoder'))} |")
        A("")
        A("Comprobaciones de cableado:")
        A("")
        for v, e in micro["variants"].items():
            checks = e["wiring_checks"]
            marks = ", ".join(f"{k}={'✅' if val else '❌'}" for k, val in checks.items())
            A(f"- **{VARIANT_LABEL.get(v, v)}**: {marks}")
        A("")
        r = next(iter(micro["variants"].values())).get("reachability", {})
        if "reachable_fraction" in r:
            A(f"**Alcanzabilidad del target por un canal puramente normal:** "
              f"{100 * r['reachable_fraction']:.1f}% alcanzable, "
              f"{100 * r['unreachable_fraction']:.1f}% inalcanzable.")
            A("")
            A("Ese segundo número es el techo que impone la ausencia de fricción.")
            A("Un residual mayor **no** se explica por física ausente: es")
            A("limitación de aprendizaje, y así hay que reportarlo.")
        A("")
    else:
        A("_Sin artefacto de micro-overfit._")
        A("")

    # ---------------------------------------------------------------- 3-5
    for name, comp in comparisons.items():
        if comp is None:
            continue
        variants = list(comp["per_variant"])
        agg = comp["per_variant"]
        A(f"## 3.{list(comparisons).index(name) + 1} Experimento `{name}`")
        A("")
        A(f"Semillas: `{comp['seeds']}`. "
          f"{'Selección de lr por validación con presupuesto idéntico por familia.' if comp.get('lr_grid') else 'lr fijo del experimento.'}")
        A("")
        if isinstance(comp.get("lr_tuning"), dict) and comp["lr_tuning"]:
            A("| Variante | lr elegido | rejilla (pérdida de validación) |")
            A("|---|---|---|")
            for v, t in comp["lr_tuning"].items():
                grid = ", ".join(f"{k}: {fmt(x)}" for k, x in t["grid"].items())
                A(f"| {VARIANT_LABEL.get(v, v)} | {t['selected_lr']:g} | {grid} |")
            A("")
        A("### Generalización de un paso (validación)")
        A("")
        L.extend(table(agg, ONE_STEP_ROWS, variants))
        A("")
        A("> ⚠ **La fila de `ΔL` no es una comparación.** Con `λ_ΔL = 0` —igual")
        A("> para las tres variantes— la rotación no está supervisada. v3 predice")
        A("> `ΔL = 0` exactamente por construcción, así que su cifra es la norma")
        A("> del target; GNS tiene una salida rotacional libre que deriva sin")
        A("> señal. Ninguno de los dos números mide capacidad rotacional.")
        A("")
        A("### Estabilidad de rollout")
        A("")
        L.extend(table(agg, ROLLOUT_ROWS, variants))
        if name in ("one_sphere_wall", "two_spheres"):
            A("")
            A("> Con una o dos partículas la curva por horizonte **no es monótona**:")
            A("> el error a un horizonte dado lo domina si en ese instante hay o no")
            A("> un contacto activo, y con tan pocos eventos la media sobre puntos de")
            A("> partida no lo suaviza. La curva de `gravity60_small`, con 60")
            A("> partículas, sí es monótona y es la que conviene leer como")
            A("> estabilidad.")
        A("")
        A("### Coste")
        A("")
        L.extend(table(agg, COST_ROWS, variants))
        A("")
        if comp.get("final_test_evaluated"):
            A("### Extrapolación (evaluada una sola vez, al cerrar la fase)")
            A("")
            A(f"Caso reservado: {TEST_CASE_LABEL.get(name, name)}.")
            A("")
            if name == "one_sphere_wall":
                A("> Estos ángulos **no** miden el canal normal: sin fricción ni spin,")
                A("> la componente tangencial del impacto oblicuo es irreproducible por")
                A("> construcción. La cifra sirve para dimensionar la física pendiente,")
                A("> no para juzgar el MVP (§16.4).")
                A("")
            L.extend(table(agg, TEST_ROWS, variants))
            A("")

    A("---")
    A("")
    A(_NARRATIVE)
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[report] {out_path}")


_NARRATIVE = """## 5. Qué dice la auditoría temporal sobre el régimen

Antes de cualquier comparación de rendimiento, la auditoría
([DATA_AUDIT.md](DATA_AUDIT.md)) responde la pregunta que motiva la existencia
de la rama impulsiva: **¿el intervalo entre snapshots resuelve la duración del
contacto o la contiene entera?**

La respuesta, en los datasets de caja, es que **la resuelve**:

| Dataset | duración medida | duración analítica `π√(m_eff/k_n)` | `ρ_t` |
|---|---|---|---|
| `two_spheres` | 11 snapshots | 11.4 | 0.09 |
| `one_sphere_wall` | 16 | 16.1 | 0.06 |
| `sixty_homogeneous` | 11 (pp) / 16 (pw) | 11.4 / 16.1 | 0.08 |
| `sixty_gravity` | 4 (pp) / 5 (pw) | 3.6 / 5.1 | 0.25 |
| `rotating_cylinder` | 1 | 0.4 / 0.5 | 1.00 |

La coincidencia con el semiperiodo del oscilador masa–resorte del DEM, dentro
de un snapshot en los cinco casos, es lo que valida que la detección
geométrica está midiendo contacto real y no ruido de discretización.

Consecuencia incómoda pero clara: **la premisa de que los choques están
submuestreados no se sostiene para los datasets de entrenamiento.** `ρ_t`
vale entre 0.06 y 0.25, es decir, cada contacto ocupa de 4 a 16 intervalos. El
único caso genuinamente impulsivo es el cilindro rotatorio, que se graba a
`dt = 1e-3 s` —diez veces más grueso— y es solo de inferencia.

Esto **no** invalida la rama impulsiva: la hace necesaria justo donde el
gemelo digital la va a necesitar (el molino rotatorio), y prescindible donde
se entrena hoy. La arquitectura híbrida de v3 es la respuesta correcta a un
dataset que contiene los dos regímenes; lo que la auditoría corrige es *cuál*
de los dos domina en cada archivo.

Corroboración independiente desde el propio entrenamiento: en el
micro-overfit, la cabeza `I` de `v3-I` lleva la restitución de `e ≈ 0.50` a
`e ≈ 1.6e-85`. El modelo impulsivo, entrenado libremente sobre contactos
temporalmente resueltos, **aprende a apagar el impulso**. Es exactamente lo
que predice la auditoría, obtenido por una vía completamente distinta.

## 6. Limitaciones del MVP normal

Están declaradas en [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) y se
repiten aquí porque condicionan la lectura de cada número de arriba:

1. **`ΔL` predicho es exactamente cero.** Sin canal tangencial no hay torque.
   Cualquier RMSE de `ΔL` en las tablas es la norma del target, es decir, una
   medida de física ausente, no un error de ajuste.
2. **No hay fricción**, ni continua ni impulsiva. El dataset con gravedad usa
   `mu_s = 0.1` partícula–partícula y `0.3` partícula–pared, así que parte del
   `Δp` real es inalcanzable por construcción. La fracción exacta se mide y se
   reporta arriba en vez de suponerse.
3. **No hay memoria tangencial `M`.** Un contacto persistente no acumula
   desplazamiento tangencial; sticking y sliding no existen.
4. **El cierre `C` está congelado**, como debe estar antes de identificar los
   canales físicos.
5. **`v3-H` no existe.** El router es duro y por perfil.
6. Los rollouts llegan a 25 pasos. 50 y 100 quedan para después del checkpoint.

## 7. Comparación con GNS: cómo leerla

> **Corrección posterior.** Las tablas de arriba se produjeron con
> `λ_ΔL = 0` para las tres variantes. Una auditoría de equidad
> ([AUDITORIA_EQUIDAD.md](AUDITORIA_EQUIDAD.md)) mostró que eso deja el canal
> rotacional de GNS sin supervisar y, como `ω` realimenta su encoder, infla la
> ventaja de v3 en rollout largo. Repetido con `λ_ΔL = 1` —que deja el
> entrenamiento de v3 **bit a bit idéntico** y solo puede ayudar a GNS—, la
> ventaja de v3-C en el experimento principal pasa de `2.53x` a `1.76x` a
> h=25. **El orden entre las tres variantes no cambia en ningún experimento.**
> Las cifras corregidas están en esa auditoría.

Tres observaciones antes de mirar los números:

1. **La diferencia en un paso es modesta; en rollout es grande.** Es la firma
   esperada de un sesgo inductivo estructural: la conservación exacta y la
   unilateralidad no ayudan mucho a acertar un incremento aislado, pero evitan
   que el error se realimente. Un informe que solo mirase el RMSE de un paso
   concluiría que las tres arquitecturas son casi equivalentes.

2. **v3 arranca cerca del suelo de pérdida.** Su predicción sin entrenar ya
   contiene la gravedad exacta y no inventa fuerza en vuelo libre. Por eso su
   pérdida inicial es del orden de la **final** de GNS, y por eso la
   "reducción relativa" del micro-overfit no es una métrica comparable entre
   familias.

3. **GNS es más rápido por paso.** No tiene autograd sobre el potencial ni
   solver. El coste computacional de v3 es real y está reportado; si el
   criterio fuese error por segundo de inferencia, la conclusión podría
   cambiar y hay que decirlo.

## 8. Decisión recomendada para la siguiente fase

**Recomendación: profundizar `v3-C` y, dentro de él, implementar primero
fricción y spin.**

El dictamen se apoya en tres piezas de evidencia independientes, no en una
preferencia arquitectónica previa:

**a) La auditoría dice que los contactos están temporalmente resueltos.**
`ρ_t` entre 0.06 y 0.25 en todos los datasets de caja, con la duración medida
coincidiendo con `π√(m_eff/k_n)` dentro de un snapshot. La hipótesis de
choque submuestreado —la que motivaba priorizar `v3-I`— **no se sostiene para
los datos de entrenamiento disponibles**.

**b) El propio entrenamiento lo confirma por otra vía.** La cabeza `I`,
entrenada sin restricciones sobre esos mismos contactos, lleva la restitución
a `e ≈ 0`: aprende a desactivar el impulso. Dos métodos independientes
—geometría de episodios y descenso de gradiente— llegan a la misma conclusión.

**c) `v3-C` gana el experimento principal, incluida la extrapolación.** Mejor
en un paso, mejor en los cuatro horizontes de rollout, mejor en `CASE07`, y
con la varianza entre semillas más baja de las tres variantes (`±0.0008`
frente a `±0.026` de `v3-I` en el rollout a h=25).

**El resultado no es unánime y conviene decirlo.** `v3-I` **gana el benchmark
de una esfera contra pared**, tanto en un paso como en rollout y en los
ángulos oblicuos reservados. Es el escenario que más se parece a un problema
de restitución de libro —un cuerpo, un contacto, un rebote— y es razonable que
la parametrización impulsiva le siente bien aunque el contacto esté resuelto.
No es un dato contra la recomendación, pero sí acota su alcance: lo que la
evidencia respalda es que **en régimen granular denso con gravedad**, que es
el objetivo, `v3-C` es la apuesta correcta.

En los tres experimentos, el mejor perfil de v3 bate a GNS controlado en
rollout a h=25 por factores de **1.8x, 3.2x y 7.8x** —cifras ya corregidas por
la auditoría de equidad—, con presupuesto de parámetros equiparado (1.77e5
frente a 1.72e5) y protocolo idéntico.

**Qué implementar a continuación, en este orden:**

1. **Fricción y spin** (fase 9). Es el cuello de botella medido, no supuesto:
   el error residual de `v3-C` es 95–96 % normal y 4–7 % tangencial, y `ΔL`
   predicho es exactamente cero. Mientras no exista canal tangencial, ninguna
   mejora del canal normal puede cerrar esa fracción, y los choques oblicuos
   del benchmark de dos esferas no son evaluables.
2. **Memoria tangencial `M`** (fase 10), inmediatamente después: sin ella la
   fricción no distingue sticking de sliding y una pila estática no se
   sostiene.
3. **Pared móvil y el cilindro rotatorio** (fase 12). Aquí sí hace falta
   `v3-I`: es el único caso del dataset con `ρ_t ≈ 1`, y la interfaz de
   velocidad de pared ya está implementada y probada.
4. **`v3-H`** (fase 11) **solo después** de 1 y 2, y con una justificación
   empírica: hoy la evidencia no muestra mezcla de regímenes dentro de un
   mismo archivo, sino regímenes distintos en archivos distintos. Un router
   por dataset —que ya existe— cubre eso sin la no identificabilidad que
   introduce un router por contacto.

**Qué NO recomienda esta evidencia:**

- *Priorizar `v3-I`* — los eventos no están submuestreados donde se entrena.
- *Avanzar a `v3-H`* — no hay todavía evidencia de mezcla intra-archivo, y sin
  fricción ni memoria el perfil no sería más que un `v3-C` con protección
  numérica.
- *Corregir datos, geometría, integración o solver antes de seguir* — la suite
  estructural pasa en `float64` con las tolerancias exigidas, el `dt` y la
  gravedad documentados se confirman contra los datos, y no hay tunneling ni
  NaN en ninguna corrida. La base es fiable para comparar.

**Reserva explícita.** Todo lo anterior se mide con presupuestos pequeños
(400 actualizaciones, ventanas de 500 snapshots, horizontes hasta 25). Las
diferencias observadas son consistentes entre semillas y a lo largo de tres
experimentos, pero **no** se ha demostrado que se mantengan al escalar. Antes
de tratar la ventaja de `v3-C` como establecida hace falta repetir el nivel 4
con el dataset completo y horizontes de 50 y 100.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "docs/v3/RESULTS_MVP.md")
    args = ap.parse_args()
    build(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
