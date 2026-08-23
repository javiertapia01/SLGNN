# DECISIONS — SLGNN-v3

Registro de decisiones técnicas y científicas tomadas durante la implementación
del MVP. Cada entrada indica la alternativa elegida, la razón y qué la
invalidaría. Las decisiones marcadas **[INVARIANTE]** no se cambian sin volver a
correr toda la suite estructural.

---

## D-000 — Punto de partida de git

`HEAD` al iniciar: `d6e5f4b47094163084da0508bdb9183da940bd41`
(`Add twin/: skeleton test-bench for the SAG digital twin proposal`).

El commit de referencia de las instrucciones (`5f9cb96`, *Expand Dataset section
in README*) es el **padre** del actual: `d6e5f4b` añade `src/twin/` (banco de
pruebas del gemelo digital) sin tocar `src/slgnn/`. La diferencia no afecta a
ningún contrato usado por v3.

El árbol de trabajo estaba **sucio** al empezar: `README.md` y
`src/twin/README.md` modificados y 22 archivos sin seguimiento (informes,
configs y scripts de la fase A / S02 / S03, `src/slgnn/diagnostics.py`,
`tests/test_diagnostics.py`, `tests/test_recover_s03_retention.py`). Ese trabajo
**no se toca**. Se creó la rama local `feature/slgnn-v3-mvp` desde `d6e5f4b`,
arrastrando los cambios no confirmados sin modificarlos.

---

## D-001 — [INVARIANTE] Convención de signo de la SDF

`phi > 0` dentro, `phi = 0` en la pared, `grad(phi)` hacia el interior,
`g_iW = phi(q_i) - R_i`. Idéntica al legacy y a la §3.1 de la formulación
oficial. Una partícula que se acerca a una pared fija tiene `u_n < 0`.

---

## D-002 — [INVARIANTE] Convención de velocidad relativa

- Par `{i, j}` con `i < j`: `n` apunta de `i` a `j`, y
  `u = (v_j + w_j x r_j) - (v_i + w_i x r_i)`.
- Pared: `u = (v_i + w_i x r_iW) - v_W`, con `n` la normal entrante.
- En ambos casos `u_n < 0` significa aproximación.
- Un vector de contacto `lambda` actúa `+lambda` sobre `j` y `-lambda` sobre `i`
  (par), o `+lambda` sobre la partícula (pared).

Esta convención vive en un solo lugar (`slgnn_v3.contact_kinematics`) y todos
los tests la verifican contra construcción directa.

---

## D-003 — [INVARIANTE] Punto común partícula–partícula

`x_c = ((q_i + R_i n) + (q_j - R_j n)) / 2` — punto medio entre las dos
superficies no deformadas. **No** el reparto proporcional a radios del legacy
(`l_i = d R_i/(R_i+R_j)`), que solo coincide con el punto medio cuando
`g = 0`. Con radios desiguales y `g != 0` el reparto proporcional pone el punto
fuera del punto físico y rompe la simetría bajo intercambio `i <-> j`.

Verificado por `tests/slgnn_v3/test_contact_kinematics.py::test_common_point_swap`
a `atol = 1e-12` en float64.

---

## D-004 — [INVARIANTE] Compresión unilateral C² exactamente nula

Se descarta `softplus(-g, beta)` del legacy: vale `log(2)/beta > 0` en `g = 0`
y nunca es exactamente cero en separación, lo que produce una fuerza elástica
espuria en vuelo libre y hace que la energía dependa del corte del grafo.

Se implementa `p_eps` (§4.5 de las instrucciones, §4.4 de la formulación):

```
p_eps(x) = 0                             si x <= 0
         = eps (6 z^3 - 8 z^4 + 3 z^5)   si 0 < x < eps,  z = x/eps
         = x                             si x >= eps
```

con `delta = p_eps(-g)`. Es C² en `0` y en `eps` (probado numéricamente),
no negativa y **exactamente** `0.0` para `x <= 0`.

---

## D-005 — [INVARIANTE] Multi-superficie para la caja

La caja se representa como **seis superficies planas independientes** con
`surface_id` estable (`0..5` = `-x, +x, -y, +y, -z, +z`), no como
`min` sobre las seis caras. Un contacto partícula–pared se emite para *toda*
cara dentro de la banda `g <= g_off`, de modo que una partícula en una arista
o esquina produce 2 o 3 contactos simultáneos. La clave estable de contacto es
`(batch_id, particle_id, surface_id)`.

La SDF global (`min` sobre caras) se conserva solo como consulta diagnóstica.

---

## D-006 — Gravedad como fuerza analítica, contada una sola vez

`F_ext = m_i g` se suma explícitamente y se registra por separado en
`diagnostics.regular.force_external`. `V_theta` **no** contiene término
gravitatorio. Elegido sobre el potencial `V_g` porque elimina la ambigüedad de
doble conteo y permite auditar la contribución externa sin derivar.

---

## D-007 — Cuadratura del potencial

`U(delta) = int_0^delta f_n(s) ds` se evalúa con Gauss–Legendre de 8 nodos
sobre `[0, delta]`, con nodos y pesos registrados como buffers no entrenables.
`U(0) = 0` por construcción (el intervalo es de medida nula) y
`dU/d(delta) = f_n(delta) >= 0` porque `f_n = f_0 a(delta) softplus(k)` con
`a(delta) = delta/L_0 >= 0`.

El prior Hertziano `a(delta) = (delta/L_0)^{3/2}` queda como opción de
configuración (`potential.exponent`), desactivada por defecto.

---

## D-008 — Familia disipativa convexa del MVP

`psi(s) = c1 s^2 / 2 + c2 s^3 / 3` con `c1, c2 >= 0` producidos por `softplus`.
`c1, c2` dependen de material, gap y compresión — **no** de `s` — para que
`psi' = c1 s + c2 s^2 >= 0` y `psi'' = c1 + 2 c2 s >= 0` se cumplan por
construcción. La construcción por doble integral de la §6.4 de la formulación
oficial queda para la extensión con dependencia no lineal completa en
velocidad.

---

## D-009 — Backend del solver impulsivo

Tres capas:

1. **Escalar exacto**: componentes de un solo contacto se resuelven en forma
   cerrada, `lambda = max(0, -b / (A_nn + kappa))`.
2. **FISTA proyectado desenrollado** para componentes de más de un contacto,
   con `H = A_n + diag(kappa)` materializada densa **por componente conexa**
   (nunca `J` global). Paso `1/L` con `L = ||H||_2` estimada por iteración de
   potencia; la estimación de `L` se hace bajo `no_grad` (documentado: es un
   parámetro de paso, no parte del modelo).
3. **Diagnósticos obligatorios** por solve: residuo primal, `min(lambda)`,
   `min(r_n)`, producto de complementariedad, iteraciones, condicionamiento y
   tamaño de componente.

Entrenamiento usa iteraciones fijas desenrolladas; evaluación permite parada
por tolerancia. Los tests corren en float64.

---

## D-010 — Restitución solo al nacimiento

`b_n = iota e min(u_n^-, 0) + beta min(g, 0)/dt`. El indicador `iota` se
calcula desde el `ContactLifecycle`, indexado por clave estable con histéresis
geométrica (`g_on < g_off`). En contacto persistente `iota = 0`, es decir,
restitución efectiva cero. Sin esto, una partícula apoyada rebota cada frame.

---

## D-011 — Perfil v3-H: fallo explícito

`RouterProfile.HYBRID` existe en el enum y en el parser de configuración, pero
`build_router` lanza `NotImplementedError` con un mensaje que nombra los tres
requisitos faltantes (fricción impulsiva, memoria tangencial persistente,
transición explícita entre regímenes). No hay fallback silencioso.

---

## D-012 — El MVP normal no predice `mu`

`ImpactHead` devuelve `mu = None` y `diagnostics.impact.mu = "disabled"`.
No se emite un tensor de ceros que pueda confundirse con "fricción aprendida
igual a cero".

---

## D-013 — Adimensionalización única y compartida

`L0 = 0.005 m` (diámetro de partícula), `T0 = 1e-3 s`, `M0 = masa de una
partícula`. `P0 = M0 L0 / T0`. Las escalas se aplican **una sola vez** en
`slgnn_experiments.nondimensionalization` y se guardan en cada checkpoint y
manifiesto. v3 y GNS reciben exactamente los mismos tensores adimensionales.

Cambio respecto del legacy (`T0 = 0.01 s`): con `T0 = 0.01` y
`dt = 1e-4 s`, el paso adimensional es `0.01`, y `u_n' ~ 2` para un impacto
típico. Con `T0 = 1e-3` el paso adimensional es `dt' = 0.1` y las velocidades
adimensionales quedan O(1), que es el rango donde `softplus` y `sigmoid`
tienen gradiente útil. El legacy no se modifica; v3 usa su propia escala y la
registra.

---

## D-014 — GNS controlado sin importar v3

`src/gns_baseline/` importa **solo** `slgnn_experiments`. Comparte el grafo
candidato, los targets, el sampler, los splits y las consultas de pared;
difiere únicamente en que decodifica `(Delta p, Delta L)` directamente por nodo
sin potencial, convexidad, solver ni conservación impuesta. El grafo físico no
ordenado se expande a aristas dirigidas dentro del baseline para message
passing (documentado en su docstring).

---

## D-015 — CASE07 fuera de toda selección

`CASE07` (extrapolación, ~3x energía cinética) se carga solo en el paso final
de evaluación. El runner rechaza con `ValueError` cualquier configuración que
lo incluya en `train_cases` o `val_case`.

---

## D-016 — `slgnn_experiments.scene` importa geometría de v3

`scene.py` importa `slgnn_v3.state`, `slgnn_v3.surfaces` y `slgnn_v3.graph`.
Esos tres módulos **no contienen física aprendida ni parámetros**: son
contenedores tipados y geometría. La prohibición de §15.1 es que
`gns_baseline` no importe el **modelo** v3, y se cumple estrictamente: el
baseline importa solo `slgnn_experiments`, y
`tests/comparison/test_shared_data.py::test_gns_baseline_does_not_import_v3_model`
lo verifica escaneando el código fuente en cada corrida.

La alternativa —duplicar la geometría en el paquete neutral— habría creado dos
definiciones del punto común y del gap, que es exactamente el fallo de
comparabilidad que estas reglas existen para evitar.

---

## D-017 — [INVARIANTE] El paso de FISTA usa la cota de Gershgorin

El tamaño de paso del solver proyectado es `1/L` con
`L = max_a sum_b |H_ab|`, que para `H` simétrica es **siempre** una cota
superior de `lambda_max`.

Se descartó estimar `lambda_max` por iteración de potencia inicializada en el
vector de unos: en una cadena de tres esferas, `H = [[2,-1],[-1,2]]` y
`(1,1)` es exactamente el autovector del autovalor **menor** (1), no del mayor
(3). La iteración devuelve `L = 1`, el paso queda tres veces por encima del
límite estable y FISTA diverge a `NaN`. El fallo se encontró con el test de
complementariedad multicontacto y quedó anclado como regresión en
`test_solver.py::test_lipschitz_is_an_upper_bound_on_lambda_max`.

Una subestimación de `lambda_max` no degrada la convergencia: la destruye. Una
sobreestimación solo cuesta iteraciones. La iteración de potencia se conserva,
con inicialización pseudoaleatoria de semilla fija, únicamente para el
diagnóstico de condicionamiento.

---

## D-018 — [INVARIANTE] Los descriptores de spin entran simétricamente

`kinematic_features` entrega `(|w_i| + |w_j|, ||w_i| - |w_j||)` y no
`(|w_i|, |w_j|)`. Con las dos entradas separadas, intercambiar `i` y `j`
cambia el vector de features y el paso completo deja de ser equivariante a
permutación: el error medido era `6.5e-9` relativo, muy por encima del ruido
de punto flotante en `float64`.

Lo detectó `test_symmetry.py::test_permutation_equivariance_of_the_full_step`.
Una arista física es no ordenada (§6.1) y sus features deben construirse
simétricamente; `u_n` y `|u_tau|` ya lo eran porque `u` y `n` se invierten a
la vez.

---

## D-019 — La normal se regulariza con un piso, no con un epsilon aditivo

`n = r_ij / max(d, eps)` en lugar de `n = r_ij / (d + eps)`. La forma aditiva
sesga la normal **en todas las distancias**: con `d = 1.2` y `eps = 1e-12`
introduce un error relativo de `8e-13` que se propaga al punto común y rompe
la igualdad exacta `r_i - r_j = q_j - q_i`. El clamp solo actúa en el caso
degenerado de dos centros coincidentes.

---

## D-020 — Ventanas de entrenamiento elegidas con la auditoría

Las ventanas de `frame_start`/`frame_stop` de los experimentos salen de
`docs/slgnn_v3/DATA_AUDIT.md`, no de un prefijo arbitrario. En
`sixty_gravity/CASE01` el primer contacto aparece cerca del snapshot **190**:
una ventana `[0, 120)` no contiene ni un solo contacto y un micro-overfit
sobre ella solo aprendería caída libre. Es el primer uso concreto de la
auditoría para tomar una decisión de diseño, que es para lo que existe.

---

## D-021 — Selección de `lr` con presupuesto idéntico por familia

Una sola rejilla `{3e-3, 1e-3, 3e-4, 1e-4}`, la misma semilla y el mismo
número de corridas para `v3-C`, `v3-I` y GNS, eligiendo por validación.

No es tuning asimétrico: es lo contrario. v3 arranca cerca del suelo de
pérdida —tiene la gravedad analítica y la unilateralidad incorporadas, así que
su pérdida inicial es del orden de la **final** de GNS— y tolera mal un paso
grande; GNS parte de cero y lo necesita. Fijar un `lr` común penalizaría a una
de las dos familias por una razón ajena a la arquitectura, que es justo lo que
§16.1 prohíbe.

Consecuencia metodológica: **el criterio de "reducción del 90–95 %" del
micro-overfit no es aplicable a v3 tal cual**. La reducción relativa mide
cuánta ignorancia tenía el modelo al empezar, no cuánto aprende. Se reportan
las tres cifras —inicial, final y reducción— y se compara la final.

---

## D-022 — [INVARIANTE] `Psi_tau` produce spin mediante `J^T`

La primera extensión posterior al MVP normal activa disipación tangencial
continua en `v3-C`:

```
lambda_tau = -d_tau(||u_tau||) u_tau / max(||u_tau||, eps)
```

con `d_tau = c1 s + c2 s^2`, `c1,c2 >= 0`. La fuerza se aplica mediante el
mismo `J^T` del resto de los contactos; no se agrega un decoder cartesiano de
torque. Por construcción, la potencia relativa es no positiva y el momento
angular orbital más spin de un contacto interno se conserva.

Los coeficientes de la familia polinómica no reciben `u_n`, `u_tau` ni
`omega`: `dissipation_context_features` entrega sólo contexto independiente
de la velocidad generalizada. Permitir que `c1` o `c2` dependan de la misma
velocidad invalidaría la garantía de convexidad que motiva `Psi`.

La bandera `state_independent_coefficients=True` hace esta elección explícita
y es obligatoria al activar `Psi_tau`. Su default es `False` exclusivamente
para poder cargar y reproducir checkpoints del MVP normal previo, cuyo
processor recibía las features cinemáticas. No se permite crear un modelo
tangencial nuevo bajo ese modo heredado.

Este canal **no se denomina memoria ni fricción de Coulomb**. Sin `M` no hay
sticking/sliding, estado `xi` ni proyección del trial. La configuración
histórica `mvp_c.yaml` permanece sin cambios; la extensión vive en
`v3_c_tangential.yaml` para que los resultados previos sigan reproducibles.
