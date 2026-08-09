# MATHEMATICAL_CONTRACT — SLGNN-v3

Contrato matemático que la implementación debe satisfacer, con el archivo y el
test que lo verifican. Nada de esta lista es aspiracional: cada fila tiene una
prueba ejecutable.

---

## 1. Ecuación discreta central

```
M (nu_{k+1} - nu_k) = dt F_reg,k + J_k^T Lambda_k
F_reg = -grad_q V_theta - d_nu Psi_theta + F^M_theta + F_ext
```

con el split semiimplícito

```
nu*      = nu_k + dt M^-1 F_reg,k
nu_{k+1} = nu*   + M^-1 J^T Lambda
q_{k+1}  = q_k + dt K nu_{k+1}        (velocidad POST-impulso)
```

| Elemento | Implementación | Test |
|---|---|---|
| Paso completo, no aceleración | [model.py](../../src/slgnn_v3/model.py) `SLGNNv3.step` | `test_integrator.py::test_step_result_is_not_just_acceleration` |
| Igualdad exacta de la ecuación | [integrator.py](../../src/slgnn_v3/integrator.py) | `test_integrator.py::test_central_discrete_equation_holds` |
| Posición con velocidad post-impulso | `advance_position` | `test_integrator.py::test_position_uses_post_impulse_velocity` |

---

## 2. Separación de régimen

`gamma_alpha in {0, 1}` por contacto y paso. `gamma = 0` contribuye por
`V + Psi`; `gamma = 1` por el solver de `I`; vuelo libre no produce respuesta
aunque exista arista candidata.

| Garantía | Implementación | Test |
|---|---|---|
| Modos disjuntos, sin doble conteo | [router.py](../../src/slgnn_v3/router.py) `assert_no_double_counting` | `test_router.py::test_router_modes_are_disjoint` |
| `v3-C` no produce impulsos | `compliant_router` | `test_router.py::test_compliant_profile_produces_no_impulses` |
| `v3-I` no produce potencial | `impulsive_router` | `test_router.py::test_impulsive_profile_produces_no_potential` |
| `v3-H` falla explícitamente | `build_router` | `test_router.py::test_hybrid_profile_fails_loudly` |

---

## 3. Geometría y convenciones

| Contrato | Valor | Test |
|---|---|---|
| SDF | `phi > 0` dentro, `grad(phi)` entrante, `g = phi - R` | `test_surfaces.py::test_sign_convention_inside_positive` |
| Velocidad relativa | `u_n < 0` es aproximación (par y pared) | `test_contact_kinematics.py::test_relative_velocity_sign_convention` |
| Punto común | `x_c = ((q_i + R_i n) + (q_j - R_j n))/2` | `test_contact_kinematics.py::test_common_point_swap_invariance` (`atol 1e-12`) |
| Brazos | `r_i - r_j = q_j - q_i` **exacto** | `test_contact_kinematics.py::test_arms_difference_is_exact` |
| Multi-superficie | arista = 2 caras, esquina = 3 | `test_surfaces.py::test_edge_gives_two_surfaces`, `..._corner_...` |
| Tiempo de pared | `v_W` cambia con `t` a SDF constante | `test_surfaces.py::test_wall_velocity_changes_with_time_at_fixed_geometry` |
| Batching | cero aristas entre ejemplos | `test_no_batch_leakage.py` |

### Compresión unilateral

```
p_eps(x) = 0                            x <= 0
         = eps (6 z^3 - 8 z^4 + 3 z^5)  0 < x < eps,  z = x/eps
         = x                            x >= eps
```

`delta = p_eps(-g)` es C² y **exactamente** cero en separación, a diferencia
de `softplus(-g, beta)`, que vale `log(2)/beta`.
Test: `test_contact_kinematics.py::test_positive_part_exactly_zero_in_separation`
y `..._c2_continuity`.

---

## 4. Operadores de contacto

| Prueba obligatoria (§5.3) | Test | Tolerancia |
|---|---|---|
| `J nu` coincide con construcción directa | `test_J_matches_direct_construction` | exacto |
| Identidad adjunta `<J nu, l> = <nu, J^T l>` | `test_adjoint_identity` | `1e-10` rel |
| Contacto interno igual y opuesto | `test_internal_contact_is_equal_and_opposite` | `1e-14` |
| Momento angular orbital + spin | `test_angular_momentum_orbital_plus_spin` | `1e-10` rel |
| Invarianza a permutación | `test_permutation_invariance` | `1e-12` |
| Equivarianza `SE(3)` | `test_se3_equivariance_of_operators` | `1e-8` rel |

`A_n = J_n M^-1 J_n^T` se materializa densa **solo por componente conexa**;
`J` global densa nunca. Tests: `test_delassus_symmetric_and_psd`,
`test_delassus_matches_operator_application`,
`test_components_couple_shared_particles`.

---

## 5. Cabeza conservativa `V`

```
f_n(delta, h) = f0 a(delta) softplus(k_theta(delta, h)),   a(0) = 0
U(delta, h)   = int_0^delta f_n(s, h) ds        (Gauss-Legendre, 8 nodos)
F_V = -grad_q V                                 (autograd)
```

| Garantía | Test | Tolerancia |
|---|---|---|
| `U(0) = 0` | `test_U_zero_at_zero_compression` | `1e-12` |
| `dU/d(delta) >= 0` | `test_dU_ddelta_non_negative` | `-1e-10` |
| `dU/d(delta) = f_n` | `test_dU_ddelta_equals_normal_force` | `1e-9` |
| Fuerza repulsiva y acción-reacción | `test_force_is_repulsive_on_a_real_pair` | `1e-12` |
| Cero exacto en separación | `test_no_force_when_separated` | exacto |
| `V` no recibe velocidades | `test_potential_receives_no_velocity` | `1e-14` |
| Diferencias finitas vs autograd | `test_finite_differences_vs_autograd_force` | `1e-5` rel |

La gravedad es fuerza externa analítica y **no** está en `V` (D-006):
`test_integrator.py::test_gravity_counted_exactly_once`.

---

## 6. Cabeza disipativa `Psi`

```
psi(s) = c1 s^2/2 + c2 s^3/3,   c1, c2 >= 0 (softplus), independientes de s
psi'(s) = c1 s + c2 s^2 >= 0
psi''(s) = c1 + 2 c2 s >= 0
lambda^Psi = psi'(s_n) n,   s_n = (-u_n)_+
```

| Garantía | Test | Tolerancia |
|---|---|---|
| `psi(0) = 0` | `test_psi_zero_at_zero` | `1e-12` |
| `psi' >= 0`, `psi'' >= 0` | `test_psi_first_and_second_derivative_non_negative` | `-1e-10` |
| Potencia relativa `<= 0` | `test_relative_power_is_non_positive` | `1e-10` |
| Inactiva en separación | `test_dissipation_inactive_when_separated` | exacto |

---

## 7. Solver impulsivo

```
H = A_n + diag(kappa)
b = u_n^* + iota e min(u_n^-, 0) + beta min(g, 0)/dt
lambda = argmin_{lambda >= 0} [ 1/2 lambda^T H lambda + b^T lambda ]
0 <= lambda _|_ u_n^+ + b_n + kappa lambda >= 0
```

| Garantía | Test | Tolerancia |
|---|---|---|
| Caso analítico `Lambda = -(1+e) u_n^-/(1/m_i + 1/m_j)` | `test_two_body_head_on_analytic` | `1e-8` rel |
| Velocidad postimpacto y restitución efectiva | `test_two_body_post_impact_velocity_and_restitution` | `1e-12` |
| Solve acoplado ≠ solve independiente | `test_coupled_solve_differs_from_independent` | `>1e-3` |
| Complementariedad en evaluación | `test_coupled_solve_satisfies_complementarity` | `1e-7` |
| Paso de FISTA estable | `test_lipschitz_is_an_upper_bound_on_lambda_max` | Gershgorin |
| Diferenciable | `test_solution_is_differentiable` | finito y no nulo |

Restitución **solo al nacimiento** (`iota = 1`):
`test_router.py::test_restitution_applied_only_at_birth`.

`D_impact >= 0` está garantizado solo con `beta = 0`; la estabilización de
Baumgarte inyecta energía a propósito y su magnitud se reporta como
`impact.stabilization_term_max`.

---

## 8. Simetrías del paso completo

| Garantía | Test | Tolerancia |
|---|---|---|
| Equivarianza `SE(3)` del paso | `test_se3_equivariance_of_the_full_step` | `1e-8` rel |
| Equivarianza a permutación del paso | `test_permutation_equivariance_of_the_full_step` | `1e-10` rel |
| `V` conserva momento interno | `test_potential_conserves_internal_momentum_structurally` | `1e-12` |
| Sistema aislado conserva `p` y `L` | `test_internal_momentum_conserved_without_walls` | `1e-12` / `1e-8` |
| Doble backward | `test_double_backward_through_the_step` | finito, no nulo |

La equivarianza es **condicionada**: se transforma el sistema físico completo
(partículas, gravedad y pared). Rotar solo las partículas describe otro
experimento.

---

## 9. Targets y pérdidas

```
Delta p_i = m_i (v_i^{k+1} - v_i^k)
Delta L_i = I_i (omega_i^{k+1} - omega_i^k)
L_dp = (1/N) sum_i || (dp_theta - dp_DEM) / P0 ||^2
```

Escalas: `P0 = M0 L0 / T0`, `L0 P0 = M0 L0^2 / T0`. Adimensionalización única,
aplicada exactamente una vez (`test_shared_data.py::
test_nondimensionalization_applied_exactly_once`).

Una propiedad garantizada por construcción **no** se duplica como
penalización: conservación y complementariedad se monitorean; la penetración
sí es una pérdida legítima porque con paso finito no está garantizada.

---

## 10. Lo que el MVP normal **no** implementa

| Canal | Estado | Dónde se declara |
|---|---|---|
| `M` memoria tangencial | contrato, sin implementar | [memory.py](../../src/slgnn_v3/memory.py) — toda función levanta `NotImplementedError` |
| `C` cierre residual | congelado | [closure.py](../../src/slgnn_v3/closure.py) — `ClosureHead(enabled=True)` falla |
| Fricción impulsiva `mu` | desactivada | `ImpactHead` devuelve `mu = None`, no ceros |
| `Psi` tangencial y rotacional | desactivados | `DissipationHead` falla si se activan |
| Perfil `v3-H` | reservado | `build_router` falla con los tres requisitos nombrados |

Consecuencia medible y probada: **`Delta L` es exactamente cero** en el MVP.
`test_symmetry.py::test_mvp_produces_exactly_zero_angular_momentum`.
