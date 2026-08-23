# Informe de sesión — SLGNN: datos, enfoque de implementación y plan de trabajo

**Proyecto:** SLGNN (SDF–Lagrangian Graph Neural Network) — modelo sustituto neuronal para dinámica granular en molinos SAG
**Curso:** MAT2320
**Fecha del informe:** 6 de julio de 2026
**Alcance de este documento:** registrar las decisiones tomadas en esta sesión sobre qué datos se usarán y por qué, cómo se comenzará a programar la arquitectura, y el plan de mediano plazo acordado. La formulación matemática completa de SLGNN vive en la *Presentación formal del proyecto*; este informe no la reemplaza, sino que documenta las decisiones de ejecución que la anteceden.

---

## 1. Punto de partida

SLGNN existe hoy como una formulación arquitectónica y matemática completa, pero sin validación empírica. El objetivo de mediano plazo es llevarla a código, conseguir datos, y alcanzar un setup lo bastante maduro como para correr pruebas y mini-entrenamientos en CPU, dejando el entrenamiento a escala y la comparación cuantitativa contra GNS y SGN para una fase posterior.

Tres restricciones prácticas condicionan todo el diseño y fueron fijadas en esta sesión:

- Los datos deben provenir de fuentes **públicas** (no se generará DEM propio).
- Un **tambor / cilindro rotatorio** es un escenario aceptable como objetivo; no es obligatorio el molino SAG completo.
- El cómputo disponible es **solo CPU**.

---

## 2. Datos: qué usaremos y por qué

### 2.1 La tensión a resolver

El componente distintivo de SLGNN es la pared móvil representada mediante una *Signed Distance Function* (SDF). Validar eso exige datos con frontera en movimiento. El problema es que las dos condiciones —"datos públicos" y "frontera móvil"— normalmente no se cruzan:

- Los datasets granulares públicos y descargables (los de GNS en DesignSafe, tipo colapso de columna) tienen **frontera estática**: masa granular dentro de una caja o rampa fija.
- Los datos con pared móvil del modelo SGN de Li y Sakai —la referencia directa de la SDF en el proyecto— **no son de descarga libre**: se entregan solo a solicitud de los autores.

### 2.2 Dataset elegido

Se usará el conjunto de datos DEM publicado junto al trabajo **Dynami-CAL GraphNet** (Sharma & Fink, EPFL), disponible en Zenodo:

- **Registro:** Zenodo, DOI `10.5281/zenodo.17589419`
- **Título:** *6 DoF Dynamics: Discrete Element Method (DEM) Simulation Dataset for Learning GNN Surrogate Model*
- **Licencia:** CC-BY-4.0 (uso libre citando a los autores)
- **Generación:** MFiX DEM, trayectorias 6-DoF resueltas en el tiempo, con condiciones de frontera explícitas y parámetros de contacto
- **Volumen total:** ~260 MB

Este dataset resuelve la tensión: es **público** y a la vez contiene un **cilindro rotatorio** (frontera móvil), que es esencialmente un tambor rotatorio, es decir, un molino SAG simplificado.

### 2.3 Por qué encaja con las tres restricciones

- **Público:** licencia abierta CC-BY-4.0.
- **Frontera móvil / tambor rotatorio:** el cilindro rotatorio da la SDF dependiente del tiempo y la velocidad de pared, que es el corazón de SLGNN.
- **Solo CPU:** el conjunto de entrenamiento base son 60 esferas en una caja estática, escala genuinamente tratable en CPU. La estructura natural del dataset —**entrenar barato en una caja estática pequeña y evaluar la extrapolación al cilindro rotatorio**— coincide además con la propia tesis de SLGNN: que los sesgos inductivos físicos permiten extrapolar a fronteras y configuraciones no vistas.

### 2.4 Ventaja adicional: baselines casi listos

El trabajo de origen ya implementó **GNS** como baseline sobre estos mismos datos, y documentó que GNS "caja negra" se desestabiliza y no generaliza a la frontera rotatoria, mientras que un modelo con estructura física mantiene rollouts estables por miles de pasos. Esto nos entrega:

- Un baseline GNS ya caracterizado, y un contraste perfecto para la hipótesis de SLGNN.
- Un segundo modelo físico-informado (Dynami-CAL, que conserva momento lineal y angular) como punto de comparación adicional.

### 2.5 Inventario de archivos y su rol en el plan

| Archivo | Tamaño | Rol previsto |
|---|---|---|
| `Detailed_Information_on_Data_Structure.pdf` | 450 kB | Documentación del esquema de datos |
| `Benchmark_2Spheres_Oblique_Collision.zip` | 93 kB | Test unitario del canal **partícula–partícula** (potencial + disipación) |
| `Benchmark_1Sphere_Multiple_Wall_Collision.zip` | 302 kB | Test unitario del canal **partícula–pared (SDF)** |
| `60Spheres_Homogeneous_Interaction_Inside_Cuboidal_Enclosure.zip` | 42 MB | Primer entrenamiento real (material homogéneo, caja) |
| `60Spheres_Gravity_Inside_Cuboidal_Enclosure.zip` | 32 MB | Entrenamiento con gravedad y contactos heterogéneos |
| `Extrapolation_2073Spheres_Gravity_Inside_Rotating_Cylinder.zip` | 185 MB | Test de frontera móvil / extrapolación (solo inferencia) |

Los dos micro-benchmarks son especialmente valiosos: permiten validar cada canal de SLGNN por separado antes de combinarlos.

### 2.6 Salvedades registradas

- **Formato interno por confirmar.** El esquema exacto de los archivos está en el PDF de estructura de datos incluido en el registro, que no pudo leerse por medios automáticos en esta sesión. Se confirmará al descargar. Al provenir de MFiX, lo más probable son salidas por paso de tiempo (CSV o VTK/`.vtp`) más archivos de parámetros de frontera y contacto.
- **Datos 3D y 6-DoF.** Incluyen rotación de las esferas. La versión base de SLGNN es translacional, así que se usarán posiciones y velocidades de los centros, dejando los grados angulares como extensión futura (coherente con el alcance declarado en la presentación).
- **Sin código público de Dynami-CAL.** Solo se liberaron los datos; se reimplementa lo necesario. Esto refuerza el carácter de aporte propio de SLGNN.
- **La descarga requiere red.** Se realizará en el entorno local del proyecto, que es además donde ocurrirá el entrenamiento en CPU.

---

## 3. Cómo empezaremos a programar SLGNN

### 3.1 Framework y referencias

- **Stack unificado:** PyTorch + PyTorch Geometric, con build de **CPU**.
- **Esqueleto base:** la implementación GNS de `geoelements/gns` (PyTorch/PyG, licencia MIT) se reutiliza como plomería probada del esquema encoder–processor–decoder y del rollout, y sirve además como baseline directo. No se reescribe esa maquinaria desde cero.
- **Referencia conceptual:** las implementaciones de LGNN (`ravinderbhattoo/LGNN`, `M3RG-IITD/LGNN`) se consultan solo como guía del mecanismo de Euler–Lagrange y la autodiferenciación del lagrangiano; no se fusionan sus códigos (están en otro framework).
- **Pared por SDF:** se reconstruye desde el paper de SGN (Li & Sakai) que ya está en el material del proyecto.

### 3.2 Sustrato compartido

GNS y SLGNN comparten dos piezas que se construyen una sola vez: el **constructor de grafo** (aristas por radio de vecindad, objetos PyG reconstruidos en cada paso) y la **SDF analítica** de la caja (planos) y del cilindro rotatorio (con velocidad de pared derivada de la velocidad angular). Esto garantiza una comparación limpia entre ambos modelos.

### 3.3 La descomposición física de SLGNN

La aceleración no sale de una MLP libre, sino de una estructura mecánica explícita que se irá activando por capas:

- Potenciales aprendidos partícula–partícula ($V_{pp}$) y partícula–pared ($V_{\text{wall}}$), con masas conocidas y gravedad explícita, dentro de un lagrangiano.
- Disipación de tipo Rayleigh con coeficientes no negativos, que siempre se opone al movimiento relativo.
- Canal de residuo histórico controlado (fricción con memoria), como capa opcional posterior.
- Fuerza obtenida vía Euler–Lagrange como $-\nabla_q V$ mediante autograd (`create_graph=True`, es decir un *doble backward*), integrada con un esquema Verlet / semiimplícito.

### 3.4 Construcción incremental validada por canal

El orden de construcción evita depurar todo a la vez:

1. Potencial partícula–partícula + gravedad + integrador simple → validado con el benchmark de **2 esferas**.
2. Se añade el potencial de pared vía SDF → validado con el benchmark de **1 esfera–pared**.
3. Se añade la disipación de Rayleigh.
4. Se añade, opcional, el canal histórico.

---

## 4. Plan de mediano plazo (hitos M0–M5)

Cada hito tiene un entregable y un criterio de "listo cuando", para avanzar sin arrastrar deuda técnica.

**M0 — Datos en local y esquema entendido.**
Descargar, verificar checksums, descomprimir y leer el PDF de estructura para documentar campos, unidades, paso de tiempo y —sobre todo— cómo se codifica el movimiento del cilindro rotatorio.
*Entregable:* `DATA_NOTES.md` con el esquema. *Listo cuando:* se puede describir exactamente el contenido de cada archivo y la representación de la pared móvil.

**M1 — Entorno reproducible.**
Entorno Python aislado con dependencias fijadas (PyTorch CPU, PyG, numpy, lector según formato). Git desde el inicio.
*Entregable:* `requirements.txt` + repo + README. *Listo cuando:* `import torch, torch_geometric` funciona y un smoke test corre.

**M2 — Ingesta y exploración (EDA).**
Loader mínimo del benchmark de 2 esferas a tensores `[T, N, d]` + metadatos, con chequeo físico de cordura (trayectorias, energía, `dt`, unidades).
*Entregable:* `load_dataset.py` + notebook exploratorio. *Listo cuando:* se carga y visualiza una trayectoria y la colisión se ve físicamente correcta.

**M3 — Representación común: grafo + SDF.**
Constructor de grafo (PyG, reconstrucción por paso) y SDF analítica de caja y cilindro rotatorio, con velocidad de pared. Verificar signo y gradiente contra diferencias finitas.
*Entregable:* `graph_builder.py`, `sdf.py` con tests. *Listo cuando:* de un frame se obtiene un grafo válido y la SDF entrega distancia con signo y gradiente correctos.

**M4 — Baseline GNS mínimo.**
Adaptar `geoelements/gns` al loader y entrenar a escala minúscula sobre 60 esferas homogéneas como smoke test; obtener un rollout de referencia y una métrica (MSE de posición, estabilidad de rollout).
*Entregable:* scripts train/inferencia + rollout + métrica. *Listo cuando:* GNS entrena en CPU y produce rollout; queda un loop train/eval reutilizable.

**M5 — Andamiaje de SLGNN + arnés de pruebas (el "setup maduro").**
Esqueleto modular de SLGNN sobre el mismo sustrato (potenciales, Rayleigh, fuerza vía Euler–Lagrange con doble backward, integrador), cada pieza tras una interfaz limpia. Arnés de tests con los dos micro-benchmarks y un script de mini-entrenamiento que hace *overfit* de una trayectoria corta.
*Entregable:* paquete `slgnn/` + `tests/` + script de mini-train. *Listo cuando:* (a) cada canal pasa su micro-benchmark aislado y (b) SLGNN sobreajusta una trayectoria corta en CPU con pérdida decreciente y rollout estable por unos pasos.

**Más allá de M5 (fase siguiente):** entrenamiento real, comparación cuantitativa contra GNS, y extrapolación al cilindro rotatorio. Se detallará al llegar.

---

## 5. Advertencias transversales

- **Reproducibilidad desde el día uno:** fijar semillas y guardar configuraciones (YAML) por experimento desde M2, porque depurar un GNN lagrangiano con doble backward en CPU es lento y se necesita reproducibilidad exacta.
- **Mantener todo diminuto:** float32, redes chicas, pocos pasos. El costo por paso de SLGNN es mayor que el de GNS por la autodiferenciación del lagrangiano, así que a 2073 esferas solo se hará inferencia, nunca entrenamiento.

---

## 6. Estructura de proyecto acordada

```
slgnn-project/
├── data/{raw, extracted, DATA_NOTES.md}
├── src/slgnn/
├── tests/
├── notebooks/
├── requirements.txt
└── README.md
```

---

## 7. Referencias

- **Dataset:** Sharma, V. & Fink, O. *6 DoF Dynamics: DEM Simulation Dataset for Learning GNN Surrogate Model.* Zenodo, DOI 10.5281/zenodo.17589419. Licencia CC-BY-4.0.
- **Paper del dataset:** Sharma, V. & Fink, O. *A physics-informed graph neural network conserving linear and angular momentum for dynamical systems.* Nature Communications 17, 1045 (2026). Preprint: arXiv:2501.07373.
- **SGN (pared vía SDF):** Li & Sakai, modelo sustituto GNN basado en signed distance function para flujos granulares en dominios de forma arbitraria (material del proyecto).
- **GNS (baseline y esqueleto):** repositorio `geoelements/gns` (implementación PyTorch/PyG, licencia MIT).
- **LGNN (referencia conceptual Euler–Lagrange):** repositorios `ravinderbhattoo/LGNN` y `M3RG-IITD/LGNN`.
- **Formulación completa de SLGNN:** *Presentación formal del proyecto SLGNN* (documento del proyecto).
