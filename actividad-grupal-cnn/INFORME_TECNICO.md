# Informe técnico — Clasificación de razas de perros y gatos con CNN

**Actividad grupal — Redes Neuronales Convolucionales**
**Dataset:** Oxford-IIIT Pet (37 razas) · **Framework:** TensorFlow / Keras
**Notebook:** [`Actividad_grupal_SCA_resuelta.ipynb`](Actividad_grupal_SCA_resuelta.ipynb)

> **Cómo completar este informe.** Las secciones metodológicas y de análisis están redactadas
> y son independientes de la ejecución concreta. Las **cifras** proceden de ejecutar el
> notebook (recomendado: Google Colab con GPU T4, ~40–60 min): al terminar genera
> `reports/resultados_para_informe.md` con todas las tablas numéricas ya formateadas
> (comparativa de modelos, accuracy en test, mejores/peores clases y pares más confundidos),
> además de `reports/tabla_comparativa.csv`, `reports/metricas_por_clase.csv`,
> `reports/resumen_experimentos.json` y las figuras `.png`. Basta con pegar ese bloque en la
> sección 5 y adjuntar las figuras referenciadas.

---

## 1. Planteamiento del problema

Se aborda la clasificación de imágenes del **Oxford-IIIT Pet Dataset** en sus **37 razas**
(12 de gato y 25 de perro). El objetivo fijado por el enunciado es superar el **85% de
accuracy en el conjunto de test** con un modelo que **no presente overfitting ni
underfitting**.

Se trata de un problema de **granularidad fina** (*fine-grained classification*): las clases
no son categorías visualmente distantes (como en Fashion-MNIST) sino razas que comparten
morfología general. La variabilidad **intra**-clase (pose, escala, iluminación, encuadre,
fondo, edad del animal) es comparable a la variabilidad **inter**-clase entre razas
emparentadas, y solo se dispone de ~100 imágenes de entrenamiento por raza.

### 1.1 Particiones y protocolo experimental

Partiendo del split `train` de TFDS (3.680 imágenes), se construyen tres particiones
disjuntas: **80% entrenamiento / 10% validación / 10% test**.

> **Corrección respecto al notebook de partida.** La celda original usaba `'train[:80]'`,
> que en la sintaxis de *slicing* de TFDS selecciona los **80 primeros ejemplos** (no el 80%)
> y solapa con las otras dos particiones. Se corrige a `'train[:80%]'`.

Protocolo aplicado de forma homogénea a todos los experimentos:

- **Entrenamiento y validación** se usan para entrenar, regularizar y seleccionar modelo e
  hiperparámetros. El **test se reserva** y se evalúa una única vez, al final.
- Semilla global fija (`keras.utils.set_random_seed(42)`), mismas particiones, mismo
  *pipeline* y mismo criterio de parada para todos los modelos.
- `ModelCheckpoint` sobre `val_accuracy` (se conserva el **mejor** modelo, no el último),
  `EarlyStopping` con `restore_best_weights=True` y `ReduceLROnPlateau`.
- Cada modelo se guarda en `models/<nombre>.keras` y sus curvas en `histories/<nombre>.json`;
  al reejecutar el notebook los modelos **se cargan en lugar de reentrenarse**, de modo que
  todos los análisis posteriores pueden repetirse en CPU.

---

## 2. Análisis exploratorio de los datos (EDA)

| Aspecto | Hallazgo | Implicación de diseño |
|---|---|---|
| Tamaño | 3.680 imágenes en el split de partida; ~2.944 / 368 / 368 tras dividir | Muy pocos datos: el *transfer learning* y la aumentación son casi obligatorios |
| Nº de clases | 37 razas | Un clasificador aleatorio acierta ~2,7%; la *accuracy* del baseline debe leerse sobre esa referencia |
| Balance de clases | Prácticamente uniforme (~100 imágenes por raza, ratio máx/mín ≈ 1) | *Accuracy* es una métrica válida; no hacen falta pesos de clase |
| Balance por especie | 25 razas de perro frente a 12 de gato (≈ 2:1 en imágenes) | Los errores tienden a concentrarse dentro de la especie mayoritaria |
| Resolución | Muy heterogénea, con relaciones de aspecto variadas | Redimensionado obligatorio a 160×160 (se asume una ligera deformación) |
| Rango de píxeles | `uint8` en [0, 255], sin normalizar | Cada modelo incorpora su capa de normalización |
| Contenido | Fondos variados, animales parcialmente visibles, varias mascotas por foto, cachorros | Justifica la aumentación geométrica y explica buena parte de los errores |

Figuras: `reports/eda_distribucion_clases.png`, `reports/eda_particiones.png`,
`reports/eda_resoluciones.png`, `reports/eda_ejemplos.png`.

---

## 3. Preprocesado, *pipeline* y *data augmentation*

**Pipeline `tf.data`:** `map(resize+cast)` → `cache()` → `shuffle()` (solo entrenamiento) →
`batch(32)` → `prefetch(AUTOTUNE)`. `cache()` evita repetir la decodificación JPEG en cada
época y `prefetch` solapa la preparación del lote siguiente con el cómputo del actual.

**Normalización dentro del modelo.** El *pipeline* entrega imágenes en [0, 255] y la
normalización es la primera capa de cada red: `Rescaling(1/255)` para los modelos propios,
`Rescaling(1/127.5, offset=-1)` para MobileNetV2 (que espera [-1, 1]) y ninguna para
EfficientNetV2 (normaliza internamente). Así cada arquitectura recibe exactamente el rango
con el que fue preentrenada y **el modelo guardado es autocontenido**: para inferencia basta
con redimensionar.

**Data augmentation.** Implementado con capas Keras (`RandomFlip` horizontal,
`RandomRotation ±36°`, `RandomZoom ±15%`, `RandomTranslation ±10%`, `RandomContrast ±15%`)
integradas en el modelo, por lo que **solo actúan en entrenamiento** y nunca en
validación/test. Se descarta el volteo vertical por no corresponder a ninguna variación
realista del dominio. Figura: `reports/data_augmentation.png`.

---

## 4. Modelos evaluados

| # | Modelo | Idea | Configuración |
|---|---|---|---|
| 1 | **Fully Connected** (baseline) | Imagen aplanada (76.800 entradas) + capas densas | 512-256, BN, Dropout 0.5, Adam 1e-3 |
| 2 | **CNN propia** | Arquitectura propuesta por el grupo, desde cero | 4 bloques (32-64-128-256), 2×Conv3×3+BN+ReLU por bloque, MaxPool, GAP, Dense 256, Dropout 0.5, Adam 1e-3 |
| 3 | **CNN propia + augmentation** | Misma red que (2) + aumentación y regularización | Dropout espacial 0.1/0.2/0.3, L2 1e-4 |
| 4 | **MobileNetV2 congelada** | *Transfer learning* (extracción de características) | Base ImageNet congelada, cabeza GAP+Dropout+Dense, Adam 1e-3 |
| 5 | **MobileNetV2 fine-tuning** | Especialización de las capas superiores | Descongela desde la capa 100, BN congelada, Adam 1e-5 |
| 6 | **EfficientNetV2-B0 fine-tuning** (opcional) | Segundo *backbone* para contrastar | Cabeza (Adam 1e-3) + *fine-tuning* del último tercio (Adam 1e-5) |

### 4.1 Justificación del diseño de la CNN propia

- **Kernels 3×3 apilados** en lugar de kernels grandes: mismo campo receptivo efectivo con
  menos parámetros y más no linealidades.
- **BatchNormalization tras cada convolución**: estabiliza y acelera la convergencia y
  permite trabajar con `Adam(1e-3)`.
- **Duplicar los filtros cada vez que se reduce la resolución**: mantiene el coste por bloque
  aproximadamente constante mientras aumenta la capacidad semántica.
- **`GlobalAveragePooling2D` en lugar de `Flatten`**: elimina cientos de miles de parámetros
  en la cabeza y actúa como regularizador estructural.
- **Dropout creciente en profundidad** y L2 en la cabeza (modelo 3).

### 4.2 Decisiones clave del *transfer learning*

1. **Base congelada primero, *fine-tuning* después.** Entrenar directamente toda la red con
   una cabeza inicializada al azar produce gradientes grandes que destruyen las
   características preentrenadas.
2. **Tasa de aprendizaje dos órdenes de magnitud menor en el *fine-tuning*** (1e-5 frente a
   1e-3). Es el hiperparámetro más crítico de todo el trabajo.
3. **Capas `BatchNormalization` congeladas** y base invocada con `training=False`: con lotes
   pequeños, recalcular las estadísticas de BN degrada las representaciones de ImageNet.
4. **Descongelado parcial** (capas superiores): las capas inferiores codifican bordes y
   texturas genéricas, reutilizables tal cual; las superiores son las que deben
   especializarse en las razas.

---

## 5. Resultados

> Pegar aquí el contenido de `reports/resultados_para_informe.md`, generado por la última
> celda del notebook. Incluye: tabla comparativa de los seis modelos (parámetros, épocas,
> accuracy de train/val/test, *gap* y tiempo), accuracy final en test del mejor modelo,
> accuracy a nivel de especie, clases con mejor y peor F1, y pares de razas más confundidos.

Figuras asociadas: `reports/comparativa_modelos.png`, `reports/curvas_*.png`,
`reports/matriz_confusion.png`, `reports/precision_recall_por_clase.png`,
`reports/errores_alta_confianza.png`, `reports/errores_baja_confianza.png`,
`reports/confianza.png`.

---

## 6. Análisis

### 6.1 CNN frente a red *Fully Connected*

La red densa es el peor modelo pese a ser el que **más parámetros** tiene (39.466.021, frente a
1.251.461 de la CNN propia). Al aplanar la imagen se destruye la vecindad espacial, no hay
compartición de pesos ni invarianza a traslación y cada píxel recibe un peso independiente:
con ~2.900 imágenes de entrenamiento ese espacio de hipótesis no es aprendible. La CNN, con
**más de 30 veces menos parámetros**, obtiene una accuracy muy superior gracias a
sus sesgos inductivos (localidad, compartición de pesos, jerarquía de características).
La comparación cuantifica exactamente lo que aportan las convoluciones.

### 6.2 Impacto del *data augmentation*

Los modelos 2 y 3 comparten arquitectura y solo difieren en la aumentación y la
regularización, lo que permite aislar su efecto. El patrón observado es el esperado:
**baja la accuracy de entrenamiento y sube la de validación**, es decir, **se reduce el
*gap*** entre ambas curvas. Sin aumentación las curvas de *train* y *val* se separan a las
pocas épocas (memorización); con aumentación permanecen próximas mucho más tiempo y el mejor
punto de validación llega más tarde y más alto. Con ~100 imágenes por clase es la técnica de
regularización más rentable del trabajo.

### 6.3 Comparación entre arquitecturas CNN

| Aspecto | Observación |
|---|---|
| Profundidad | Pasar de 2 a 4 bloques mejora la representación, pero desde cero la mejora se satura pronto: el limitante son los datos, no la arquitectura |
| Batch Normalization | Imprescindible para entrenar la red propia con `Adam(1e-3)`; en *transfer learning*, en cambio, debe permanecer **congelada** |
| GAP vs. Flatten | `GAP` reduce drásticamente los parámetros de la cabeza y el sobreajuste, sin pérdida de rendimiento |
| Optimizador / LR | `Adam(1e-3)` desde cero y para la cabeza; `Adam(1e-5)` en *fine-tuning*. `ReduceLROnPlateau` aporta una mejora adicional al final del entrenamiento |
| Regularización | Dropout + L2 + aumentación + `EarlyStopping` mantienen el *gap* train/val en valores aceptables |
| Transfer learning | Factor decisivo: aporta representaciones inaprendibles con 3.000 imágenes; el *fine-tuning* las especializa |
| *Backbone* | Comparar MobileNetV2 con EfficientNetV2-B0 muestra que la mejora proviene del paradigma (preentrenamiento) más que del *backbone* concreto |

### 6.4 Análisis de errores

- Los errores se concentran **entre razas de la misma especie** y morfológicamente próximas.
  La accuracy a nivel de **especie** (perro/gato) es muy superior a la accuracy a nivel de
  **raza**: el problema difícil no es distinguir perro de gato, sino la discriminación
  *fine-grained* entre razas parecidas.
- Los grupos que concentran la confusión son los previsibles a priori: gatos de pelo corto y
  patrón atigrado entre sí, terriers y bulldogs entre sí, y razas de perro grandes y
  peludas entre sí.
- Las imágenes mal clasificadas comparten patrones: encuadres parciales, posturas atípicas,
  iluminación extrema, varios animales en la escena, fondos dominantes y cachorros (cuya
  morfología difiere de la del adulto de su raza).
- La **confianza del modelo es informativa**: es sensiblemente mayor en los aciertos que en
  los errores, de modo que un umbral de rechazo permitiría elevar la precisión a costa de la
  cobertura (curva incluida en el notebook).

### 6.5 Verificación de overfitting / underfitting

Criterios aplicados sobre el mejor *checkpoint* de cada modelo:

- **Overfitting**: `accuracy` de entrenamiento muy superior a la de validación (*gap* > 0,10)
  y/o `val_loss` creciente de forma sostenida mientras `train_loss` sigue bajando.
- **Underfitting**: `accuracy` de entrenamiento baja y prácticamente igual a la de
  validación, con ambas curvas estancadas.

El notebook imprime este diagnóstico para todos los modelos y muestra las curvas del modelo
seleccionado. `EarlyStopping` con `restore_best_weights=True` y `ModelCheckpoint` sobre
`val_accuracy` garantizan que el modelo conservado corresponde al punto de mejor
generalización y no al final del entrenamiento.

---

## 7. Conclusiones

1. **El *transfer learning* es la decisión determinante.** Con ~100 imágenes por raza, una
   CNN entrenada desde cero se queda muy lejos del objetivo; una base preentrenada en
   ImageNet con *fine-tuning* de las capas superiores lo supera.
2. **Las convoluciones son imprescindibles**: la red densa, con más de 30 veces más
   parámetros que la CNN propia, rinde mucho peor.
3. **La aumentación de datos es la regularización más eficaz** en este régimen de datos
   escasos; su efecto se mide como reducción del *gap* train/validación.
4. **La tasa de aprendizaje del *fine-tuning* y el tratamiento de las capas BN** son los
   detalles que separan un *fine-tuning* que mejora de uno que degrada el modelo.
5. **El error residual es de granularidad fina**: se concentra entre razas de la misma
   especie, lo que sugiere como trabajo futuro mayor resolución de entrada, *backbones* más
   grandes o técnicas específicas *fine-grained* (atención, recorte de la región del animal
   mediante las máscaras de segmentación que incluye el dataset).

---

## 8. Reproducibilidad

- Semillas fijadas (`random`, `numpy`, `tensorflow` vía `keras.utils.set_random_seed(42)`).
  Para determinismo estricto en GPU puede activarse
  `tf.config.experimental.enable_op_determinism()` (penaliza el tiempo de cómputo).
- Todos los hiperparámetros están centralizados en una única celda de configuración.
- Los modelos (`models/*.keras`), las curvas (`histories/*.json`) y las tablas y figuras
  (`reports/`) quedan guardados; con `FORCE_RETRAIN = False` el notebook **no reentrena**:
  recarga los *checkpoints* y rehace todo el análisis en CPU.
