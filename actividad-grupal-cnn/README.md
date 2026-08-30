# Actividad grupal — Redes Neuronales Convolucionales (Oxford-IIIT Pet)

Solución completa de la actividad de clasificación de **razas de perros y gatos** (37 clases
del *Oxford-IIIT Pet Dataset*) con CNN en TensorFlow/Keras.

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/careb36/careb36/blob/main/actividad-grupal-cnn/Actividad_grupal_SCA_resuelta.ipynb)

## Contenido

| Fichero | Descripción |
|---|---|
| `Actividad_grupal_SCA_resuelta.ipynb` | Notebook con la solución completa: EDA, *pipeline* `tf.data`, seis modelos, comparativa, evaluación final en test y análisis de errores |
| `INFORME_TECNICO.md` | Informe técnico: metodología, decisiones de diseño, análisis y conclusiones |
| `requirements.txt` | Dependencias mínimas |

## Cómo ejecutarlo

1. Abrir el notebook en **Google Colab** y activar la GPU
   (*Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU*).
2. Ejecutar todas las celdas. La primera ejecución descarga el dataset (~800 MB) y entrena
   los seis modelos (aproximadamente 40–60 minutos en una GPU T4).
3. Al finalizar quedan generados:
   - `models/*.keras` — mejor *checkpoint* de cada modelo (criterio: `val_accuracy`),
   - `histories/*.json` — curvas de entrenamiento,
   - `reports/` — figuras, tablas (`.csv`), `resumen_experimentos.json` y
     `resultados_para_informe.md` (bloque de resultados listo para pegar en el informe).

Al reejecutar el notebook con `FORCE_RETRAIN = False` (valor por defecto) **no se reentrena
nada**: los modelos se cargan desde `models/` y todo el análisis se rehace en CPU.

## Estructura de la solución

| Sección | Contenido |
|---|---|
| 1–2 | Configuración, semillas y particiones 80/10/10 del split `train` |
| 3 | Análisis exploratorio: distribución de clases, balance por especie, resoluciones, ejemplos |
| 4 | *Pipeline* `tf.data`, normalización dentro del modelo y *data augmentation* |
| 5 | Utilidades de entrenamiento (`ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`) y registro experimental |
| 6 | **Modelo 1** — red *Fully Connected* sobre imágenes aplanadas (baseline) |
| 7 | **Modelo 2** — CNN propia entrenada desde cero |
| 8 | **Modelo 3** — CNN propia + *data augmentation* + regularización |
| 9 | **Modelo 4** — MobileNetV2 congelada (*feature extraction*) |
| 10 | **Modelo 5** — *fine-tuning* de MobileNetV2 (modelo seleccionado) |
| 11 | **Modelo 6** (opcional) — *fine-tuning* de EfficientNetV2-B0 |
| 12–13 | Comparativa de modelos y diagnóstico de *overfitting* / *underfitting* |
| 14–15 | Evaluación final en test (precision/recall por clase, matriz de confusión) y análisis visual de errores |
| 16 | Conclusiones e informe |

## Notas

- El conjunto de **test se utiliza exclusivamente para la evaluación final**; la selección de
  modelo e hiperparámetros se realiza con la partición de validación.
- Se corrige el *slicing* del notebook de partida: `'train[:80]'` (los 80 primeros ejemplos,
  además solapado con las demás particiones) → `'train[:80%]'`.
- Las imágenes no vienen normalizadas: la normalización se aplica como primera capa de cada
  modelo, con el rango que espera cada arquitectura.
