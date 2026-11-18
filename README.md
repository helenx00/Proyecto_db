# Proyecto Final - Ciencia de Datos

## Universidad de San Buenaventura

### Integrantes
- Helen Lavao
- Brayan Afanador
- David Pira

## Descripción del proyecto

Este proyecto aplica técnicas de Ciencia de Datos y Machine Learning para analizar el desempeño académico de estudiantes colombianos en las pruebas Saber 11.

Se utiliza un conjunto de datos oficial del portal Datos Abiertos de Colombia, correspondiente al periodo 2020-2.

El proyecto incluye exploración, limpieza y preparación de datos, entrenamiento de modelos predictivos y una aplicación web interactiva desarrollada con Streamlit.

## Problemática

En Colombia existen diferencias en el desempeño académico que pueden estar relacionadas con factores socioeconómicos, familiares y educativos.

El proyecto busca estudiar asociaciones entre estas características y los resultados de las pruebas Saber 11, utilizando técnicas estadísticas y de aprendizaje automático.

## Objetivo general

Analizar factores asociados al desempeño académico en las pruebas Saber 11 y desarrollar modelos de Machine Learning para estimar el puntaje global de los estudiantes.

## Objetivos específicos

1. Explorar las características del conjunto de datos.
2. Identificar y tratar valores faltantes.
3. Preparar las variables para el entrenamiento.
4. Entrenar tres modelos de Machine Learning.
5. Optimizar hiperparámetros y comparar resultados.
6. Evaluar los modelos mediante MAE, RMSE y R².
7. Seleccionar y justificar el modelo final.
8. Desarrollar una aplicación interactiva en Streamlit.

## Fuente de datos

**Fuente:** Datos Abiertos de Colombia - ICFES.

**Dataset:** Saber 11, periodo 2020-2.

**Enlace:** https://www.datos.gov.co/Education/Saber-11-2020-2/rnvb-vnyh

**Muestra utilizada:** 10.000 registros.

Se trabajó con los primeros 10.000 registros consultados. Por lo tanto, la muestra no necesariamente representa a todos los estudiantes colombianos.

## Variables utilizadas

Se seleccionaron ocho variables predictoras:

- Estrato socioeconómico.
- Educación de la madre.
- Educación del padre.
- Acceso a internet.
- Disponibilidad de computador.
- Naturaleza del colegio.
- Ubicación del colegio.
- Género del estudiante.

**Variable objetivo:** puntaje global de Saber 11 (`punt_global`).

## Metodología

1. Carga y exploración del dataset.
2. Identificación de valores faltantes.
3. Preparación de variables categóricas mediante imputación y One-Hot Encoding.
4. División de datos: 80 % entrenamiento y 20 % prueba.
5. Entrenamiento de tres modelos de regresión.
6. Optimización de hiperparámetros mediante RandomizedSearchCV.
7. Evaluación y comparación de resultados.
8. Exportación del mejor modelo mediante Joblib.
9. Integración con una aplicación Streamlit.

## Modelos de Machine Learning

Se entrenaron y evaluaron tres algoritmos:

1. Regresión lineal.
2. Árbol de decisión optimizado.
3. Random Forest optimizado.

### Resultados de evaluación

| Modelo | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest optimizado | 34.5366 | 43.0518 | 0.1762 |
| Regresión lineal | 34.4945 | 43.1097 | 0.1740 |
| Árbol de decisión optimizado | 35.2354 | 43.9658 | 0.1408 |

**Modelo seleccionado:** Random Forest optimizado, debido a que obtuvo el menor RMSE y el mayor R² en el conjunto de prueba.

Su ventaja sobre la regresión lineal fue pequeña y su capacidad predictiva es limitada.

## Tecnologías utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Streamlit
- Google Colab
- GitHub

## Estructura del repositorio

```text
Proyecto_db/
├── .streamlit/
├── data/
│   └── saber11_modelado.csv
├── models/
│   ├── README.md
│   ├── modelo_saber11.ipynb
│   └── modelo_saber11.joblib
├── notebook.ipynb
├── app.py
├── requirements.txt
└── README.md
```

## Aplicación Streamlit

Se desarrolló una aplicación web interactiva que permite:

- Consultar la descripción y los objetivos del proyecto.
- Explorar los resultados de las pruebas Saber 11 mediante gráficas.
- Visualizar una muestra del dataset.
- Ingresar ocho características de un estudiante.
- Obtener una estimación de su puntaje global mediante Random Forest.
- Comparar los resultados de los tres modelos mediante métricas y visualizaciones.

**Aplicación publicada:**

https://saber11-ciencia-datos-helen.streamlit.app/

## Ejecución local

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar la aplicación:

```bash
streamlit run app.py
```

Se requiere una versión de Python compatible con las dependencias del proyecto. El modelo guardado utiliza scikit-learn 1.6.1.

## Conclusiones

1. Se implementaron tres modelos de regresión para estimar el puntaje global de Saber 11.

2. Random Forest optimizado obtuvo el menor RMSE, aunque su rendimiento fue similar al de la regresión lineal.

3. El R² de 0.1762 indica que el modelo explica aproximadamente el 17.62 % de la variación observada en el puntaje global.

4. Las características seleccionadas aportan información predictiva, pero no explican la mayor parte de las diferencias entre estudiantes.

5. Los resultados representan asociaciones estadísticas y no permiten establecer relaciones causales.

6. La aplicación Streamlit permite demostrar el funcionamiento del modelo mediante predicciones interactivas.

## Estado del proyecto

**Proyecto desarrollado y aplicación funcional.**

Se completaron las etapas de exploración, preparación de datos, entrenamiento, optimización, evaluación, comparación e integración del modelo predictivo.
