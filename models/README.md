# Modelos de Machine Learning - Saber 11

Esta carpeta contiene los archivos correspondientes al entrenamiento, evaluación y exportación del modelo de Machine Learning utilizado para predecir el puntaje global de las pruebas Saber 11 en Colombia.

## Archivos

- **modelo_saber11.ipynb:** Notebook que contiene el entrenamiento, optimización y comparación de tres modelos de regresión, además de la visualización de sus resultados.
- **modelo_saber11.joblib:** Modelo Random Forest optimizado, utilizado por la aplicación desarrollada en Streamlit.

## Modelos evaluados

Se implementaron tres algoritmos:

1. Regresión lineal.
2. Árbol de decisión optimizado.
3. Random Forest optimizado.

## Modelo seleccionado

Se seleccionó Random Forest optimizado por presentar el menor RMSE (43.0518) y el mayor R² (0.1762) en el conjunto de prueba.

Aunque el modelo permite identificar asociaciones entre las variables utilizadas y el puntaje global, su capacidad predictiva es limitada. Por lo tanto, los resultados deben interpretarse como un ejercicio académico y exploratorio.
