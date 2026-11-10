
# Dataset - Pruebas Saber 11

## Fuente de información

Los datos utilizados en este proyecto provienen del
Portal de Datos Abiertos de Colombia (datos.gov.co).

Dataset: Resultados de las pruebas Saber 11.

Fuente:
https://www.datos.gov.co/resource/rnvb-vnyh.csv

## Descripción

Se trabaja con una muestra de 10.000 registros
correspondientes al periodo 2024-2.

El dataset original contiene 81 variables relacionadas
con características de los estudiantes, sus familias,
instituciones educativas y resultados académicos.

## Exploración de datos

Se realizaron las siguientes actividades:

- Revisión de dimensiones y tipos de datos.
- Cálculo de estadísticas descriptivas.
- Identificación de valores faltantes.
- Identificación de registros duplicados.
- Análisis de la distribución del puntaje global.
- Comparación del puntaje promedio según estrato.
- Comparación del puntaje promedio según acceso a internet.

## Limpieza inicial

Resultados obtenidos:

- Registros originales: 10.000.
- Duplicados completos encontrados: 0.
- Puntajes globales fuera del rango 0 a 500: 0.
- Registros con puntaje global inferior a 100: 8.

Los ocho registros se conservaron y marcaron para revisión.

Los valores faltantes no fueron imputados en esta etapa.
Su tratamiento se realizará durante la preparación
de datos para el entrenamiento de los modelos.

## Archivo para modelado

Archivo: saber11_modelado.csv

Contiene 10.000 registros y 9 columnas:

- fami_estratovivienda
- fami_educacionmadre
- fami_educacionpadre
- fami_tieneinternet
- fami_tienecomputador
- cole_naturaleza
- cole_area_ubicacion
- estu_genero
- punt_global

La variable objetivo propuesta es punt_global.

Las demás variables son características predictoras
iniciales, sujetas a revisión durante el modelado.

## Reproducibilidad

El archivo notebook.ipynb, ubicado en la raíz del
repositorio, contiene el proceso de exploración
y limpieza inicial.
