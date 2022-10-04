# Proyecto demanda y produccion electrica

![image](https://user-images.githubusercontent.com/99116361/193780617-203bc4a8-f31f-4650-a12f-d3b64b5b091e.png)

## 1.Objetivo del proyecto
Este proyecto tiene principalmente dos objetivos:
  - El primero consiste en estudiar y predecir la demanda eléctrica en un intervalo de tiempo.
  - El segundo objetivo es estudiar y analizar los métodos y tecnologías de producción electrica.
## 2.Extracción de datos
La información se ha extraído de la API de Red Eléctrica Española. Los datos de producción eléctrica consisten en registros por día desde 2011 hasta 2022. Los datos de demanda eléctrica consisten en registros por cada hora (00:00-23:00) desde 2014 hasta 2022.

Fuente: https://www.ree.es/es/apidatos

## 3.Modelos ML
Para realizar la predicción de la demanda eléctrica se han usado los siguientes modelo:
  - XGBoost Regressor
  - Random Forest
  
## 4.Recursos
Para realizar el proycto he usado:
  - Jupyter notebooks
  - Python scripts
  - Librearias usadas: Pandas, Numpy, Matplotlib, Seaborn, sklearn, entre otras.
