# effective-contact-prediction-banking

Modelo predictivo para estimar la probabilidad de lograr un **contacto telefónico efectivo** con clientes de un banco, a partir de su historial de gestiones comerciales, canal de contacto, ubicación geográfica y nivel de ingresos — con el fin de priorizar la asignación de recursos de televentas hacia los clientes con mayor probabilidad de éxito.

**Autor:** Stefano Giacomo Landa Morante — Universidad Nacional de Ingeniería (UNI)

## Datos

* **Tamaño:** 757,391 registros, 40 variables
* **Variable objetivo (`TARGET`):** contacto efectivo (1) vs. no efectivo (0) — con desbalance de clases (la clase positiva supera el doble de volumen que la negativa)
* **Variables predictoras:** cantidad y distribución de gestiones (`TOTGEST`, `NC\_CTD`, `CNE\_CTD`), días de actividad de la línea (`DIAS\_ACT`), feedback del último y mejor contacto (`FBK\_ULT`, `FBK\_BEST`), nivel de prioridad, segmento y rango de ingresos, ubicación (departamento/provincia)

> \*\*Nota sobre el dataset:\*\* el archivo original (`data\_selec\_entre.csv`) pesa más de lo que GitHub permite subir en un repositorio normal (límite de 100 MB por archivo) y contiene datos comerciales de un banco, por lo que \*\*no se incluye en este repositorio\*\*. El notebook documenta el proceso completo de limpieza, imputación y modelado sobre esa base.

## Preprocesamiento

1. **Limpieza de códigos nulos:** los valores `-999` se reemplazaron por nulos de Python.
2. **Eliminación de variables:** columnas con 90–96% de datos nulos se descartaron por baja utilidad para el modelado.
3. **Tratamiento de atípicos:** eliminación de outliers en variables como `DIAS\_ACT` y `TOTGEST`.
4. **Imputación segmentada:**

   * Variables de conteo de gestiones (`NC\_CTD`, `CNE\_CTD`) imputadas respetando que su suma no exceda el total de gestiones (`TOTGEST`)
   * Variables numéricas imputadas por mediana global o segmentada quando existían diferencias significativas entre grupos (ej. ingresos por segmento)
   * Variables categóricas (`RANGO\_INGRESOS`, `FBK\_ULT`, `FBK\_BEST`, `DEPARTAMENTO`, `SEGMENTO`, `COD\_SALA`) imputadas por moda
5. **Balanceo de clases:** aplicación de **SMOTE** sobre el conjunto de entrenamiento para corregir el desbalance de la variable objetivo.

## Análisis exploratorio — hallazgos clave

* Los segmentos de alto valor alcanzan tasas de contacto efectivo de 67–75%, frente a 62% en segmentos de bajo valor.
* A mayor nivel de ingresos, mayor probabilidad de contacto efectivo (66.5% en ingresos bajos → 73.5% en ingresos altos).
* El nivel de prioridad del canal es determinante: Nivel 1 alcanza \~80% de éxito, cayendo por debajo de 30% en los niveles 9–10.
* La frescura del feedback es crítica: a mayor cantidad de días desde el último mejor contacto, la tasa de éxito cae significativamente.
* Lima concentra más de 3.6 millones de contactos, muy por encima de otras regiones (Arequipa, Callao, La Libertad).

## Modelos evaluados

Se entrenaron y compararon 5 algoritmos de clasificación sobre los datos balanceados, evaluando su capacidad de discriminación (AUC-ROC) en un conjunto de test independiente:

|Modelo|AUC (Test)|
|-|-|
|**Red Neuronal (MLP)**|**0.8117**|
|Random Forest|0.8047|
|Bagging (con árbol de decisión)|0.7960|
|Árbol de Decisión|0.7928|
|Regresión Logística|0.7724|

**Configuración de los modelos:**

* Red Neuronal: MLPClassifier, capas ocultas (16, 8), activación ReLU, optimizador Adam, early stopping
* Random Forest: 100 árboles, profundidad máxima 10, `max\_features='sqrt'`
* Árbol de Decisión: profundidad máxima 6, `min\_samples\_leaf=300`
* Bagging: 100 estimadores sobre árboles de decisión

## Conclusión

La **Red Neuronal** obtuvo el mejor desempeño (AUC 0.81), seguida de **Random Forest** (AUC 0.80), superando ampliamente a la Regresión Logística. En todos los modelos, el desempeño en test fue ligeramente superior al de entrenamiento (ej. RN: 0.79 → 0.81), lo que descarta sobreajuste y confirma que la limpieza de datos, las imputaciones segmentadas y el balanceo de clases permitieron que los modelos aprendieran patrones reales del negocio en lugar de ruido.

## Estructura del repositorio

```
├── Predicción_script.ipynb   # Notebook con limpieza, EDA, balanceo y modelado
└── README.md              # El dataset original no se incluye (ver nota en "Datos")
```

## Cómo ejecutarlo

```bash
git clone https://github.com/Stefanolm/effective-contact-prediction-banking.git
cd \[nombre-del-repo]
pip install -r requirements.txt   # pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, seaborn
jupyter notebook Predicción_script.ipynb
```

## Herramientas utilizadas

Python — pandas, scikit-learn (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, BaggingClassifier, MLPClassifier), imbalanced-learn (SMOTE), seaborn, matplotlib.
