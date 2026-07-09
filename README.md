# 📐 Aplicativo de Inferencia Estadística en RSM (Segundo Orden)

Aplicativo interactivo desarrollado en **Python + Streamlit** que integra los
métodos de **Metodología de Superficie de Respuesta (RSM)** de segundo orden
revisados en clase, aplicado a un caso real del **sector agroindustrial
ecuatoriano**: la optimización del proceso de secado de **chips de plátano
verde** (*Musa paradisiaca*).

> Curso: Optimización  Universidad Central del Ecuador — Facultad de Ciencias Económicas — Carrera de Estadística
> **Tema:** Inferencia estadística en RSM de segundo orden

---

## 1. Objetivo

Permitir que un usuario **no especialista**:
1. Cargue sus propios datos (o use el conjunto de ejemplo incluido),
2. Ejecute el análisis estadístico completo de RSM de segundo orden, y
3. Obtenga **recomendaciones operativas concretas** (condiciones óptimas de
   proceso) sin necesidad de escribir código.

## 2. Métodos integrados

| Módulo | Métodos |
|---|---|
| **Diseño experimental** | Diseño Central Compuesto (CCD, rotable o centrado en caras), Diseño Box-Behnken (BBD, k = 3, 4, 5) |
| **Ajuste del modelo** | Modelos de 1er y 2do orden (mínimos cuadrados), tabla ANOVA de regresión, prueba de **falta de ajuste** (lack of fit) con error puro, R², R² ajustado, R² predicho (PRESS), residuos estandarizados/estudentizados, gráfico de probabilidad normal |
| **Optimización** | Ascenso/descenso más pronunciado, análisis canónico (punto estacionario, eigenvalores/eigenvectores, naturaleza del punto), análisis de cresta (Ridge Analysis, método de Lagrange de Hoerl), optimización numérica restringida a la región experimental |
| **Múltiples respuestas** | Función de deseabilidad de **Derringer-Suich** (individual y compuesta, con pesos de importancia) |
| **Visualización** | Gráficos de contorno, superficies de respuesta 3D, diagrama de **Pareto de efectos estandarizados**, gráfico de **perturbación** |

## 3. Caso aplicado

**Optimización del secado de chips de plátano verde**

- **Factores (X):** Temperatura de secado (60–80 °C), Tiempo de secado
  (90–150 min), Espesor de rebanada (1–3 mm).
- **Respuestas (Y):** Humedad final (%, menor es mejor), Dureza/textura
  (N, valor objetivo), Color L\* — luminosidad (mayor es mejor, evita
  pardeamiento excesivo).
- **Diseño:** Box-Behnken con k=3 factores, 17 corridas (12 puntos
  factoriales + 5 puntos centrales replicados), incluido en
  `data/ejemplo_chips_platano.csv`.

Puedes reemplazar este conjunto por tus propios datos (CSV) desde la
página de **Inicio** de la aplicación.

## 4. Estructura del repositorio

```
rsm-app/
├── Inicio.py                          # Página principal (carga de datos)
├── pages/
│   ├── 1_📐_Diseño_Experimental.py    # Generador de diseños CCD / BBD
│   ├── 2_📊_Ajuste_del_Modelo.py      # Ajuste, ANOVA, falta de ajuste, residuos
│   ├── 3_🎯_Optimización.py           # Ascenso, canónico, cresta, numérico
│   ├── 4_🧪_Multiples_Respuestas.py   # Deseabilidad de Derringer-Suich
│   └── 5_📈_Visualización.py          # Contornos, superficies, Pareto, perturbación
├── rsm/                                # Motor estadístico (sin dependencia de Streamlit)
│   ├── design.py                      # Generación y codificación/decodificación de diseños
│   ├── model.py                       # Ajuste MCO, ANOVA, falta de ajuste, PRESS
│   ├── optimization.py                 # Ascenso, canónico, cresta, optimización numérica
│   ├── desirability.py                 # Derringer-Suich
│   ├── viz.py                          # Gráficos Plotly
│   └── app_utils.py                    # Utilidades compartidas de la app
├── data/
│   └── ejemplo_chips_platano.csv       # Datos de prueba (agroindustria ecuatoriana)
├── requirements.txt
├── .streamlit/config.toml              # Tema visual
└── README.md
```

El módulo `rsm/` es **independiente de Streamlit**: toda la estadística
(diseños, ajuste, ANOVA, optimización, deseabilidad) puede reutilizarse o
probarse por separado (por ejemplo, en un notebook de Jupyter) sin depender
de la interfaz.

## 5. Ejecución local

```bash
git clone https://github.com/<tu-usuario>/rsm-app.git
cd rsm-app
python3 -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run Inicio.py
```

La aplicación abrirá en `http://localhost:8501`.

## 6. Despliegue en la nube (Streamlit Community Cloud)

1. Sube este repositorio a GitHub (ver sección 7).
2. Entra a [share.streamlit.io](https://share.streamlit.io) con tu cuenta de GitHub.
3. Clic en **"New app"** → selecciona el repositorio, la rama (`main`) y el
   archivo principal `Inicio.py`.
4. Clic en **"Deploy"**. Streamlit instalará automáticamente las
   dependencias de `requirements.txt`.

También es compatible con **Hugging Face Spaces** (SDK: Streamlit) subiendo
los mismos archivos.

## 7. Cómo subir este proyecto a GitHub

```bash
cd rsm-app
git init
git add .
git commit -m "Aplicativo RSM segundo orden - proyecto de inferencia estadística"
git branch -M main
git remote add origin https://github.com/<tu-usuario>/rsm-app.git
git push -u origin main
```

## 8. Guía rápida de uso

1. **Inicio:** carga tus datos o usa el ejemplo → selecciona columnas de
   factores y respuesta(s) → define si tus datos ya están codificados o
   ingresa los niveles bajo/alto de cada factor.
2. **Diseño Experimental:** (opcional) genera y descarga una nueva matriz
   de diseño CCD o BBD para una futura corrida experimental.
3. **Ajuste del Modelo:** elige la respuesta y el orden del modelo →
   revisa ANOVA, falta de ajuste, coeficientes y residuos.
4. **Optimización:** con el modelo ajustado, explora ascenso más
   pronunciado, análisis canónico, análisis de cresta y optimización
   numérica.
5. **Múltiples Respuestas:** define especificaciones de deseabilidad
   (menor/mayor es mejor o valor objetivo) para 2 o más respuestas y
   obtén las condiciones que maximizan la deseabilidad compuesta D.
6. **Visualización:** genera contornos, superficies 3D, diagrama de
   Pareto de efectos y gráfico de perturbación para comunicar resultados.

## 9. Notas técnicas

- El motor de regresión usa mínimos cuadrados con `numpy.linalg.lstsq` y
  calcula errores estándar, estadísticos t y ANOVA manualmente (sin
  dependencias pesadas como `statsmodels`), lo que facilita el despliegue.
- La prueba de **falta de ajuste** requiere puntos de diseño repetidos
  (p. ej., los puntos centrales); si no existen réplicas, la app lo indica
  claramente y omite la prueba.
- Todas las optimizaciones (ascenso, cresta, numérica) trabajan en
  **unidades codificadas** y se apoyan en los límites bajo/alto ingresados
  para traducir los resultados a unidades naturales del proceso.
- El diseño Box-Behnken generalizado (k = 4, 5) se construye combinando el
  factorial 2² de cada par de factores; para k = 3 coincide exactamente
  con la tabla estándar de Box & Behnken (1960).

## 10. Referencias

- Box, G. E. P., & Behnken, D. W. (1960). *Some new three level designs
  for the study of quantitative variables*. Technometrics, 2(4), 455–475.
- Myers, R. H., Montgomery, D. C., & Anderson-Cook, C. M. (2016).
  *Response Surface Methodology: Process and Product Optimization Using
  Designed Experiments* (4th ed.). Wiley.
- Derringer, G., & Suich, R. (1980). *Simultaneous optimization of
  several response variables*. Journal of Quality Technology, 12(4), 214–219.
- Montgomery, D. C. (2019). *Design and Analysis of Experiments* (10th ed.). Wiley.

## 11. Autoría

Proyecto desarrollado para el curso de Inferencia Estadística — RSM de
segundo orden (ES10-001), Carrera de Estadística, Facultad de Ciencias
Económicas, Universidad Central del Ecuador.
