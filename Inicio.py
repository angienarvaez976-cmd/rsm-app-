"""
Inicio.py
=========
Página principal de la aplicación de Inferencia Estadística en RSM
(Metodología de Superficie de Respuesta) de segundo orden.

Universidad Central del Ecuador - Facultad de Ciencias Económicas
Carrera de Estadística
"""

import numpy as np
import pandas as pd
import streamlit as st

from rsm.design import encode_value

st.set_page_config(
    page_title="RSM Segundo Orden | Inferencia Estadística",
    page_icon="📐",
    layout="wide",
)

# ----------------------------------------------------------------------
# Estado global de la sesión
# ----------------------------------------------------------------------
if "data" not in st.session_state:
    st.session_state["data"] = None
if "factor_cols" not in st.session_state:
    st.session_state["factor_cols"] = []
if "response_cols" not in st.session_state:
    st.session_state["response_cols"] = []
if "coding_limits" not in st.session_state:
    st.session_state["coding_limits"] = {}
if "already_coded" not in st.session_state:
    st.session_state["already_coded"] = False
if "models" not in st.session_state:
    st.session_state["models"] = {}
if "opt_results" not in st.session_state:
    st.session_state["opt_results"] = {}
if "desirability_result" not in st.session_state:
    st.session_state["desirability_result"] = None
if "dataset_name" not in st.session_state:
    st.session_state["dataset_name"] = None

st.title("📐 Aplicativo de RSM — Inferencia Estadística de Segundo Orden")
st.caption(
    "Diseño y análisis de Superficies de Respuesta (CCD / Box-Behnken) aplicado "
    "a un caso del sector agroindustrial ecuatoriano: optimización del secado de "
    "chips de plátano verde."
)

st.markdown(
    """
Este aplicativo integra, en un flujo guiado, los métodos de **Metodología de
Superficie de Respuesta (RSM)** revisados en clase, permitiendo a un usuario
no especialista **cargar datos, ejecutar el análisis y obtener recomendaciones
operativas concretas** para un proceso agroindustrial real.

### Navegación (panel izquierdo)
1. **📐 Diseño Experimental** — genera diseños Central Compuesto (CCD) y Box-Behnken (BBD).
2. **📊 Ajuste del Modelo** — carga tus datos, ajusta el modelo de 2do orden, ANOVA, falta de ajuste, residuos.
3. **🎯 Optimización** — ascenso más pronunciado, análisis canónico, análisis de cresta, optimización numérica.
4. **🧪 Múltiples Respuestas** — función de deseabilidad de Derringer-Suich.
5. **📈 Visualización** — contornos, superficies 3D, Pareto de efectos y perturbación.

---
"""
)

st.header("1. Cargar datos de trabajo")

col1, col2 = st.columns([2, 1])
with col2:
    usar_ejemplo = st.button("📂 Usar datos de ejemplo (chips de plátano)", use_container_width=True)
    st.caption(
        "Caso agroindustrial ecuatoriano: optimización de temperatura, tiempo de "
        "secado y espesor de rebanada de chips de plátano verde (*Musa "
        "paradisiaca*), midiendo humedad final, dureza (textura) y color (L*)."
    )

with col1:
    uploaded = st.file_uploader(
        "O sube tu propio archivo CSV (una columna por factor, una o más columnas de respuesta)",
        type=["csv"],
    )

if usar_ejemplo:
    st.session_state["data"] = pd.read_csv("data/ejemplo_chips_platano.csv")
    st.session_state["factor_cols"] = ["Temperatura", "Tiempo", "Espesor"]
    st.session_state["response_cols"] = ["Humedad_final_pct", "Dureza_N", "Color_L"]
    st.session_state["already_coded"] = False
    st.session_state["coding_limits"] = {
        "Temperatura": (60.0, 80.0),
        "Tiempo": (90.0, 150.0),
        "Espesor": (1.0, 3.0),
    }
    st.session_state["models"] = {}
    st.session_state["opt_results"] = {}
    st.session_state["desirability_result"] = None
    st.session_state["dataset_name"] = "ejemplo_chips_platano.csv"
    st.success("Datos de ejemplo cargados: 17 corridas (Box-Behnken, k=3, 5 puntos centrales).")

elif uploaded is not None:
    st.session_state["data"] = pd.read_csv(uploaded)
    st.session_state["models"] = {}
    st.session_state["opt_results"] = {}
    st.session_state["desirability_result"] = None
    st.session_state["dataset_name"] = uploaded.name
    st.success(f"Archivo cargado: {uploaded.name} ({len(st.session_state['data'])} filas).")

data = st.session_state["data"]

if data is not None:
    st.subheader("Vista previa de los datos")
    st.dataframe(data, use_container_width=True, height=250)

    st.header("2. Configurar variables")
    all_cols = list(data.columns)

    c1, c2 = st.columns(2)
    with c1:
        factor_cols = st.multiselect(
            "Selecciona las columnas de FACTORES (variables independientes, X)",
            options=all_cols,
            default=[c for c in st.session_state["factor_cols"] if c in all_cols],
        )
    with c2:
        response_cols = st.multiselect(
            "Selecciona las columnas de RESPUESTA (variables dependientes, Y)",
            options=[c for c in all_cols if c not in factor_cols],
            default=[c for c in st.session_state["response_cols"] if c in all_cols],
        )

    st.session_state["factor_cols"] = factor_cols
    st.session_state["response_cols"] = response_cols

    if factor_cols:
        st.subheader("Codificación de factores")
        already_coded = st.checkbox(
            "Mis datos de factores ya están codificados en (-1, 0, +1, ±alpha)",
            value=st.session_state["already_coded"],
        )
        st.session_state["already_coded"] = already_coded

        if not already_coded:
            st.caption(
                "Indica el nivel BAJO (-1) y ALTO (+1) de cada factor en unidades "
                "naturales; el resto de valores se codificarán linealmente."
            )
            limits = {}
            cols = st.columns(min(3, len(factor_cols)))
            for idx, f in enumerate(factor_cols):
                with cols[idx % len(cols)]:
                    default_low, default_high = st.session_state["coding_limits"].get(
                        f, (float(data[f].min()), float(data[f].max()))
                    )
                    low = st.number_input(f"{f} — nivel bajo (-1)", value=float(default_low), key=f"low_{f}")
                    high = st.number_input(f"{f} — nivel alto (+1)", value=float(default_high), key=f"high_{f}")
                    limits[f] = (low, high)
            st.session_state["coding_limits"] = limits

        st.info(
            f"✅ Configuración lista: **{len(factor_cols)} factor(es)** — "
            f"{', '.join(factor_cols)} · **{len(response_cols)} respuesta(s)** — "
            f"{', '.join(response_cols) if response_cols else '(ninguna seleccionada)'}.\n\n"
            "Continúa en **📊 Ajuste del Modelo** en el panel izquierdo."
        )
else:
    st.warning("Carga un archivo CSV o usa el conjunto de datos de ejemplo para comenzar.")

st.divider()
with st.expander("ℹ️ Acerca de este aplicativo"):
    st.markdown(
        """
**Tema:** Inferencia estadística en RSM de segundo orden.

**Métodos integrados:**
- Diseños: Central Compuesto (CCD, rotable o centrado en caras) y Box-Behnken (BBD).
- Ajuste: modelos de 1er y 2do orden por mínimos cuadrados, ANOVA de regresión,
  prueba de falta de ajuste (usando puntos repetidos como error puro), R²,
  R² ajustado, R² predicho (PRESS), residuos estandarizados/estudentizados.
- Optimización: ascenso más pronunciado, análisis canónico (punto estacionario
  y eigenvalores), análisis de cresta (Hoerl), optimización numérica restringida.
- Múltiples respuestas: deseabilidad individual y compuesta de Derringer-Suich.
- Visualización: contornos, superficies 3D, Pareto de efectos y perturbación.

**Caso aplicado:** optimización del proceso de secado de chips de plátano
verde, un producto del sector agroindustrial ecuatoriano, considerando
tres factores de proceso (temperatura, tiempo de secado, espesor de
rebanada) y tres respuestas de calidad (humedad final, dureza/textura,
color).
        """
    )
