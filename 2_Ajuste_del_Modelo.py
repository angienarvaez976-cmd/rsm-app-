import numpy as np
import pandas as pd
import streamlit as st

from rsm.app_utils import check_ready
from rsm.report import build_pdf_report

st.set_page_config(page_title="Reporte Final", page_icon="📄", layout="wide")
st.title("📄 Reporte Final del Estudio RSM")
st.markdown(
    "Genera y descarga un **informe en PDF** que resume el estudio realizado: "
    "configuración de factores, ajuste del modelo, ANOVA, falta de ajuste, "
    "optimización y (si aplica) la deseabilidad compuesta de Derringer-Suich."
)

data, factor_cols, response_cols = check_ready(st, require_response=False)
fitted_responses = [r for r in response_cols if r in st.session_state["models"]]

if not fitted_responses:
    st.warning(
        "Aún no has ajustado ningún modelo. Ve primero a **📊 Ajuste del Modelo** "
        "para al menos una respuesta antes de generar el reporte."
    )
    st.stop()

st.subheader("Contenido a incluir")
c1, c2 = st.columns(2)
with c1:
    autor = st.text_input("Elaborado por (opcional)", value="")
with c2:
    incluidas = st.multiselect(
        "Respuestas a incluir en el reporte",
        fitted_responses, default=fitted_responses,
    )

incluir_opt = st.checkbox("Incluir resultados de la página de Optimización (si ya los generaste)", value=True)
incluir_deseab = st.checkbox("Incluir resultados de Múltiples Respuestas / Deseabilidad (si ya los generaste)", value=True)

if st.button("🖨️ Generar informe en PDF", type="primary"):
    # -------- Configuración de factores --------
    factor_info = []
    for f in factor_cols:
        if st.session_state["already_coded"]:
            factor_info.append({"nombre": f, "bajo": -1.0, "alto": 1.0})
        else:
            low, high = st.session_state["coding_limits"].get(f, (np.nan, np.nan))
            factor_info.append({"nombre": f, "bajo": low, "alto": high})

    # -------- Secciones por respuesta --------
    response_sections = []
    for resp in incluidas:
        model = st.session_state["models"][resp]
        lof = model.lack_of_fit()
        response_sections.append({
            "nombre": resp,
            "order": model.order,
            "R2": model.R2,
            "R2_adj": model.R2_adj,
            "R2_pred": model.R2_pred if not np.isnan(model.R2_pred) else None,
            "anova_df": model.anova_table(),
            "coef_df": model.coef_table(),
            "lof_available": lof["available"],
            "lof_p": lof.get("p_value"),
            "lof_ok": (lof.get("p_value", 1.0) >= 0.05) if lof["available"] else None,
        })

    # -------- Optimización --------
    opt_sections = []
    if incluir_opt:
        opt_results = st.session_state.get("opt_results", {})
        for resp in incluidas:
            if resp in opt_results:
                r = opt_results[resp]
                opt_sections.append({
                    "response": resp,
                    "objetivo": r.get("objetivo", "N/D"),
                    "canonical": r.get("canonical"),
                    "numeric": r.get("numeric"),
                })

    # -------- Deseabilidad --------
    desirability_section = None
    if incluir_deseab and st.session_state.get("desirability_result"):
        desirability_section = st.session_state["desirability_result"]

    meta = {
        "titulo": "Informe de Análisis RSM de Segundo Orden",
        "autor": autor,
        "dataset_name": st.session_state.get("dataset_name", "Datos cargados por el usuario"),
        "n_obs": len(data),
    }

    pdf_bytes = build_pdf_report(meta, factor_info, response_sections, opt_sections, desirability_section)

    st.success("✅ Informe generado correctamente.")
    st.download_button(
        "⬇️ Descargar informe (PDF)",
        data=pdf_bytes,
        file_name="informe_rsm_segundo_orden.pdf",
        mime="application/pdf",
    )

st.divider()
with st.expander("ℹ️ Recomendaciones antes de generar el reporte"):
    st.markdown(
        """
- Para que la sección de **Optimización** aparezca en el reporte, visita primero
  la página **🎯 Optimización** para cada respuesta que quieras incluir (las
  pestañas de análisis canónico y optimización numérica guardan sus resultados
  automáticamente).
- Para que aparezca la sección de **Deseabilidad**, visita primero la página
  **🧪 Múltiples Respuestas** con al menos 2 respuestas seleccionadas.
- El informe **no incluye los gráficos interactivos** (contornos, superficies
  3D, Pareto, perturbación) para mantener el despliegue ligero y confiable;
  esos gráficos siguen disponibles dentro de la app, en **📈 Visualización**,
  y puedes capturarlos de pantalla si los necesitas en el documento.
        """
    )
