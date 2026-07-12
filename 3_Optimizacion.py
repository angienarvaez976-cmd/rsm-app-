import numpy as np
import streamlit as st

from rsm.viz import contour_plot, surface_plot, pareto_effects_plot, perturbation_plot
from rsm.app_utils import check_ready

st.set_page_config(page_title="Visualización", page_icon="📈", layout="wide")
st.title("📈 Visualización de la Superficie de Respuesta")

data, factor_cols, response_cols = check_ready(st)
fitted_responses = [r for r in response_cols if r in st.session_state["models"]]

if not fitted_responses:
    st.warning("Primero ajusta un modelo en **📊 Ajuste del Modelo**.")
    st.stop()

response = st.selectbox("Variable de respuesta", fitted_responses)
model = st.session_state["models"][response]
k = model.k

tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Contorno", "🏔️ Superficie 3D", "📊 Pareto de efectos", "🎛️ Perturbación"])

# Selección de factores fijos para contorno/superficie
if k > 2:
    st.sidebar.subheader("Factores fijos (para contorno / superficie)")

fixed_values = np.zeros(k)
if k > 2:
    for i, f in enumerate(factor_cols):
        fixed_values[i] = st.sidebar.slider(f"{f} (fijo, codificado)", -1.5, 1.5, 0.0, 0.1, key=f"fix_{f}")

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        fi = st.selectbox("Factor eje X", factor_cols, index=0, key="c_fi")
    with c2:
        fj = st.selectbox("Factor eje Y", factor_cols, index=1 if k > 1 else 0, key="c_fj")
    if fi == fj:
        st.warning("Selecciona dos factores distintos.")
    else:
        fig = contour_plot(model, factor_cols.index(fi), factor_cols.index(fj),
                            fixed_values, factor_cols, response_name=response)
        st.plotly_chart(fig, use_container_width=True)
        if k > 2:
            st.caption("Los factores no graficados se mantienen en los valores fijos definidos en el panel lateral.")

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        fi3 = st.selectbox("Factor eje X", factor_cols, index=0, key="s_fi")
    with c2:
        fj3 = st.selectbox("Factor eje Y", factor_cols, index=1 if k > 1 else 0, key="s_fj")
    if fi3 == fj3:
        st.warning("Selecciona dos factores distintos.")
    else:
        fig = surface_plot(model, factor_cols.index(fi3), factor_cols.index(fj3),
                            fixed_values, factor_cols, response_name=response)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    if model.order != "second":
        st.info("El diagrama de Pareto se muestra para todos los términos del modelo ajustado (orden actual).")
    alpha_p = st.slider("Nivel de significancia (α)", 0.01, 0.15, 0.05, 0.01, key="pareto_alpha")
    fig = pareto_effects_plot(model, alpha=alpha_p)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Barras verdes: efectos estadísticamente significativos (|t0| ≥ t crítico). "
        "Barras grises: no significativos al nivel α seleccionado."
    )

with tab4:
    st.markdown(
        "Muestra cómo cambia la respuesta al variar **cada factor individualmente** desde "
        "-1 a +1, manteniendo los demás factores en su punto central (0)."
    )
    fig = perturbation_plot(model, factor_cols, response_name=response)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Líneas con mayor pendiente indican factores con mayor efecto local sobre la respuesta "
        "cerca del centro del diseño."
    )
