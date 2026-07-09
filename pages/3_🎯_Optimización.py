import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from rsm.model import RSMModel
from rsm.optimization import steepest_ascent, canonical_analysis, ridge_analysis, numeric_optimize
from rsm.app_utils import get_coded_X, check_ready

st.set_page_config(page_title="Optimización", page_icon="🎯", layout="wide")
st.title("🎯 Optimización de la Superficie de Respuesta")

data, factor_cols, response_cols = check_ready(st)

fitted_responses = [r for r in response_cols if r in st.session_state["models"]]
if not fitted_responses:
    st.warning("Primero ajusta un modelo para al menos una respuesta en **📊 Ajuste del Modelo**.")
    st.stop()

response = st.selectbox("Variable de respuesta a optimizar", fitted_responses)
model: RSMModel = st.session_state["models"][response]
k = model.k

maximize = st.radio("Objetivo", ["Maximizar", "Minimizar"], horizontal=True) == "Maximizar"

tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Ascenso más pronunciado", "🧭 Análisis canónico", "💍 Análisis de cresta", "🔢 Optimización numérica"
])

# ------------------------------------------------------------------
with tab1:
    st.markdown(
        "Parte de un **modelo de primer orden** ajustado sobre los mismos datos, y se mueve "
        "en la dirección del gradiente (la de mayor incremento/decremento de la respuesta)."
    )
    X = model.X_raw
    model1 = RSMModel(X, model.y, order="first", factor_names=factor_cols)
    b0_1, b_1 = model1.beta[0], model1.beta[1:1 + k]

    st.write("**Coeficientes del modelo de primer orden:**")
    st.dataframe(pd.DataFrame({"Factor": ["Intercepto"] + factor_cols, "Coeficiente": model1.beta}),
                 use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        ref_factor = st.selectbox("Factor de referencia para el tamaño de paso", factor_cols)
    with c2:
        step_size = st.number_input("Tamaño de paso (unidades codificadas)", value=0.5, step=0.1)
    with c3:
        n_steps = st.number_input("Número de pasos", value=6, min_value=1, max_value=20)

    ref_idx = factor_cols.index(ref_factor)
    traj = steepest_ascent(b0_1, b_1, base_point=np.zeros(k), step_factor_index=ref_idx,
                            step_size=step_size, n_steps=int(n_steps), maximize=maximize)
    st.dataframe(traj.style.format({c: "{:.4f}" for c in traj.columns if c not in ("Paso",)}),
                 use_container_width=True)

    fig = go.Figure(go.Scatter(x=traj["Paso"], y=traj["y_predicho"], mode="lines+markers",
                                line=dict(color="#2E7D6E", width=3)))
    fig.update_layout(title=f"Trayectoria de {'ascenso' if maximize else 'descenso'} más pronunciado",
                       xaxis_title="Paso", yaxis_title=f"{response} predicho", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "⚠️ El modelo de primer orden es válido lejos de la curvatura; una vez que la "
        "respuesta deje de mejorar con los pasos, conviene correr un nuevo diseño de "
        "segundo orden (CCD/BBD) alrededor de esa región para el análisis canónico y de cresta."
    )

# ------------------------------------------------------------------
with tab2:
    if model.order != "second":
        st.warning("El análisis canónico requiere un modelo de **segundo orden**. Ajusta uno en la página anterior.")
    else:
        b0, b, B = model.get_quadratic_matrices()
        ca = canonical_analysis(b0, b, B)

        st.subheader("Punto estacionario (unidades codificadas)")
        st.dataframe(pd.DataFrame({"Factor": factor_cols, "Valor codificado": ca["stationary_point"]}),
                     use_container_width=True)
        st.metric(f"{response} en el punto estacionario", f"{ca['y_stationary']:.4f}")

        st.subheader("Naturaleza del punto estacionario")
        st.write(f"**{ca['nature']}**")
        eig_df = pd.DataFrame({
            "Eigenvalor": ca["eigenvalues"],
        })
        for i in range(k):
            eig_df[f"Eigenvector (eje {i+1})"] = ca["eigenvectors"][:, i]
        st.dataframe(eig_df, use_container_width=True)

        in_region = np.all(np.abs(ca["stationary_point"]) <= 1.6)
        if in_region:
            st.success("El punto estacionario está dentro (o cerca) de la región experimental explorada.")
        else:
            st.warning(
                "⚠️ El punto estacionario cae **fuera** de la región experimental (|x| > 1.6). "
                "Interprétalo con cautela: puede requerirse un nuevo diseño centrado en esa dirección, "
                "o usar el **análisis de cresta** para explorar de forma restringida."
            )

        if "Máximo" in ca["nature"] and not maximize:
            st.info("Nota: el punto estacionario corresponde a un máximo, pero seleccionaste 'Minimizar'.")
        if "Mínimo" in ca["nature"] and maximize:
            st.info("Nota: el punto estacionario corresponde a un mínimo, pero seleccionaste 'Maximizar'.")

# ------------------------------------------------------------------
with tab3:
    if model.order != "second":
        st.warning("El análisis de cresta requiere un modelo de **segundo orden**.")
    else:
        b0, b, B = model.get_quadratic_matrices()
        st.markdown(
            "Traza la mejor respuesta posible (máx. o mín.) a distintas **distancias (radios)** "
            "del centro del diseño, útil cuando el punto estacionario cae fuera de la región "
            "experimental o cuando se desea explorar de forma controlada."
        )
        r_max = st.slider("Radio máximo a explorar (unidades codificadas)", 0.5, 3.0, 1.5, 0.1)
        n_radii = st.slider("Número de radios", 3, 15, 6)
        radii = np.linspace(0.1, r_max, n_radii)

        ridge_df = ridge_analysis(b0, b, B, radii=radii, maximize=maximize)
        # Rename coded X columns to actual factor names for display
        rename_map = {f"X{i+1}": factor_cols[i] for i in range(k)}
        ridge_df_display = ridge_df.rename(columns=rename_map)
        st.dataframe(
            ridge_df_display[["radio_objetivo", "radio", "y_predicho"] + factor_cols].style.format(
                {c: "{:.4f}" for c in ridge_df_display.columns if c != "radio_objetivo"}
            ),
            use_container_width=True,
        )

        fig = go.Figure(go.Scatter(x=ridge_df_display["radio"], y=ridge_df_display["y_predicho"],
                                    mode="lines+markers", line=dict(color="#7E57C2", width=3)))
        fig.update_layout(title=f"Análisis de cresta — {'máximo' if maximize else 'mínimo'} restringido",
                           xaxis_title="Radio (distancia codificada al centro)",
                           yaxis_title=f"{response} predicho", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
with tab4:
    st.markdown(
        "Optimización numérica restringida a la región experimental codificada "
        "[-1, 1] en cada factor (evita extrapolar fuera del espacio explorado)."
    )
    bound = st.slider("Límite de la región de búsqueda (unidades codificadas)", 1.0, 2.0, 1.0, 0.1)

    def pred(x):
        return model.predict(x.reshape(1, -1))[0]

    result = numeric_optimize(pred, k=k, bounds=(-bound, bound), maximize=maximize)

    st.subheader("Condiciones óptimas (unidades codificadas)")
    st.dataframe(pd.DataFrame({"Factor": factor_cols, "Valor codificado óptimo": result["x_opt"]}),
                 use_container_width=True)
    st.metric(f"{response} óptimo estimado", f"{result['y_opt']:.4f}")

    if not st.session_state["already_coded"] and st.session_state["coding_limits"]:
        st.subheader("Condiciones óptimas (unidades naturales)")
        rows = []
        for i, f in enumerate(factor_cols):
            low, high = st.session_state["coding_limits"][f]
            center = (low + high) / 2
            half = (high - low) / 2
            natural_val = center + result["x_opt"][i] * half
            rows.append({"Factor": f, "Valor natural óptimo": natural_val})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if not result["success"]:
        st.warning(f"El optimizador no convergió del todo: {result['message']}")

st.info("➡️ Continúa en **🧪 Múltiples Respuestas** para optimizar varias respuestas a la vez (deseabilidad).")
