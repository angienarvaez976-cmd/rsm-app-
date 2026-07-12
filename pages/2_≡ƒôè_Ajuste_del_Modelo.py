import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from rsm.model import RSMModel
from rsm.app_utils import get_coded_X, check_ready

st.set_page_config(page_title="Ajuste del Modelo", page_icon="📊", layout="wide")
st.title("📊 Ajuste del Modelo de Superficie de Respuesta")

data, factor_cols, response_cols = check_ready(st)

col1, col2 = st.columns(2)
with col1:
    response = st.selectbox("Variable de respuesta a modelar", response_cols)
with col2:
    order = st.radio("Orden del modelo", ["second", "first"], horizontal=True,
                      format_func=lambda x: "Segundo orden completo (recomendado)" if x == "second" else "Primer orden")

X = get_coded_X(data, factor_cols, st.session_state["already_coded"], st.session_state["coding_limits"])
y = data[response].values.astype(float)

k_ = len(factor_cols)
p_expected = 1 + 2 * k_ + k_ * (k_ - 1) // 2 if order == "second" else 1 + k_
if len(y) < p_expected:
    st.error(
        f"No hay suficientes corridas ({len(y)}) para ajustar el modelo elegido, que requiere "
        f"al menos {p_expected} parámetros. Usa un diseño con más corridas o el modelo de primer orden."
    )
    st.stop()

model = RSMModel(X, y, order=order, factor_names=factor_cols)
st.session_state["models"][response] = model
st.session_state["models"][f"{response}__order"] = order

st.success(
    f"Modelo ajustado para **{response}** ({'2do' if order=='second' else '1er'} orden) "
    f"con {model.n} observaciones y {model.p} parámetros."
)

# ------------------------------------------------------------------
st.header("1. Bondad de ajuste")
c1, c2, c3, c4 = st.columns(4)
c1.metric("R²", f"{model.R2:.4f}")
c2.metric("R² ajustado", f"{model.R2_adj:.4f}")
c3.metric("R² predicho (PRESS)", f"{model.R2_pred:.4f}" if not np.isnan(model.R2_pred) else "N/D")
c4.metric("F0 (regresión)", f"{model.F0:.2f}" if not np.isnan(model.F0) else "N/D")

if model.R2_adj - model.R2_pred > 0.20 and not np.isnan(model.R2_pred):
    st.warning(
        "⚠️ La diferencia entre R² ajustado y R² predicho es mayor a 0.20. Esto puede "
        "indicar sobreajuste o puntos con influencia excesiva; revisa los residuos."
    )

# ------------------------------------------------------------------
st.header("2. Tabla ANOVA de regresión")
anova = model.anova_table()
st.dataframe(
    anova.style.format({"SC": "{:.4f}", "gl": "{:.0f}", "CM": "{:.4f}", "F0": "{:.3f}", "p-valor": "{:.4g}"}),
    use_container_width=True,
)

alpha = st.slider("Nivel de significancia (α) para interpretación", 0.01, 0.15, 0.05, 0.01)
if model.p_value_reg < alpha:
    st.success(
        f"El modelo de regresión es **significativo** (p = {model.p_value_reg:.4g} < α = {alpha})."
    )
else:
    st.error(
        f"El modelo de regresión **no** resulta significativo al nivel α = {alpha} "
        f"(p = {model.p_value_reg:.4g})."
    )

lof = model.lack_of_fit()
if lof["available"]:
    if lof["p_value"] < alpha:
        st.error(
            f"⚠️ **Falta de ajuste significativa** (p = {lof['p_value']:.4g} < α = {alpha}): "
            "el modelo actual podría no capturar adecuadamente la curvatura de la respuesta. "
            "Considera agregar términos o revisar el diseño."
        )
    else:
        st.success(
            f"✅ **No hay evidencia de falta de ajuste** (p = {lof['p_value']:.4g} ≥ α = {alpha}): "
            "el modelo describe adecuadamente la relación entre los factores y la respuesta."
        )
else:
    st.info(f"Prueba de falta de ajuste no disponible: {lof['reason']}")

# ------------------------------------------------------------------
st.header("3. Coeficientes del modelo")
coef = model.coef_table()
st.dataframe(
    coef.style.format({"Coeficiente": "{:.4f}", "Error Std.": "{:.4f}", "t0": "{:.3f}", "p-valor": "{:.4g}"}),
    use_container_width=True,
)
st.caption(
    "Términos con p-valor < α son estadísticamente significativos y contribuyen a explicar la respuesta."
)

# ------------------------------------------------------------------
st.header("4. Análisis de residuos")
r1, r2 = st.columns(2)

with r1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=model.y_hat, y=model.resid_student, mode="markers",
                              marker=dict(color="#2E7D6E", size=9)))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_hline(y=2, line_dash="dot", line_color="crimson")
    fig.add_hline(y=-2, line_dash="dot", line_color="crimson")
    fig.update_layout(title="Residuos estudentizados vs. valores predichos",
                       xaxis_title="Valor predicho", yaxis_title="Residuo estudentizado",
                       template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

with r2:
    n = len(model.residuals)
    osm, osr = stats.probplot(model.resid_student, dist="norm", fit=False)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=osm, y=osr, mode="markers", marker=dict(color="#42A5F5", size=9)))
    line = np.linspace(min(osm), max(osm), 10)
    fig2.add_trace(go.Scatter(x=line, y=line, mode="lines", line=dict(color="gray", dash="dash"), showlegend=False))
    fig2.update_layout(title="Gráfico de probabilidad normal de los residuos",
                        xaxis_title="Cuantiles teóricos", yaxis_title="Residuo estudentizado",
                        template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

st.caption(
    "Se espera que los residuos se distribuyan aleatoriamente alrededor de cero (sin patrones) "
    "y sigan aproximadamente una línea recta en el gráfico de probabilidad normal, sin puntos "
    "fuera de ±2 (posibles atípicos)."
)

with st.expander("Ver tabla completa de residuos"):
    resid_df = pd.DataFrame({
        "Observado": model.y, "Predicho": model.y_hat,
        "Residuo": model.residuals, "Residuo estandarizado": model.resid_std,
        "Residuo estudentizado": model.resid_student, "Leverage (hii)": model.hii,
    })
    st.dataframe(resid_df, use_container_width=True)

st.info("➡️ Continúa en **🎯 Optimización** para encontrar las condiciones óptimas de proceso.")
