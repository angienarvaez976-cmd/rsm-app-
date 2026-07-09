import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from rsm.model import RSMModel
from rsm.optimization import numeric_optimize
from rsm.desirability import (
    desirability_larger_is_better, desirability_smaller_is_better,
    desirability_target_is_best, composite_desirability,
)
from rsm.app_utils import check_ready

st.set_page_config(page_title="Múltiples Respuestas", page_icon="🧪", layout="wide")
st.title("🧪 Optimización Simultánea de Múltiples Respuestas")
st.markdown(
    "Función de **deseabilidad de Derringer-Suich (1980)** para combinar varias "
    "respuestas de calidad en un único índice compuesto D ∈ [0, 1] y encontrar "
    "las condiciones de proceso que mejor las satisfacen simultáneamente."
)

data, factor_cols, response_cols = check_ready(st)
fitted_responses = [r for r in response_cols if r in st.session_state["models"]]

if len(fitted_responses) < 2:
    st.warning(
        "Ajusta modelos (en **📊 Ajuste del Modelo**) para **al menos 2 respuestas** "
        "antes de usar esta página."
    )
    st.stop()

selected = st.multiselect("Respuestas a incluir en la deseabilidad compuesta",
                           fitted_responses, default=fitted_responses)
if len(selected) < 2:
    st.stop()

st.header("1. Especificaciones de deseabilidad por respuesta")

specs = {}
for resp in selected:
    model = st.session_state["models"][resp]
    y_min, y_max = float(model.y.min()), float(model.y.max())
    st.subheader(resp)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        tipo = st.selectbox("Criterio", ["Menor es mejor", "Mayor es mejor", "Valor objetivo (target)"],
                             key=f"tipo_{resp}")
    with c2:
        low = st.number_input("Límite inferior", value=round(y_min, 3), key=f"low_{resp}")
    with c3:
        if tipo == "Valor objetivo (target)":
            target = st.number_input("Valor objetivo", value=round((y_min + y_max) / 2, 3), key=f"target_{resp}")
        else:
            target = None
    with c4:
        high = st.number_input("Límite superior", value=round(y_max, 3), key=f"high_{resp}")
    with c5:
        weight = st.number_input("Peso (importancia)", value=1.0, min_value=0.1, max_value=5.0, step=0.1, key=f"w_{resp}")
    specs[resp] = {"tipo": tipo, "low": low, "high": high, "target": target, "weight": weight}

st.header("2. Optimización numérica de la deseabilidad compuesta")

k = st.session_state["models"][selected[0]].k
bound = st.slider("Límite de la región de búsqueda (unidades codificadas)", 1.0, 2.0, 1.0, 0.1)


def compute_d_for_point(x: np.ndarray) -> tuple[float, dict]:
    d_vals = {}
    for resp in selected:
        model = st.session_state["models"][resp]
        y_hat = model.predict(x.reshape(1, -1))[0]
        s = specs[resp]
        if s["tipo"] == "Mayor es mejor":
            d = desirability_larger_is_better(np.array([y_hat]), s["low"], s["high"])[0]
        elif s["tipo"] == "Menor es mejor":
            d = desirability_smaller_is_better(np.array([y_hat]), s["low"], s["high"])[0]
        else:
            d = desirability_target_is_best(np.array([y_hat]), s["low"], s["target"], s["high"])[0]
        d_vals[resp] = (y_hat, d)
    D = composite_desirability(
        [np.array([d_vals[r][1]]) for r in selected],
        weights=[specs[r]["weight"] for r in selected],
    )[0]
    return D, d_vals


def neg_D(x):
    D, _ = compute_d_for_point(x)
    return D


result = numeric_optimize(lambda x: neg_D(x), k=k, bounds=(-bound, bound), maximize=True)
D_opt, d_detail = compute_d_for_point(result["x_opt"])

st.subheader("Condiciones óptimas encontradas")
cols = st.columns(k)
factor_names = st.session_state["factor_cols"]
for i, f in enumerate(factor_names):
    with cols[i]:
        st.metric(f"{f} (codificado)", f"{result['x_opt'][i]:.3f}")
        if not st.session_state["already_coded"]:
            low_l, high_l = st.session_state["coding_limits"][f]
            center, half = (low_l + high_l) / 2, (high_l - low_l) / 2
            natural = center + result["x_opt"][i] * half
            st.caption(f"≈ {natural:.2f} (unidades naturales)")

st.metric("🎯 Deseabilidad compuesta (D)", f"{D_opt:.4f}")

st.subheader("Valores predichos y deseabilidad individual en el óptimo")
detail_rows = []
for resp in selected:
    y_hat, d = d_detail[resp]
    detail_rows.append({"Respuesta": resp, "Valor predicho": y_hat, "Deseabilidad individual (d)": d,
                         "Peso": specs[resp]["weight"]})
st.dataframe(pd.DataFrame(detail_rows).style.format(
    {"Valor predicho": "{:.3f}", "Deseabilidad individual (d)": "{:.3f}", "Peso": "{:.1f}"}
), use_container_width=True)

if D_opt < 0.4:
    st.warning(
        "⚠️ La deseabilidad compuesta óptima es baja (D < 0.4). Esto sugiere que las "
        "especificaciones son difíciles de satisfacer simultáneamente en la región "
        "experimental; considera relajar límites, ampliar la región de búsqueda, o "
        "priorizar (aumentar peso) la(s) respuesta(s) más crítica(s)."
    )
elif D_opt < 0.7:
    st.info("La deseabilidad compuesta es moderada (0.4 ≤ D < 0.7): un buen compromiso, pero con margen de mejora.")
else:
    st.success("✅ Deseabilidad compuesta alta (D ≥ 0.7): las respuestas se satisfacen bien de forma simultánea.")

st.header("3. Mapa de deseabilidad compuesta (2 factores)")
if k >= 2:
    c1, c2 = st.columns(2)
    with c1:
        fi = st.selectbox("Factor eje X", factor_names, index=0)
    with c2:
        fj = st.selectbox("Factor eje Y", factor_names, index=1 if k > 1 else 0)
    fi_idx, fj_idx = factor_names.index(fi), factor_names.index(fj)

    n_grid = 40
    grid = np.linspace(-bound, bound, n_grid)
    Z = np.zeros((n_grid, n_grid))
    base = result["x_opt"].copy()
    for a in range(n_grid):
        for b in range(n_grid):
            x = base.copy()
            x[fi_idx] = grid[a]
            x[fj_idx] = grid[b]
            D_val, _ = compute_d_for_point(x)
            Z[b, a] = D_val

    fig = go.Figure(go.Contour(x=grid, y=grid, z=Z, colorscale="Tealrose",
                                contours=dict(showlabels=True)))
    fig.add_trace(go.Scatter(x=[result["x_opt"][fi_idx]], y=[result["x_opt"][fj_idx]],
                              mode="markers", marker=dict(color="black", size=12, symbol="star"),
                              name="Óptimo"))
    fig.update_layout(title=f"Deseabilidad compuesta D vs. {fi} y {fj} (resto en el óptimo)",
                       xaxis_title=f"{fi} (codificado)", yaxis_title=f"{fj} (codificado)",
                       template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

with st.expander("📖 Sobre la función de deseabilidad"):
    st.markdown(
        r"""
La deseabilidad individual $d_i$ transforma cada respuesta predicha $\hat{y}_i$ en una
escala de 0 (totalmente indeseable) a 1 (ideal):

- **Mayor es mejor:** $d=0$ si $\hat y \le L$; crece hasta $d=1$ en $\hat y \ge T$ (objetivo/tope superior).
- **Menor es mejor:** $d=1$ si $\hat y \le T$ (objetivo/tope inferior); decrece hasta $d=0$ en $\hat y \ge U$.
- **Valor objetivo:** $d=1$ exactamente en el valor objetivo $T$, y decrece hacia 0 en los límites $L$ y $U$.

La **deseabilidad compuesta** es la media geométrica ponderada:

$$D = \left(\prod_{i=1}^n d_i^{w_i}\right)^{1/\sum w_i}$$

Si **cualquier** respuesta tiene $d_i = 0$ (fuera de especificación), $D = 0$: una sola
respuesta inaceptable invalida la combinación completa, reflejando que todas las
respuestas deben satisfacerse simultáneamente.
        """
    )

st.info("➡️ Continúa en **📈 Visualización** para explorar contornos, superficies 3D, Pareto y perturbación.")
