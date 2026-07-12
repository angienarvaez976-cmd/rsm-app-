import numpy as np
import pandas as pd
import streamlit as st

from rsm.design import ccd_design, bbd_design, decode_design

st.set_page_config(page_title="Diseño Experimental", page_icon="📐", layout="wide")
st.title("📐 Generación de Diseños Experimentales")
st.markdown(
    "Genera un **Diseño Central Compuesto (CCD)** o un **Diseño Box-Behnken (BBD)** "
    "en unidades codificadas, y decodifícalo a unidades naturales para tu proceso."
)

tipo = st.radio("Tipo de diseño", ["Central Compuesto (CCD)", "Box-Behnken (BBD)"], horizontal=True)
k = st.slider("Número de factores (k)", min_value=2, max_value=5, value=3)

nombres_default = [f"X{i+1}" for i in range(k)]
nombres_str = st.text_input(
    "Nombres de los factores (separados por coma)", value=", ".join(nombres_default)
)
factor_names = [n.strip() for n in nombres_str.split(",")][:k]
while len(factor_names) < k:
    factor_names.append(f"X{len(factor_names)+1}")

col1, col2 = st.columns(2)

if tipo == "Central Compuesto (CCD)":
    with col1:
        alpha_type = st.selectbox(
            "Tipo de alpha (puntos axiales)",
            ["rotatable", "face"],
            format_func=lambda x: "Rotable — alpha = (2^k)^(1/4)" if x == "rotatable" else "Centrado en las caras — alpha = 1",
        )
    with col2:
        n_center = st.number_input("Número de puntos centrales", min_value=1, max_value=10, value=5)
    df = ccd_design(k, alpha_type=alpha_type, n_center=int(n_center), factor_names=factor_names)
    st.info(f"Diseño CCD generado: **{len(df)} corridas** · alpha = **{df.attrs['alpha']:.4f}**")
else:
    if k > 5:
        st.error("El BBD está implementado para k = 3, 4 o 5 factores.")
        st.stop()
    if k < 3:
        st.error("El Box-Behnken requiere al menos 3 factores.")
        st.stop()
    with col2:
        n_center = st.number_input("Número de puntos centrales", min_value=1, max_value=10, value=3)
    df = bbd_design(k, n_center=int(n_center), factor_names=factor_names)
    st.info(f"Diseño Box-Behnken generado: **{len(df)} corridas**")

st.subheader("Matriz de diseño (unidades codificadas)")
st.dataframe(df, use_container_width=True)

st.subheader("Decodificar a unidades naturales (opcional)")
st.caption("Ingresa el nivel bajo (-1) y alto (+1) de cada factor en las unidades de tu proceso.")

decode = st.checkbox("Decodificar diseño a unidades naturales")
if decode:
    low, high = {}, {}
    cols = st.columns(min(3, k))
    for i, f in enumerate(factor_names):
        with cols[i % len(cols)]:
            low[f] = st.number_input(f"{f} — bajo (-1)", value=0.0, key=f"low_d_{f}")
            high[f] = st.number_input(f"{f} — alto (+1)", value=1.0, key=f"high_d_{f}")
    natural_df = decode_design(df, factor_names, low, high)
    st.subheader("Matriz de diseño (unidades naturales)")
    st.dataframe(natural_df, use_container_width=True)
    csv = natural_df.to_csv(index=False).encode("utf-8")
else:
    csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇️ Descargar matriz de diseño (CSV)",
    data=csv,
    file_name=f"diseño_{tipo.split()[0].lower()}_{k}factores.csv",
    mime="text/csv",
)

with st.expander("📖 Notas teóricas"):
    st.markdown(
        """
- **CCD (Central Composite Design):** combina puntos factoriales (2^k),
  puntos axiales o estrella (2k, a distancia ±α del centro) y puntos
  centrales replicados. Es *rotable* cuando α = (2^k)^(1/4), lo que
  garantiza varianza de predicción constante a igual distancia del centro.
  La variante *centrada en las caras* (α=1) mantiene los puntos axiales
  dentro de la región experimental original.
- **BBD (Box-Behnken):** para cada par de factores se corre un factorial
  2² completo manteniendo los factores restantes en el nivel central (0);
  no incluye combinaciones extremas (todos los factores en ±1
  simultáneamente), lo que resulta útil cuando esas combinaciones no son
  factibles en la práctica (p. ej. temperatura y tiempo máximos a la vez).
- Ambos diseños permiten ajustar un **modelo de segundo orden completo**
  (lineal + interacciones + cuadrático) y, si incluyen puntos centrales
  replicados, permiten estimar el **error puro** para la prueba de
  **falta de ajuste**.
        """
    )
