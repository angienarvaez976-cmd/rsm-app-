"""Funciones auxiliares compartidas entre las páginas de la app Streamlit."""

from __future__ import annotations
import numpy as np
import pandas as pd

from rsm.design import encode_value


def get_coded_X(data: pd.DataFrame, factor_cols: list[str], already_coded: bool,
                 coding_limits: dict) -> np.ndarray:
    """Devuelve la matriz X codificada (-1,+1) a partir de los datos crudos."""
    if already_coded:
        return data[factor_cols].values.astype(float)
    cols = []
    for f in factor_cols:
        low, high = coding_limits[f]
        cols.append(data[f].apply(lambda v: encode_value(v, low, high)).values)
    return np.column_stack(cols).astype(float)


def check_ready(st, require_response: bool = True):
    """Verifica que existan datos y variables configuradas; si no, detiene la página."""
    data = st.session_state.get("data")
    factor_cols = st.session_state.get("factor_cols", [])
    response_cols = st.session_state.get("response_cols", [])

    if data is None:
        st.warning("Primero carga los datos en la página **Inicio**.")
        st.stop()
    if not factor_cols:
        st.warning("Selecciona al menos un factor en la página **Inicio**.")
        st.stop()
    if require_response and not response_cols:
        st.warning("Selecciona al menos una variable de respuesta en la página **Inicio**.")
        st.stop()
    return data, factor_cols, response_cols
