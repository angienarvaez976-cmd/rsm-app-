"""
rsm.design
==========
Generación de diseños experimentales para Metodología de Superficie de
Respuesta (RSM): Diseño Central Compuesto (CCD) y Diseño Box-Behnken (BBD).

Todas las funciones trabajan en UNIDADES CODIFICADAS (-1, 0, +1, ±alpha).
La decodificación a unidades naturales se hace con `decode_design`.
"""

from __future__ import annotations
import itertools
import numpy as np
import pandas as pd


def ccd_design(k: int, alpha_type: str = "rotatable", n_center: int = 5,
               factor_names: list[str] | None = None) -> pd.DataFrame:
    """
    Genera un Diseño Central Compuesto (Central Composite Design) codificado.

    Parameters
    ----------
    k : número de factores (>=2)
    alpha_type : 'rotatable' (alpha = (2^k)^(1/4)) o 'face' (alpha = 1, CCD centrado en las caras)
    n_center : número de puntos centrales replicados
    factor_names : nombres de columnas (por defecto X1..Xk)

    Returns
    -------
    DataFrame con columnas [X1..Xk, PointType]
    """
    if factor_names is None:
        factor_names = [f"X{i+1}" for i in range(k)]
    assert len(factor_names) == k

    # 1) Puntos factoriales (2^k)
    factorial = np.array(list(itertools.product([-1, 1], repeat=k)), dtype=float)

    # 2) Puntos axiales (2k)
    if alpha_type == "rotatable":
        alpha = (2 ** k) ** 0.25
    elif alpha_type == "face":
        alpha = 1.0
    else:
        raise ValueError("alpha_type debe ser 'rotatable' o 'face'")

    axial = []
    for i in range(k):
        for s in (-1, 1):
            row = [0.0] * k
            row[i] = s * alpha
            axial.append(row)
    axial = np.array(axial)

    # 3) Puntos centrales
    center = np.zeros((n_center, k))

    df = pd.DataFrame(
        np.vstack([factorial, axial, center]), columns=factor_names
    )
    df["PointType"] = (
        ["Factorial"] * len(factorial)
        + ["Axial"] * len(axial)
        + ["Central"] * len(center)
    )
    df.insert(0, "Run", range(1, len(df) + 1))
    df.attrs["alpha"] = alpha
    df.attrs["design"] = "CCD"
    df.attrs["k"] = k
    return df.reset_index(drop=True)


def bbd_design(k: int, n_center: int = 3,
               factor_names: list[str] | None = None) -> pd.DataFrame:
    """
    Genera un Diseño Box-Behnken codificado para k = 3, 4 o 5 factores.

    Construcción: para cada par de factores (i, j) se corre el factorial
    2^2 completo (±1, ±1) manteniendo el resto de los factores en 0. Esto
    reproduce exactamente el diseño estándar de Box-Behnken para k=3
    (12 puntos + centrales) y es la generalización estándar usada en
    software de RSM para k=4 y k=5.
    """
    if k not in (3, 4, 5):
        raise ValueError("BBD implementado para k = 3, 4 o 5 factores")
    if factor_names is None:
        factor_names = [f"X{i+1}" for i in range(k)]
    assert len(factor_names) == k

    rows = []
    for i, j in itertools.combinations(range(k), 2):
        for si, sj in itertools.product([-1, 1], repeat=2):
            row = [0.0] * k
            row[i] = si
            row[j] = sj
            rows.append(row)

    factorial = np.array(rows)
    center = np.zeros((n_center, k))

    df = pd.DataFrame(np.vstack([factorial, center]), columns=factor_names)
    df["PointType"] = ["Factorial"] * len(factorial) + ["Central"] * len(center)
    df.insert(0, "Run", range(1, len(df) + 1))
    df.attrs["design"] = "BBD"
    df.attrs["k"] = k
    return df.reset_index(drop=True)


def decode_design(coded_df: pd.DataFrame, factor_names: list[str],
                   low: dict, high: dict) -> pd.DataFrame:
    """
    Convierte un diseño codificado (-1,+1, ±alpha) a unidades naturales,
    usando los niveles bajo/alto (-1/+1) de cada factor. La transformación
    es lineal: x_natural = center + coded * half_range
    """
    out = coded_df.copy()
    for f in factor_names:
        center = (low[f] + high[f]) / 2.0
        half_range = (high[f] - low[f]) / 2.0
        out[f] = center + out[f] * half_range
    return out


def encode_value(x_natural: float, low: float, high: float) -> float:
    """Codifica un valor natural a escala (-1, +1) dado bajo/alto."""
    center = (low + high) / 2.0
    half_range = (high - low) / 2.0
    return (x_natural - center) / half_range
