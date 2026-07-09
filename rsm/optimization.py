"""
rsm.optimization
================
Herramientas de optimización para RSM:
- Ascenso/descenso más pronunciado (steepest ascent/descent), modelo 1er orden
- Análisis canónico (punto estacionario, eigenvalores/eigenvectores, naturaleza)
- Análisis de cresta (ridge analysis, método de Lagrange de Hoerl)
- Optimización numérica restringida (dentro de la región experimental)
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import minimize


def steepest_ascent(b0: float, b: np.ndarray, base_point: np.ndarray,
                     step_factor_index: int = 0, step_size: float = 1.0,
                     n_steps: int = 6, maximize: bool = True) -> "pd.DataFrame":
    """
    Genera la trayectoria de ascenso (o descenso) más pronunciado a partir
    de un modelo de primer orden y_hat = b0 + b'x, en unidades codificadas.

    step_factor_index: índice del factor usado como referencia para definir
        el tamaño de paso (usualmente el de mayor |b_i|).
    step_size: tamaño de paso codificado para ese factor de referencia.
    """
    import pandas as pd
    sign = 1.0 if maximize else -1.0
    direction = sign * b / abs(b[step_factor_index])
    steps = np.arange(0, n_steps + 1)
    coords = base_point[None, :] + steps[:, None] * step_size * direction[None, :]
    y_pred = b0 + coords @ b
    df = pd.DataFrame(coords, columns=[f"X{i+1}" for i in range(len(b))])
    df.insert(0, "Paso", steps)
    df["y_predicho"] = y_pred
    return df


def canonical_analysis(b0: float, b: np.ndarray, B: np.ndarray) -> dict:
    """
    Análisis canónico del modelo de segundo orden:
        y = b0 + b'x + x'Bx
    Punto estacionario: x_s = -0.5 * B^-1 * b
    y en el punto estacionario: y_s = b0 + 0.5 * b' x_s
    Naturaleza según signos de los eigenvalores de B.
    """
    try:
        B_inv = np.linalg.inv(B)
        singular = False
    except np.linalg.LinAlgError:
        B_inv = np.linalg.pinv(B)
        singular = True

    x_s = -0.5 * B_inv @ b
    y_s = b0 + 0.5 * b @ x_s

    eigvals, eigvecs = np.linalg.eigh(B)

    if np.all(eigvals < -1e-10):
        nature = "Máximo (todos los eigenvalores son negativos)"
    elif np.all(eigvals > 1e-10):
        nature = "Mínimo (todos los eigenvalores son positivos)"
    elif np.any(np.abs(eigvals) < 1e-10):
        nature = "Sistema estacionario con eje mayor (algún eigenvalor ~ 0)"
    else:
        nature = "Punto silla (eigenvalores con signos mixtos)"

    return {
        "stationary_point": x_s,
        "y_stationary": y_s,
        "eigenvalues": eigvals,
        "eigenvectors": eigvecs,
        "nature": nature,
        "B_singular": singular,
    }


def ridge_analysis(b0: float, b: np.ndarray, B: np.ndarray,
                    radii: np.ndarray, maximize: bool = True) -> "pd.DataFrame":
    """
    Análisis de cresta (Ridge Analysis, método de Hoerl/Myers-Montgomery):
    para cada radio r se busca x que optimice y = b0 + b'x + x'Bx sujeto a
    ||x|| = r, resolviendo (B - mu*I) x = -0.5 b para el multiplicador de
    Lagrange mu adecuado (barrido numérico) y evaluando y(x).
    """
    import pandas as pd
    k = len(b)
    I = np.eye(k)
    eigvals = np.linalg.eigvalsh(B)

    # Rango de mu a explorar: debe estar fuera del rango de eigenvalores de B
    # para invertir (B - mu I). Para maximizar sobre radios crecientes,
    # mu debe ser mayor que el mayor eigenvalor (maximize) o menor que el
    # menor eigenvalor (minimize) para que el punto crítico sea de verdad
    # un extremo en la esfera.
    lo, hi = eigvals.min(), eigvals.max()
    if maximize:
        mu_grid = np.linspace(hi + 1e-3, hi + 50, 4000)
    else:
        mu_grid = np.linspace(lo - 50, lo - 1e-3, 4000)

    rows = []
    for mu in mu_grid:
        M = B - mu * I
        try:
            x = np.linalg.solve(M, -0.5 * b)
        except np.linalg.LinAlgError:
            continue
        r = np.linalg.norm(x)
        y = b0 + b @ x + x @ B @ x
        rows.append((r, mu, y, *x))

    cols = ["radio", "mu", "y_predicho"] + [f"X{i+1}" for i in range(k)]
    full = pd.DataFrame(rows, columns=cols).sort_values("radio").reset_index(drop=True)

    # Interpolar a los radios solicitados
    out = []
    for r_target in radii:
        idx = (full["radio"] - r_target).abs().idxmin()
        out.append(full.loc[idx])
    result = pd.DataFrame(out).reset_index(drop=True)
    result["radio_objetivo"] = radii
    return result


def numeric_optimize(predict_fn, k: int, x0: np.ndarray | None = None,
                      bounds: tuple = (-1.0, 1.0), maximize: bool = True):
    """
    Optimización numérica restringida a la región experimental
    (hipercubo codificado, por defecto [-1,1]^k), usando SLSQP.
    predict_fn: función que recibe un array (k,) y retorna un escalar y_hat.
    """
    if x0 is None:
        x0 = np.zeros(k)
    sign = -1.0 if maximize else 1.0
    obj = lambda x: sign * predict_fn(x)
    bnds = [bounds] * k
    res = minimize(obj, x0, method="SLSQP", bounds=bnds)
    return {
        "x_opt": res.x,
        "y_opt": predict_fn(res.x),
        "success": res.success,
        "message": res.message,
    }
