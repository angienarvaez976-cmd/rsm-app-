"""
rsm.model
=========
Ajuste de modelos de primer y segundo orden para RSM, con:
- Matriz de diseño (lineal, interacciones, cuadráticos)
- Estimación por mínimos cuadrados (numpy, sin dependencias pesadas)
- Tabla ANOVA de regresión (SC, gl, CM, F, p-valor)
- Prueba de falta de ajuste (lack of fit) usando puntos repetidos (error puro)
- R2, R2 ajustado, R2 predicho (PRESS)
- Estadísticos de residuos (estandarizados, estudentizados)
"""

from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from scipy import stats


def build_design_matrix(X: np.ndarray, order: str = "second") -> tuple[np.ndarray, list[str]]:
    """
    Construye la matriz de diseño (con intercepto) para un modelo de
    primer orden ('first') o segundo orden completo ('second').

    X : array (n, k) con los niveles codificados de los factores
    Devuelve (matriz, nombres_de_terminos)
    """
    n, k = X.shape
    cols = [np.ones(n)]
    names = ["Intercepto"]

    for i in range(k):
        cols.append(X[:, i])
        names.append(f"X{i+1}")

    if order == "second":
        for i, j in itertools.combinations(range(k), 2):
            cols.append(X[:, i] * X[:, j])
            names.append(f"X{i+1}X{j+1}")
        for i in range(k):
            cols.append(X[:, i] ** 2)
            names.append(f"X{i+1}^2")

    M = np.column_stack(cols)
    return M, names


class RSMModel:
    """Ajusta y almacena un modelo de superficie de respuesta por MCO."""

    def __init__(self, X: np.ndarray, y: np.ndarray, order: str = "second",
                 factor_names: list[str] | None = None):
        self.X_raw = X
        self.y = y
        self.n, self.k = X.shape
        self.order = order
        self.factor_names = factor_names or [f"X{i+1}" for i in range(self.k)]

        self.M, self.term_names = build_design_matrix(X, order=order)
        self.p = self.M.shape[1]  # número de parámetros (incluye intercepto)

        # Mínimos cuadrados
        beta, residuals, rank, sv = np.linalg.lstsq(self.M, y, rcond=None)
        self.beta = beta
        self.y_hat = self.M @ beta
        self.residuals = y - self.y_hat

        # Matriz (M'M)^-1 para errores estándar
        self.MtM_inv = np.linalg.pinv(self.M.T @ self.M)

        self._fit_stats()

    # ------------------------------------------------------------------
    def _fit_stats(self):
        n, p = self.n, self.p
        y = self.y
        y_bar = y.mean()

        self.SST = float(np.sum((y - y_bar) ** 2))
        self.SSE = float(np.sum(self.residuals ** 2))
        self.SSR = self.SST - self.SSE

        self.df_reg = p - 1
        self.df_err = n - p
        self.df_tot = n - 1

        self.MSR = self.SSR / self.df_reg if self.df_reg > 0 else np.nan
        self.MSE = self.SSE / self.df_err if self.df_err > 0 else np.nan

        self.F0 = self.MSR / self.MSE if self.MSE > 0 else np.nan
        self.p_value_reg = (
            1 - stats.f.cdf(self.F0, self.df_reg, self.df_err)
            if not np.isnan(self.F0) else np.nan
        )

        self.R2 = self.SSR / self.SST if self.SST > 0 else np.nan
        self.R2_adj = (
            1 - (1 - self.R2) * (n - 1) / (n - p) if (n - p) > 0 else np.nan
        )

        # Errores estándar y t de cada coeficiente
        se_beta = np.sqrt(np.diag(self.MtM_inv) * self.MSE)
        self.se_beta = se_beta
        with np.errstate(divide="ignore", invalid="ignore"):
            self.t_values = self.beta / se_beta
        self.p_values_beta = 2 * (1 - stats.t.cdf(np.abs(self.t_values), self.df_err))

        # PRESS y R2 predicho (usando leverage hii)
        H = self.M @ self.MtM_inv @ self.M.T
        hii = np.diag(H)
        with np.errstate(divide="ignore", invalid="ignore"):
            press_res = self.residuals / (1 - hii)
        self.PRESS = float(np.sum(press_res ** 2))
        self.R2_pred = 1 - self.PRESS / self.SST if self.SST > 0 else np.nan
        self.hii = hii

        # Residuos estandarizados y estudentizados (internos)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.resid_std = self.residuals / np.sqrt(self.MSE)
            self.resid_student = self.residuals / np.sqrt(self.MSE * (1 - hii))

    # ------------------------------------------------------------------
    def lack_of_fit(self, tol: float = 1e-8) -> dict:
        """
        Prueba de falta de ajuste (Lack of Fit). Requiere puntos de
        diseño repetidos (p.ej. los puntos centrales) para estimar el
        error puro. Agrupa observaciones con niveles de X idénticos.
        """
        # Agrupar filas idénticas de X_raw
        keys = [tuple(np.round(row, 6)) for row in self.X_raw]
        groups: dict[tuple, list[int]] = {}
        for idx, key in enumerate(keys):
            groups.setdefault(key, []).append(idx)

        m = len(groups)  # número de puntos distintos de diseño
        SSPE = 0.0
        df_pe = 0
        for idx_list in groups.values():
            if len(idx_list) > 1:
                y_group = self.y[idx_list]
                SSPE += float(np.sum((y_group - y_group.mean()) ** 2))
                df_pe += len(idx_list) - 1

        if df_pe == 0:
            return {
                "available": False,
                "reason": "No hay puntos de diseño repetidos (se necesitan réplicas, "
                          "p.ej. puntos centrales) para estimar el error puro.",
            }

        SSLOF = self.SSE - SSPE
        df_lof = (self.n - self.p) - df_pe

        if df_lof <= 0:
            return {"available": False, "reason": "Grados de libertad insuficientes para falta de ajuste."}

        MSLOF = SSLOF / df_lof
        MSPE = SSPE / df_pe
        F0 = MSLOF / MSPE if MSPE > tol else np.nan
        p_val = 1 - stats.f.cdf(F0, df_lof, df_pe) if not np.isnan(F0) else np.nan

        return {
            "available": True,
            "SSLOF": SSLOF, "df_lof": df_lof, "MSLOF": MSLOF,
            "SSPE": SSPE, "df_pe": df_pe, "MSPE": MSPE,
            "F0": F0, "p_value": p_val,
            "n_distinct_points": m,
        }

    # ------------------------------------------------------------------
    def anova_table(self) -> pd.DataFrame:
        lof = self.lack_of_fit()
        rows = [
            {"Fuente": "Regresión", "SC": self.SSR, "gl": self.df_reg,
             "CM": self.MSR, "F0": self.F0, "p-valor": self.p_value_reg},
        ]
        if lof["available"]:
            rows.append({"Fuente": "  Falta de ajuste", "SC": lof["SSLOF"], "gl": lof["df_lof"],
                         "CM": lof["MSLOF"], "F0": lof["F0"], "p-valor": lof["p_value"]})
            rows.append({"Fuente": "  Error puro", "SC": lof["SSPE"], "gl": lof["df_pe"],
                         "CM": lof["MSPE"], "F0": np.nan, "p-valor": np.nan})
        rows.append({"Fuente": "Residual", "SC": self.SSE, "gl": self.df_err,
                     "CM": self.MSE, "F0": np.nan, "p-valor": np.nan})
        rows.append({"Fuente": "Total", "SC": self.SST, "gl": self.df_tot,
                     "CM": np.nan, "F0": np.nan, "p-valor": np.nan})
        return pd.DataFrame(rows)

    def coef_table(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Término": self.term_names,
            "Coeficiente": self.beta,
            "Error Std.": self.se_beta,
            "t0": self.t_values,
            "p-valor": self.p_values_beta,
        })

    # ------------------------------------------------------------------
    def predict(self, X_new: np.ndarray) -> np.ndarray:
        M_new, _ = build_design_matrix(X_new, order=self.order)
        return M_new @ self.beta

    def get_quadratic_matrices(self):
        """
        Devuelve b0 (escalar), b (vector lineal k) y B (matriz simétrica kxk
        de coeficientes cuadráticos e interacciones/2), para análisis
        canónico y de cresta. Sólo válido si order == 'second'.
        """
        if self.order != "second":
            raise ValueError("Se requiere un modelo de segundo orden completo.")
        k = self.k
        b0 = self.beta[0]
        b = self.beta[1:1 + k].copy()
        B = np.zeros((k, k))
        idx = 1 + k
        for i, j in itertools.combinations(range(k), 2):
            B[i, j] = self.beta[idx] / 2.0
            B[j, i] = self.beta[idx] / 2.0
            idx += 1
        for i in range(k):
            B[i, i] = self.beta[idx]
            idx += 1
        return b0, b, B
