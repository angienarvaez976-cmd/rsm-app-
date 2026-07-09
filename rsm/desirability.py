"""
rsm.desirability
================
Función de deseabilidad de Derringer-Suich (1980) para optimización
simultánea de múltiples respuestas.
"""

from __future__ import annotations
import numpy as np


def desirability_larger_is_better(y, low, target, s=1.0):
    """d = 0 si y<=low; ((y-low)/(target-low))^s si low<y<target; 1 si y>=target"""
    y = np.asarray(y, dtype=float)
    d = np.where(
        y <= low, 0.0,
        np.where(y >= target, 1.0, ((y - low) / (target - low)) ** s)
    )
    return d


def desirability_smaller_is_better(y, target, high, t=1.0):
    """d = 1 si y<=target; ((high-y)/(high-target))^t si target<y<high; 0 si y>=high"""
    y = np.asarray(y, dtype=float)
    d = np.where(
        y <= target, 1.0,
        np.where(y >= high, 0.0, ((high - y) / (high - target)) ** t)
    )
    return d


def desirability_target_is_best(y, low, target, high, s=1.0, t=1.0):
    """d crece de 0 en 'low' a 1 en 'target' y decrece a 0 en 'high'."""
    y = np.asarray(y, dtype=float)
    d_low = np.where((y > low) & (y <= target), ((y - low) / (target - low)) ** s, 0.0)
    d_high = np.where((y > target) & (y < high), ((high - y) / (high - target)) ** t, 0.0)
    d = np.where(y == target, 1.0, d_low + d_high)
    d = np.where((y <= low) | (y >= high), 0.0, d)
    d = np.where(y == target, 1.0, d)
    return np.clip(d, 0.0, 1.0)


def composite_desirability(d_list: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """
    Deseabilidad global (media geométrica ponderada de Derringer-Suich):
        D = ( prod(d_i ^ w_i) ) ^ (1 / sum(w_i))
    Si algún d_i = 0, D = 0 (una respuesta inaceptable invalida el conjunto).
    """
    d_arr = np.array(d_list, dtype=float)  # (n_respuestas, n_puntos)
    if weights is None:
        weights = [1.0] * d_arr.shape[0]
    weights = np.array(weights, dtype=float)

    with np.errstate(divide="ignore"):
        log_d = np.where(d_arr > 0, np.log(np.clip(d_arr, 1e-300, None)), -np.inf)
    weighted_log = (weights[:, None] * log_d).sum(axis=0) / weights.sum()
    D = np.exp(weighted_log)
    D = np.where(np.any(d_arr <= 0, axis=0), 0.0, D)
    return D
