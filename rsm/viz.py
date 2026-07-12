"""
rsm.viz
=======
Visualizaciones para RSM usando Plotly: gráficos de contorno, superficies
3D, diagrama de Pareto de efectos estandarizados y gráfico de perturbación.
"""

from __future__ import annotations
import itertools
import numpy as np
import plotly.graph_objects as go


def contour_plot(model, factor_i: int, factor_j: int, fixed_values: np.ndarray,
                  factor_names: list[str], n_grid: int = 60, response_name: str = "y"):
    lo, hi = -1.5, 1.5
    xi = np.linspace(lo, hi, n_grid)
    xj = np.linspace(lo, hi, n_grid)
    XI, XJ = np.meshgrid(xi, xj)

    Z = np.zeros_like(XI)
    for a in range(n_grid):
        for b in range(n_grid):
            x = fixed_values.copy()
            x[factor_i] = XI[a, b]
            x[factor_j] = XJ[a, b]
            Z[a, b] = model.predict(x.reshape(1, -1))[0]

    fig = go.Figure(data=go.Contour(
        x=xi, y=xj, z=Z,
        colorscale="Tealrose",
        contours=dict(showlabels=True, labelfont=dict(size=11, color="white")),
        colorbar=dict(title=response_name),
    ))
    fig.update_layout(
        title=f"Contorno: {response_name} vs {factor_names[factor_i]} y {factor_names[factor_j]}",
        xaxis_title=f"{factor_names[factor_i]} (codificado)",
        yaxis_title=f"{factor_names[factor_j]} (codificado)",
        template="plotly_white",
    )
    return fig


def surface_plot(model, factor_i: int, factor_j: int, fixed_values: np.ndarray,
                  factor_names: list[str], n_grid: int = 50, response_name: str = "y"):
    lo, hi = -1.5, 1.5
    xi = np.linspace(lo, hi, n_grid)
    xj = np.linspace(lo, hi, n_grid)
    XI, XJ = np.meshgrid(xi, xj)

    Z = np.zeros_like(XI)
    for a in range(n_grid):
        for b in range(n_grid):
            x = fixed_values.copy()
            x[factor_i] = XI[a, b]
            x[factor_j] = XJ[a, b]
            Z[a, b] = model.predict(x.reshape(1, -1))[0]

    fig = go.Figure(data=go.Surface(x=xi, y=xj, z=Z, colorscale="Tealrose"))
    fig.update_layout(
        title=f"Superficie 3D: {response_name} vs {factor_names[factor_i]} y {factor_names[factor_j]}",
        scene=dict(
            xaxis_title=factor_names[factor_i],
            yaxis_title=factor_names[factor_j],
            zaxis_title=response_name,
        ),
        template="plotly_white",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def pareto_effects_plot(model, alpha: float = 0.05):
    """
    Diagrama de Pareto de efectos estandarizados (|t0| de cada término,
    excluyendo el intercepto), con línea de referencia en el valor crítico
    t de Student para el nivel de significancia dado.
    """
    from scipy import stats as sstats

    terms = model.term_names[1:]
    t_vals = np.abs(model.t_values[1:])
    order = np.argsort(t_vals)[::-1]
    terms_sorted = [terms[i] for i in order]
    t_sorted = t_vals[order]

    t_crit = sstats.t.ppf(1 - alpha / 2, model.df_err)

    colors = ["#2E7D6E" if t >= t_crit else "#B0BEC5" for t in t_sorted]

    fig = go.Figure(go.Bar(
        x=t_sorted, y=terms_sorted, orientation="h", marker_color=colors,
    ))
    fig.add_vline(x=t_crit, line_dash="dash", line_color="crimson",
                  annotation_text=f"t crítico ({alpha:.2f}) = {t_crit:.2f}")
    fig.update_layout(
        title="Diagrama de Pareto de efectos estandarizados",
        xaxis_title="|t0| (efecto estandarizado)",
        yaxis_title="Término",
        template="plotly_white",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def perturbation_plot(model, factor_names: list[str], center: np.ndarray | None = None,
                       response_name: str = "y", n_points: int = 41):
    k = model.k
    if center is None:
        center = np.zeros(k)
    dev = np.linspace(-1, 1, n_points)

    fig = go.Figure()
    palette = ["#2E7D6E", "#D98E04", "#7E57C2", "#EF5350", "#42A5F5", "#8D6E63"]
    for i in range(k):
        Y = []
        for d in dev:
            x = center.copy()
            x[i] = d
            Y.append(model.predict(x.reshape(1, -1))[0])
        fig.add_trace(go.Scatter(
            x=dev, y=Y, mode="lines", name=factor_names[i],
            line=dict(color=palette[i % len(palette)], width=2.5),
        ))
    fig.update_layout(
        title=f"Gráfico de perturbación: {response_name}",
        xaxis_title="Desviación desde el centro (unidades codificadas)",
        yaxis_title=response_name,
        template="plotly_white",
    )
    return fig
