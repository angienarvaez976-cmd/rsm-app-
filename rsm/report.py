"""
rsm.report
==========
Generación de un informe final en PDF (usando reportlab) que resume el
estudio de RSM realizado en la app: configuración, ajuste de modelo(s),
ANOVA, falta de ajuste, optimización y deseabilidad compuesta.

No depende de Streamlit ni de Plotly/kaleido (para evitar problemas de
despliegue en la nube relacionados con renderizado de gráficos
headless); el informe se centra en tablas y texto interpretativo.
"""

from __future__ import annotations
import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, ListFlowable, ListItem
)

TEAL = colors.HexColor("#2E7D6E")
LIGHT_TEAL = colors.HexColor("#EAF3F1")
GRAY = colors.HexColor("#4B5563")


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle(name="TituloInforme", fontSize=20, leading=24,
                           textColor=TEAL, spaceAfter=6, fontName="Helvetica-Bold"))
    ss.add(ParagraphStyle(name="Subtitulo", fontSize=11, leading=14,
                           textColor=GRAY, spaceAfter=14))
    ss.add(ParagraphStyle(name="Seccion", fontSize=14, leading=18,
                           textColor=TEAL, spaceBefore=14, spaceAfter=8, fontName="Helvetica-Bold"))
    ss.add(ParagraphStyle(name="Subseccion", fontSize=12, leading=15,
                           textColor=colors.HexColor("#1F2937"), spaceBefore=10, spaceAfter=6,
                           fontName="Helvetica-Bold"))
    ss.add(ParagraphStyle(name="CuerpoInforme", fontSize=10, leading=14, spaceAfter=6))
    ss.add(ParagraphStyle(name="Nota", fontSize=9, leading=12, textColor=GRAY,
                           spaceAfter=6, fontName="Helvetica-Oblique"))
    return ss


def _df_to_table(df, styles, col_widths=None, fmt=None):
    """Convierte un DataFrame en una Table de reportlab con formato aplicado."""
    fmt = fmt or {}
    header = [Paragraph(f"<b>{c}</b>", styles["CuerpoInforme"]) for c in df.columns]
    rows = [header]
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            if c in fmt and v is not None:
                try:
                    v = fmt[c].format(v)
                except (ValueError, TypeError):
                    v = "" if v is None else str(v)
            else:
                v = "" if v is None else str(v)
            cells.append(Paragraph(v, styles["CuerpoInforme"]))
        rows.append(cells)

    t = Table(rows, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_TEAL]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def build_pdf_report(meta: dict, factor_info: list, response_sections: list,
                      opt_sections: list, desirability_section: dict | None) -> bytes:
    """
    meta: {"titulo":str, "autor":str, "dataset_name":str, "n_obs":int}
    factor_info: [{"nombre":str, "bajo":float|None, "alto":float|None}, ...]
    response_sections: [{
        "nombre": str, "order": str, "R2":float, "R2_adj":float, "R2_pred":float,
        "anova_df": DataFrame, "coef_df": DataFrame,
        "lof_available": bool, "lof_p": float|None, "significativo": bool,
    }, ...]
    opt_sections: [{
        "response": str, "objetivo": str,
        "canonical": {"stationary_point": list, "y_stationary": float, "nature": str} | None,
        "numeric": {"x_opt": list, "y_opt": float} | None,
    }, ...]
    desirability_section: {
        "responses": [str], "specs": dict, "x_opt": list, "D_opt": float,
        "detail": [{"respuesta":str, "y_hat":float, "d":float}, ...],
    } | None
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=LETTER,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
        title=meta.get("titulo", "Informe RSM"),
    )
    styles = _styles()
    story = []

    # ---------------- Portada / encabezado ----------------
    story.append(Paragraph(meta.get("titulo", "Informe de Análisis RSM de Segundo Orden"), styles["TituloInforme"]))
    story.append(Paragraph(
        f"Conjunto de datos: <b>{meta.get('dataset_name', 'N/D')}</b> · "
        f"{meta.get('n_obs', '?')} observaciones · "
        f"Generado el {datetime.now().strftime('%d/%m/%Y %H:%M')}"
        + (f" · Elaborado por: {meta['autor']}" if meta.get("autor") else ""),
        styles["Subtitulo"],
    ))

    # ---------------- 1. Configuración ----------------
    story.append(Paragraph("1. Configuración del estudio", styles["Seccion"]))
    if factor_info:
        import pandas as pd
        fdf = pd.DataFrame(factor_info)
        fdf.columns = ["Factor", "Nivel bajo (-1)", "Nivel alto (+1)"]
        story.append(_df_to_table(fdf, styles, fmt={"Nivel bajo (-1)": "{:.3f}", "Nivel alto (+1)": "{:.3f}"}))
    story.append(Spacer(1, 10))

    # ---------------- 2. Ajuste de modelo(s) ----------------
    story.append(Paragraph("2. Ajuste del Modelo de Superficie de Respuesta", styles["Seccion"]))
    for rs in response_sections:
        story.append(Paragraph(f"Respuesta: {rs['nombre']}", styles["Subseccion"]))
        orden_txt = "segundo orden completo" if rs["order"] == "second" else "primer orden"
        story.append(Paragraph(
            f"Modelo de {orden_txt} ajustado por mínimos cuadrados. "
            f"R² = {rs['R2']:.4f} · R² ajustado = {rs['R2_adj']:.4f}"
            + (f" · R² predicho = {rs['R2_pred']:.4f}" if rs.get("R2_pred") is not None else ""),
            styles["CuerpoInforme"],
        ))
        story.append(_df_to_table(
            rs["anova_df"], styles,
            fmt={"SC": "{:.4f}", "gl": "{:.0f}", "CM": "{:.4f}", "F0": "{:.3f}", "p-valor": "{:.4g}"},
        ))
        if rs["lof_available"]:
            concl = (
                "se detectó falta de ajuste significativa; el modelo podría no capturar "
                "adecuadamente la curvatura real de la respuesta."
                if not rs["lof_ok"] else
                "no se encontró evidencia de falta de ajuste: el modelo describe adecuadamente la relación."
            )
            story.append(Paragraph(
                f"<b>Falta de ajuste:</b> p-valor = {rs['lof_p']:.4g} — {concl}",
                styles["CuerpoInforme"],
            ))
        else:
            story.append(Paragraph(
                "<b>Falta de ajuste:</b> no disponible (se requieren puntos de diseño repetidos).",
                styles["Nota"],
            ))
        story.append(Spacer(1, 4))
        story.append(Paragraph("Coeficientes del modelo:", styles["CuerpoInforme"]))
        story.append(_df_to_table(
            rs["coef_df"], styles,
            fmt={"Coeficiente": "{:.4f}", "Error Std.": "{:.4f}", "t0": "{:.3f}", "p-valor": "{:.4g}"},
        ))
        story.append(Spacer(1, 12))

    # ---------------- 3. Optimización ----------------
    if opt_sections:
        story.append(PageBreak())
        story.append(Paragraph("3. Optimización", styles["Seccion"]))
        for op in opt_sections:
            story.append(Paragraph(f"Respuesta: {op['response']} — Objetivo: {op['objetivo']}", styles["Subseccion"]))
            if op.get("canonical"):
                ca = op["canonical"]
                pts = ", ".join(f"{v:.3f}" for v in ca["stationary_point"])
                story.append(Paragraph(
                    f"<b>Análisis canónico:</b> punto estacionario (codificado) = ({pts}); "
                    f"respuesta estimada en ese punto = {ca['y_stationary']:.4f}. "
                    f"Naturaleza: {ca['nature']}.",
                    styles["CuerpoInforme"],
                ))
            if op.get("numeric"):
                nm = op["numeric"]
                pts = ", ".join(f"{v:.3f}" for v in nm["x_opt"])
                story.append(Paragraph(
                    f"<b>Optimización numérica restringida:</b> condiciones óptimas (codificadas) = "
                    f"({pts}); valor óptimo estimado = {nm['y_opt']:.4f}.",
                    styles["CuerpoInforme"],
                ))
            story.append(Spacer(1, 8))

    # ---------------- 4. Deseabilidad compuesta ----------------
    if desirability_section:
        story.append(Paragraph("4. Optimización Simultánea (Deseabilidad de Derringer-Suich)", styles["Seccion"]))
        d = desirability_section
        pts = ", ".join(f"{v:.3f}" for v in d["x_opt"])
        story.append(Paragraph(
            f"Respuestas combinadas: {', '.join(d['responses'])}. "
            f"Condiciones óptimas (codificadas) = ({pts}). "
            f"<b>Deseabilidad compuesta D = {d['D_opt']:.4f}</b>.",
            styles["CuerpoInforme"],
        ))
        import pandas as pd
        ddf = pd.DataFrame(d["detail"])
        story.append(_df_to_table(ddf, styles, fmt={"y_hat": "{:.3f}", "d": "{:.3f}"}))
        interp = (
            "alta: las respuestas se satisfacen bien de forma simultánea." if d["D_opt"] >= 0.7 else
            "moderada: buen compromiso, con margen de mejora." if d["D_opt"] >= 0.4 else
            "baja: conviene relajar especificaciones o priorizar la(s) respuesta(s) más crítica(s)."
        )
        story.append(Paragraph(f"Interpretación: deseabilidad compuesta {interp}", styles["CuerpoInforme"]))

    story.append(Spacer(1, 16))
    story.append(Paragraph(
        "Informe generado automáticamente por el aplicativo de RSM de segundo orden. "
        "Los gráficos interactivos (contornos, superficies 3D, Pareto de efectos y perturbación) "
        "están disponibles dentro de la aplicación, en la sección de Visualización.",
        styles["Nota"],
    ))

    doc.build(story)
    return buf.getvalue()
