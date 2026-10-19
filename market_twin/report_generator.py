"""
Market Twin • Gerador de Relatório Executivo em PDF (ReportLab)
Produz um dossiê executivo para comitês de investimento, sócios e venture capital.
"""

from __future__ import annotations

import io
import os
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from market_twin.models import SimulationResult


class ExecutiveReportGenerator:
    """
    Gera relatórios executivos em formato PDF vetorial com métricas quantitativas e qualitativas.
    """

    def generate_pdf(self, result: SimulationResult, output_path: Optional[str] = None) -> bytes:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            output_path or buffer,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        styles = getSampleStyleSheet()

        # Estilos Customizados
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#0F172A"),
        )
        subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=15,
            textColor=colors.HexColor("#475569"),
        )
        section_style = ParagraphStyle(
            "SectionHeading",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#1E293B"),
            spaceBefore=14,
            spaceAfter=6,
        )
        body_style = ParagraphStyle(
            "BodyDark",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13.5,
            textColor=colors.HexColor("#334155"),
        )
        table_header_style = ParagraphStyle(
            "TableHeader",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11,
            textColor=colors.white,
        )
        table_cell_style = ParagraphStyle(
            "TableCell",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#1E293B"),
        )

        elements = []

        # Cabeçalho Executivo
        elements.append(Paragraph("MARKET TWIN • DOSSIÊ DE VALIDAÇÃO DE MERCADO", subtitle_style))
        elements.append(Spacer(1, 4))
        elements.append(Paragraph(f"Análise de Viabilidade: {result.project_name}", title_style))
        elements.append(Spacer(1, 4))
        meta_text = (
            f"População Sintética: <b>{result.population_size:,} clones virtuais</b> | "
            f"Data: {result.timestamp} | Tempo de Inferência: {result.simulation_duration_seconds}s"
        )
        elements.append(Paragraph(meta_text, subtitle_style))
        elements.append(Spacer(1, 10))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#CBD5E1"), spaceAfter=14))

        # Quadro do Veredito Executivo
        badge_color = colors.HexColor("#059669") if result.verdict_badge == "success" else (
            colors.HexColor("#D97706") if result.verdict_badge == "warning" else colors.HexColor("#DC2626")
        )
        bg_color = colors.HexColor("#ECFDF5") if result.verdict_badge == "success" else (
            colors.HexColor("#FFFBEB") if result.verdict_badge == "warning" else colors.HexColor("#FEF2F2")
        )

        verdict_header = Paragraph(f"<b>VEREDITO DE INVESTIMENTO: {result.verdict}</b>", ParagraphStyle(
            "VerdictHead", parent=title_style, fontSize=13, leading=16, textColor=badge_color
        ))
        verdict_text = Paragraph(result.verdict_rationale, body_style)

        verdict_table = Table([[verdict_header], [verdict_text]], colWidths=[540])
        verdict_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), bg_color),
            ("BOX", (0, 0), (-1, -1), 1.5, badge_color),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ]))
        elements.append(verdict_table)
        elements.append(Spacer(1, 14))

        # Cards com Indicadores-Chave (PMF, NPS, Conversão)
        kpi_data = [
            [
                Paragraph("<b>Índice PMF (0-100)</b>", table_header_style),
                Paragraph("<b>Net Promoter Score (NPS)</b>", table_header_style),
                Paragraph("<b>Taxa de Conversão Global</b>", table_header_style),
            ],
            [
                Paragraph(f"<font size=16><b>{result.pmf_index:.1f}</b></font>/100", table_cell_style),
                Paragraph(f"<font size=16><b>{result.nps_score:+.1f}</b></font>", table_cell_style),
                Paragraph(f"<font size=16><b>{result.conversion_rate:.2f}%</b></font>", table_cell_style),
            ],
        ]
        kpi_table = Table(kpi_data, colWidths=[180, 180, 180])
        kpi_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
            ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#F8FAFC")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#E2E8F0")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        elements.append(kpi_table)
        elements.append(Spacer(1, 14))

        # Tabela: Segmentação por Macrorregião Global
        elements.append(Paragraph("1. Desempenho e Aderência por Macrorregião Global", section_style))
        reg_rows = [
            [
                Paragraph("Região", table_header_style),
                Paragraph("Amostra", table_header_style),
                Paragraph("Compradores", table_header_style),
                Paragraph("Taxa Conversão", table_header_style),
                Paragraph("Média PMF", table_header_style),
                Paragraph("WTP Médio (USD)", table_header_style),
            ]
        ]
        for r in result.breakdown_by_region:
            reg_rows.append([
                Paragraph(r.region, table_cell_style),
                Paragraph(f"{r.total_population:,}", table_cell_style),
                Paragraph(f"{r.converted_users:,}", table_cell_style),
                Paragraph(f"<b>{r.conversion_rate:.2f}%</b>", table_cell_style),
                Paragraph(f"{r.average_pmf:.2f}/10", table_cell_style),
                Paragraph(f"${r.average_wtp_usd:.2f}", table_cell_style),
            ])

        reg_table = Table(reg_rows, colWidths=[130, 75, 80, 85, 85, 85])
        reg_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        elements.append(reg_table)
        elements.append(Spacer(1, 14))

        # Tabela: Curva de Elasticidade de Preço
        elements.append(Paragraph("2. Sensibilidade de Preço & Elasticidade (Van Westendorp PSM)", section_style))
        price_rows = [
            [
                Paragraph("Preço Simulado (USD/mês)", table_header_style),
                Paragraph("Taxa de Conversão", table_header_style),
                Paragraph("Índice de Faturamento Projetado", table_header_style),
            ]
        ]
        for pe in result.price_elasticity:
            price_rows.append([
                Paragraph(f"${pe.price_usd:.2f}", table_cell_style),
                Paragraph(f"{pe.conversion_rate:.2f}%", table_cell_style),
                Paragraph(f"{pe.projected_revenue_index:.1f}%", table_cell_style),
            ])

        price_table = Table(price_rows, colWidths=[180, 180, 180])
        price_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#475569")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(price_table)
        elements.append(Spacer(1, 14))

        # Focus Group Qualitativo (Tier 2)
        elements.append(Paragraph("3. Voz do Cliente: Verbatims do Focus Group Sintético (SLM Tier 2)", section_style))
        for fg in result.focus_group:
            fg_content = [
                [
                    Paragraph(
                        f"<b>{fg.representative_name}</b> ({fg.country} • {fg.region.value}) | "
                        f"Perfil: <i>{fg.archetype.value}</i> | Sentimento: <b>{fg.sentiment}</b>",
                        ParagraphStyle("FGHead", parent=table_cell_style, fontName="Helvetica-Bold")
                    )
                ],
                [
                    Paragraph(f'"{fg.verbatim_quote}"', ParagraphStyle("FGQuote", parent=body_style, fontName="Helvetica-Oblique"))
                ],
                [
                    Paragraph(f"<b>Principal Objeção:</b> {fg.top_objection} | <b>Gatilho de Fechamento:</b> {fg.key_buying_trigger}", body_style)
                ],
            ]
            fg_t = Table(fg_content, colWidths=[540])
            fg_t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#CBD5E1")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]))
            elements.append(fg_t)
            elements.append(Spacer(1, 6))

        # Rodapé de Confidencialidade
        elements.append(Spacer(1, 10))
        elements.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#CBD5E1"), spaceAfter=6))
        elements.append(Paragraph(
            "Dossiê gerado confidencialmente pelo ecossistema Market Twin. Uso exclusivo para avaliação de investimentos.",
            ParagraphStyle("Footer", parent=subtitle_style, fontSize=8, alignment=1)
        ))

        doc.build(elements)

        if output_path and os.path.exists(output_path):
            with open(output_path, "rb") as f:
                return f.read()

        return buffer.getvalue()
