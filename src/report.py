import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from xml.sax.saxutils import escape

from src import details as details_module
from src.aggregator import CATEGORY_ICON, RECOMMENDATIONS

FONTS_DIR = Path(__file__).resolve().parent / "fonts"
FONT_REGULAR = "DejaVuSans"
FONT_BOLD = "DejaVuSans-Bold"

# reportlab's built-in fonts (Helvetica etc.) have no Cyrillic glyphs; register a
# Unicode font so Ukrainian text renders instead of "notdef" boxes.
pdfmetrics.registerFont(TTFont(FONT_REGULAR, str(FONTS_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont(FONT_BOLD, str(FONTS_DIR / "DejaVuSans-Bold.ttf")))

STYLES = getSampleStyleSheet()
TITLE_STYLE = ParagraphStyle("ReportTitle", parent=STYLES["Title"], fontName=FONT_BOLD, fontSize=20)
SECTION_STYLE = ParagraphStyle(
    "Section", parent=STYLES["Heading2"], fontName=FONT_BOLD, spaceBefore=12, spaceAfter=6
)
BODY_STYLE = ParagraphStyle("Body", parent=STYLES["BodyText"], fontName=FONT_REGULAR)
LINE_STYLE = ParagraphStyle("Line", parent=BODY_STYLE, fontSize=9, leading=12, leftIndent=8)
CELL_STYLE = ParagraphStyle("Cell", parent=BODY_STYLE, fontSize=8, leading=10)
CELL_HEAD_STYLE = ParagraphStyle("CellHead", parent=CELL_STYLE, fontName=FONT_BOLD)

TABLE_STYLE = TableStyle(
    [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("FONTNAME", (0, 0), (-1, -1), FONT_REGULAR),
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
)


def _fmt(value, suffix=""):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def build_report_elements(context):
    """context keys: name, industry, funding_stage, date, score, category,
    ml_score, fuzzy_score, deal_score (optional), weights (optional dict),
    team_quality, traction, market_size, competition,
    funding_ask (optional), pre_money_valuation (optional)."""
    elements = []

    elements.append(Paragraph("Звіт з оцінки інвестиційної привабливості стартапу", TITLE_STYLE))
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(
        Paragraph(
            f"{context['name']} · {context.get('industry', '—')} · {context.get('funding_stage', '—')}",
            BODY_STYLE,
        )
    )
    elements.append(
        Paragraph(
            f"Дата аналізу: {context.get('date', datetime.date.today().isoformat())}",
            BODY_STYLE,
        )
    )
    elements.append(Spacer(1, 0.5 * cm))

    elements.append(Paragraph("Результат", SECTION_STYLE))
    icon = CATEGORY_ICON.get(context["category"], "")
    elements.append(
        Paragraph(f"<b>Score: {context['score']:.1f} / 100</b>", BODY_STYLE)
    )
    elements.append(
        Paragraph(f"{icon} Категорія: <b>{context['category']}</b>", BODY_STYLE)
    )
    elements.append(Spacer(1, 0.2 * cm))
    elements.append(Paragraph(RECOMMENDATIONS.get(context["category"], ""), BODY_STYLE))
    elements.append(Spacer(1, 0.4 * cm))

    elements.append(Paragraph("Розбивка за компонентами", SECTION_STYLE))
    weights = context.get("weights") or {}
    breakdown_data = [["Компонент", "Оцінка", "Вага"]]
    breakdown_data.append(["ML-прогноз M(X)", _fmt(context.get("ml_score")), _fmt(weights.get("ml"))])
    breakdown_data.append(["Fuzzy-оцінка F(Y)", _fmt(context.get("fuzzy_score")), _fmt(weights.get("fuzzy"))])
    if context.get("deal_score") is not None:
        breakdown_data.append(["Умови угоди G(D)", _fmt(context.get("deal_score")), _fmt(weights.get("deal"))])
    breakdown_table = Table(breakdown_data, colWidths=[7 * cm, 4 * cm, 4 * cm])
    breakdown_table.setStyle(TABLE_STYLE)
    elements.append(breakdown_table)
    elements.append(Spacer(1, 0.4 * cm))

    elements.append(Paragraph("Експертні оцінки (0-10)", SECTION_STYLE))
    qual_data = [
        ["Команда", "Traction", "Розмір ринку", "Конкуренція"],
        [
            _fmt(context.get("team_quality")),
            _fmt(context.get("traction")),
            _fmt(context.get("market_size")),
            _fmt(context.get("competition")),
        ],
    ]
    qual_table = Table(qual_data, colWidths=[3.75 * cm] * 4)
    qual_table.setStyle(TABLE_STYLE)
    elements.append(qual_table)

    if context.get("funding_ask") is not None:
        elements.append(Spacer(1, 0.4 * cm))
        elements.append(Paragraph("Умови угоди", SECTION_STYLE))
        deal_data = [
            ["Funding Ask ($M)", "Pre-money Valuation ($M)"],
            [_fmt(context.get("funding_ask")), _fmt(context.get("pre_money_valuation"))],
        ]
        deal_table = Table(deal_data, colWidths=[7.5 * cm] * 2)
        deal_table.setStyle(TABLE_STYLE)
        elements.append(deal_table)

    if context.get("details"):
        elements.extend(build_details_elements(context["details"]))

    return elements


def _section_table(table):
    """Details table with wrapping cells; the widest columns get the most room."""
    n = len(table["headers"])
    total = 17 * cm
    shares = {5: [0.06, 0.2, 0.3, 0.3, 0.14], 4: [0.2, 0.15, 0.45, 0.2]}.get(n, [1 / n] * n)
    rows = [[Paragraph(escape(str(h)), CELL_HEAD_STYLE) for h in table["headers"]]]
    rows += [[Paragraph(escape(str(c)), CELL_STYLE) for c in row] for row in table["rows"]]
    t = Table(rows, colWidths=[total * share for share in shares], repeatRows=1)
    t.setStyle(TABLE_STYLE)
    return t


def build_details_elements(details):
    """'Деталі розрахунку' on a new page (SPEC.md §4.4): the same sections as the screen."""
    elements = [PageBreak(), Paragraph("Деталі розрахунку", TITLE_STYLE), Spacer(1, 0.2 * cm)]
    for section in details_module.build_sections(details):
        elements.append(Paragraph(escape(section["title"]), SECTION_STYLE))
        for line in section["lines"]:
            elements.append(Paragraph("• " + escape(line), LINE_STYLE))
        if section["table"]:
            elements.append(Spacer(1, 0.15 * cm))
            elements.append(_section_table(section["table"]))
    return elements


def generate_pdf_report(context, output_path):
    """Render `context` (see build_report_elements) to a PDF at output_path."""
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    doc.build(build_report_elements(context))
    return output_path
