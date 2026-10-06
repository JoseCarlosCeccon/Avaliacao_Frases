"""Gera o relatório final em PDF a partir do Markdown verificado."""

from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
DELIVERY = ROOT / "entrega-trabalho3"
SOURCE = DELIVERY / "relatorio-trabalho3.md"
OUTPUT = DELIVERY / "relatorio-trabalho3.pdf"

GREEN = colors.HexColor("#168821")
DARK_GREEN = colors.HexColor("#0B5D1E")
LIGHT_GREEN = colors.HexColor("#E9F5EB")
DARK = colors.HexColor("#1E293B")
MID = colors.HexColor("#475569")
LIGHT = colors.HexColor("#F1F5F9")
GRID = colors.HexColor("#CBD5E1")


def register_fonts() -> tuple[str, str, str]:
    font_dir = Path("C:/Windows/Fonts")
    regular = font_dir / "segoeui.ttf"
    semibold = font_dir / "seguisb.ttf"
    bold = font_dir / "segoeuib.ttf"
    if regular.exists() and semibold.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("SegoeUI", regular))
        pdfmetrics.registerFont(TTFont("SegoeUISemibold", semibold))
        pdfmetrics.registerFont(TTFont("SegoeUIBold", bold))
        return "SegoeUI", "SegoeUISemibold", "SegoeUIBold"
    return "Helvetica", "Helvetica-Bold", "Helvetica-Bold"


REGULAR, SEMIBOLD, BOLD = register_fonts()


def inline(text: str) -> str:
    """Converte um subconjunto seguro de Markdown para marcação ReportLab."""

    value = escape(text)
    value = re.sub(r"`([^`]+)`", r"<font name='SegoeUISemibold'>\1</font>", value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    value = re.sub(
        r"(https://[^\s<]+)",
        r"<link href='\1' color='#0B5D1E'>\1</link>",
        value,
    )
    return value


def make_styles():
    styles = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "CoverTitle",
            parent=styles["Title"],
            fontName=BOLD,
            fontSize=28,
            leading=34,
            textColor=DARK,
            alignment=TA_LEFT,
            spaceAfter=12,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle",
            parent=styles["Normal"],
            fontName=REGULAR,
            fontSize=15,
            leading=21,
            textColor=MID,
            spaceAfter=18,
        ),
        "meta": ParagraphStyle(
            "Meta",
            parent=styles["Normal"],
            fontName=REGULAR,
            fontSize=10.5,
            leading=16,
            textColor=MID,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=styles["Heading1"],
            fontName=BOLD,
            fontSize=17,
            leading=22,
            textColor=DARK_GREEN,
            spaceBefore=16,
            spaceAfter=8,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=styles["Heading2"],
            fontName=SEMIBOLD,
            fontSize=13,
            leading=17,
            textColor=DARK,
            spaceBefore=12,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=styles["BodyText"],
            fontName=REGULAR,
            fontSize=9.7,
            leading=14.2,
            textColor=DARK,
            alignment=TA_JUSTIFY,
            spaceAfter=7,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=styles["BodyText"],
            fontName=REGULAR,
            fontSize=9.5,
            leading=13.5,
            leftIndent=14,
            firstLineIndent=-8,
            textColor=DARK,
            spaceAfter=4,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=styles["Normal"],
            fontName=SEMIBOLD,
            fontSize=8,
            leading=10,
            textColor=colors.white,
            alignment=TA_CENTER,
        ),
        "table_cell": ParagraphStyle(
            "TableCell",
            parent=styles["Normal"],
            fontName=REGULAR,
            fontSize=7.8,
            leading=10,
            textColor=DARK,
            alignment=TA_CENTER,
        ),
        "callout": ParagraphStyle(
            "Callout",
            parent=styles["BodyText"],
            fontName=SEMIBOLD,
            fontSize=11,
            leading=16,
            textColor=DARK_GREEN,
            alignment=TA_LEFT,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=styles["BodyText"],
            fontName=REGULAR,
            fontSize=8.2,
            leading=11,
            textColor=MID,
            spaceAfter=5,
        ),
    }


STYLES = make_styles()


class ReportDocTemplate(BaseDocTemplate):
    def __init__(self, filename: Path):
        super().__init__(
            str(filename),
            pagesize=A4,
            leftMargin=2.0 * cm,
            rightMargin=2.0 * cm,
            topMargin=1.9 * cm,
            bottomMargin=1.8 * cm,
            title="Trabalho 3 - Busca semântica com ChromaDB",
            author="Trabalho acadêmico",
            subject="Comparação entre all-MiniLM-L6-v2 e all-mpnet-base-v2",
        )
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="content",
        )
        self.addPageTemplates(PageTemplate(id="report", frames=frame, onPage=draw_page))


def draw_page(canvas, doc):
    canvas.saveState()
    width, height = A4
    page_number = canvas.getPageNumber()
    if page_number > 1:
        canvas.setStrokeColor(GREEN)
        canvas.setLineWidth(1.2)
        canvas.line(doc.leftMargin, height - 1.25 * cm, width - doc.rightMargin, height - 1.25 * cm)
        canvas.setFont(SEMIBOLD, 8)
        canvas.setFillColor(DARK_GREEN)
        canvas.drawString(doc.leftMargin, height - 1.0 * cm, "TRABALHO 3 | CHROMADB")
    canvas.setFont(REGULAR, 8)
    canvas.setFillColor(MID)
    canvas.drawCentredString(width / 2, 0.85 * cm, f"Página {page_number}")
    canvas.restoreState()


def cover_story() -> list:
    metric_data = [
        [Paragraph("MODELO VENCEDOR", STYLES["table_header"]), Paragraph("RESULTADO", STYLES["table_header"])],
        [Paragraph("all-mpnet-base-v2", STYLES["table_cell"]), Paragraph("Recall@5: 0,0833 | MRR: 0,20", STYLES["table_cell"])],
    ]
    metric_table = Table(metric_data, colWidths=[6.2 * cm, 8.2 * cm], rowHeights=[0.75 * cm, 1.0 * cm])
    metric_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), DARK_GREEN),
                ("BACKGROUND", (0, 1), (-1, -1), LIGHT_GREEN),
                ("BOX", (0, 0), (-1, -1), 0.8, GREEN),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, GRID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return [
        Spacer(1, 2.3 * cm),
        Table([[""]], colWidths=[3.2 * cm], rowHeights=[0.16 * cm], style=[("BACKGROUND", (0, 0), (-1, -1), GREEN)]),
        Spacer(1, 0.45 * cm),
        Paragraph("Trabalho 3", STYLES["cover_title"]),
        Paragraph("Busca semântica com ChromaDB", STYLES["cover_subtitle"]),
        Paragraph(
            "Comparação entre os modelos <b>all-MiniLM-L6-v2</b> e "
            "<b>all-mpnet-base-v2</b> em um corpus tecnológico em português.",
            STYLES["body"],
        ),
        Spacer(1, 1.0 * cm),
        metric_table,
        Spacer(1, 1.3 * cm),
        Table(
            [
                [Paragraph("Disciplina", STYLES["meta"]), Paragraph("Tópicos Especiais em Banco de Dados", STYLES["meta"])],
                [Paragraph("Professor", STYLES["meta"]), Paragraph("Leonardo Fiório Soares", STYLES["meta"])],
                [Paragraph("Corpus", STYLES["meta"]), Paragraph("1.000 frases únicas", STYLES["meta"])],
                [Paragraph("Data", STYLES["meta"]), Paragraph("25 de setembro de 2026", STYLES["meta"])],
            ],
            colWidths=[3.2 * cm, 11.2 * cm],
            style=[
                ("BACKGROUND", (0, 0), (0, -1), LIGHT),
                ("BOX", (0, 0), (-1, -1), 0.5, GRID),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, GRID),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ],
        ),
        Spacer(1, 1.25 * cm),
        Table(
            [[Paragraph("Conclusão: MPNet venceu relativamente, mas ambos os modelos apresentaram baixa adequação ao português.", STYLES["callout"])]],
            colWidths=[14.4 * cm],
            style=[
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GREEN),
                ("LINEBEFORE", (0, 0), (0, -1), 4, GREEN),
                ("LEFTPADDING", (0, 0), (-1, -1), 14),
                ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
            ],
        ),
        PageBreak(),
    ]


def markdown_table(rows: list[str]):
    parsed = []
    for index, row in enumerate(rows):
        cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
        if index == 1 and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        style = STYLES["table_header"] if not parsed else STYLES["table_cell"]
        parsed.append([Paragraph(inline(cell), style) for cell in cells])
    columns = len(parsed[0])
    if columns == 5:
        first_header = rows[0].strip().strip("|").split("|")[0].strip()
        if first_header == "Modelo":
            widths = [3.3 * cm, 4.1 * cm, 2.1 * cm, 1.8 * cm, 3.1 * cm]
        else:
            widths = [0.8 * cm, 6.7 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm]
    elif columns == 2:
        widths = [6.8 * cm, 7.6 * cm]
    else:
        widths = [14.4 * cm / columns] * columns
    table = Table(parsed, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), DARK_GREEN),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("GRID", (0, 0), (-1, -1), 0.35, GRID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def body_story(markdown: str) -> list:
    lines = markdown.splitlines()
    start = next(index for index, line in enumerate(lines) if line == "## Resumo")
    lines = lines[start:]
    story: list = []
    paragraph_lines: list[str] = []
    in_references = False

    def flush_paragraph():
        if paragraph_lines:
            text = " ".join(part.strip() for part in paragraph_lines)
            story.append(Paragraph(inline(text), STYLES["body"]))
            paragraph_lines.clear()

    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        if not line:
            flush_paragraph()
            index += 1
            continue
        if line.startswith("## "):
            flush_paragraph()
            heading = line[3:]
            in_references = heading == "Referências"
            story.append(Paragraph(inline(heading), STYLES["h1"]))
            index += 1
            continue
        if line.startswith("### "):
            flush_paragraph()
            story.append(Paragraph(inline(line[4:]), STYLES["h2"]))
            index += 1
            continue
        if line.startswith("- "):
            flush_paragraph()
            story.append(Paragraph("• " + inline(line[2:]), STYLES["bullet"]))
            index += 1
            continue
        if line.startswith("|"):
            flush_paragraph()
            table_rows = []
            while index < len(lines) and lines[index].startswith("|"):
                table_rows.append(lines[index])
                index += 1
            story.extend([Spacer(1, 0.15 * cm), markdown_table(table_rows), Spacer(1, 0.3 * cm)])
            continue
        if in_references and re.match(r"^\d+\. ", line):
            flush_paragraph()
            story.append(Paragraph(inline(line), STYLES["small"]))
            index += 1
            continue
        paragraph_lines.append(line)
        index += 1
    flush_paragraph()
    return story


def build_pdf() -> None:
    markdown = SOURCE.read_text(encoding="utf-8")
    story = cover_story() + body_story(markdown)
    ReportDocTemplate(OUTPUT).build(story)


if __name__ == "__main__":
    build_pdf()
    print(OUTPUT)
