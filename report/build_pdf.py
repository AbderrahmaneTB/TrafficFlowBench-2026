"""Render REPORT_DRAFT.md (a small markdown subset) to TECHNICAL_REPORT.pdf with ReportLab.

Supported: '#'/'##'/'###' headings, paragraphs, '* '/'- ' bullets, '|' tables, **bold**, *italic*, `code`,
and images written as ![caption](fig/name.png). Run: python report/build_pdf.py
"""
import os
import re

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "REPORT_DRAFT.md")
OUT = os.path.join(HERE, "TECHNICAL_REPORT.pdf")

TTF = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
for name, f in (("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSans-Oblique.ttf"),
                ("DV-BI", "DejaVuSans-BoldOblique.ttf"), ("DVM", "DejaVuSansMono.ttf")):
    pdfmetrics.registerFont(TTFont(name, os.path.join(TTF, f)))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

INK, INK2 = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e")
BODY = ParagraphStyle("body", fontName="DV", fontSize=9, leading=12.5, textColor=INK, alignment=TA_JUSTIFY,
                      spaceAfter=5)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=7.8, leading=9.6, alignment=0, spaceAfter=0)
H = {1: ParagraphStyle("h1", fontName="DV-B", fontSize=15, leading=19, textColor=INK, spaceAfter=6),
     2: ParagraphStyle("h2", fontName="DV-B", fontSize=11.5, leading=15, textColor=INK, spaceBefore=9, spaceAfter=4),
     3: ParagraphStyle("h3", fontName="DV-B", fontSize=10, leading=13, textColor=INK, spaceBefore=6, spaceAfter=3)}
CAP = ParagraphStyle("cap", parent=BODY, fontSize=8, leading=10, textColor=INK2, alignment=1, spaceAfter=8)


def inline(t: str) -> str:
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"`([^`]+)`", r'<font face="DVM" size="8">\1</font>', t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*([^*\s][^*]*?)\*(?![\w*])", r"<i>\1</i>", t)
    return t


def table(rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [r for r in cells if not all(re.fullmatch(r":?-{2,}:?", c) for c in r)]
    data = [[Paragraph(("<b>%s</b>" if i == 0 else "%s") % inline(c), CELL) for c in r] for i, r in enumerate(cells)]
    n = max(len(r) for r in data)
    data = [r + [Paragraph("", CELL)] * (n - len(r)) for r in data]
    tb = Table(data, repeatRows=1, hAlign="LEFT")
    tb.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, 0), 0.6, INK2), ("LINEBELOW", (0, -1), (-1, -1), 0.6, INK2),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 3),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 3), ("TOPPADDING", (0, 0), (-1, -1), 2),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return tb


def build():
    lines = open(SRC, encoding="utf-8").read().splitlines()
    story, para, bullets, trows = [], [], [], []

    def flush():
        nonlocal para, bullets, trows
        if para:
            story.append(Paragraph(inline(" ".join(para)), BODY)); para = []
        if bullets:
            story.append(ListFlowable([ListItem(Paragraph(inline(b), BODY), leftIndent=10) for b in bullets],
                                      bulletType="bullet", start="•", leftIndent=12, bulletFontName="DV"))
            bullets = []
        if trows:
            story.append(table(trows)); story.append(Spacer(1, 6)); trows = []

    for ln in lines:
        s = ln.rstrip()
        m = re.match(r"^(#{1,3}) (.*)", s)
        img = re.match(r"^!\[(.*)\]\((.*)\)", s)
        if m:
            flush(); story.append(Paragraph(inline(m.group(2)), H[len(m.group(1))]))
        elif img:
            flush()
            path = os.path.join(HERE, img.group(2))
            im = Image(path); w = 160 * mm
            im.drawHeight = im.imageHeight * w / im.imageWidth; im.drawWidth = w
            if im.drawHeight > 95 * mm:
                im.drawWidth *= 95 * mm / im.drawHeight; im.drawHeight = 95 * mm
            story.append(im); story.append(Paragraph(inline(img.group(1)), CAP))
        elif s.startswith("|"):
            if (para or bullets) and not trows:
                flush()
            trows.append(s)
        elif re.match(r"^\s*[*-] ", s):
            if para or trows:
                flush()
            bullets.append(re.sub(r"^\s*[*-] ", "", s))
        elif not s.strip():
            flush()
        else:
            if bullets and s.startswith("  "):
                bullets[-1] += " " + s.strip()
            else:
                if trows or bullets:
                    flush()
                para.append(s.strip())
    flush()
    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title="TrafficFlowBench 2026 — Team Abderahmane_tb",
                            author="Team Abderahmane_tb")

    def page(canvas, d):
        canvas.saveState(); canvas.setFont("DV", 7.5); canvas.setFillColor(INK2)
        canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"{d.page}"); canvas.restoreState()

    doc.build(story, onFirstPage=page, onLaterPages=page)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
