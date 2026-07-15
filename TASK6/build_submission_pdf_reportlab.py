"""Generate the final Chinese TASK6 PDF directly with an embedded Song font."""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


BASE_DIR = Path(__file__).resolve().parent
SOURCE = BASE_DIR / "analysis_report.md"
OUTPUT = BASE_DIR / "薛智鸣TASK6.pdf"
FONT = "SongtiSC"
FONT_FILE = "/System/Library/Fonts/Supplemental/Songti.ttc"
BODY_SIZE = 10.5
LEADING = BODY_SIZE * 1.5
BLUE = colors.HexColor("#1F4E79")
LIGHT_BLUE = colors.HexColor("#D9EAF7")

TABLE_TITLES = [
    "常见机器学习量化自变量因子",
    "常见机器学习量化应变量",
    "测试集各季度策略收益率",
    "模型测试集预测效果",
    "策略回测核心指标",
]

FIGURE_INTERPRETATIONS = {
    "测试集累计净值": (
        "图1显示，测试期内三种机器学习策略的净值整体均高于全市场等权基准。"
        "随机森林在2022年第一季度反弹时取得较高收益，期末净值约为1.005；尽管测试期很短，"
        "其下行幅度和最终表现仍优于岭回归与单棵决策树。"
    ),
    "各季度收益率": (
        "图2显示，三种模型在2021年第四季度和2022年第二季度均出现绝对亏损，但亏损幅度小于等权基准；"
        "在2022年第一季度，模型策略均取得正收益。测试期超额收益主要来自下跌季度的相对抗跌能力。"
    ),
    "模型预测效果对比": (
        "图3显示，随机森林的Rank IC最高，说明其对股票下一季度收益的截面排序能力最强；"
        "三种模型的方向准确率均较低，表明预测收益方向仍然困难。策略构建因此更适合使用相对排名，"
        "不宜把预测值直接解释为精确收益。"
    ),
}


def clean_text(text):
    text = text.replace("`", "").replace("**", "")
    text = text.replace("\\(", "").replace("\\)", "").replace("\\log", "log")
    return html.escape(text)


def make_styles():
    # Embed the regular simplified-Chinese face (TTC index 6) so the PDF is
    # portable and does not depend on the reader's CJK language packs.
    pdfmetrics.registerFont(TTFont(FONT, FONT_FILE, subfontIndex=6))
    return {
        "body": ParagraphStyle(
            "BodyCN",
            fontName=FONT,
            fontSize=BODY_SIZE,
            leading=LEADING,
            alignment=TA_JUSTIFY,
            firstLineIndent=BODY_SIZE * 2,
            spaceBefore=0,
            spaceAfter=0,
            wordWrap="CJK",
        ),
        "list": ParagraphStyle(
            "ListCN",
            fontName=FONT,
            fontSize=BODY_SIZE,
            leading=LEADING,
            alignment=TA_JUSTIFY,
            leftIndent=BODY_SIZE * 2,
            firstLineIndent=-BODY_SIZE * 1.2,
            bulletIndent=0,
            spaceBefore=0,
            spaceAfter=0,
            wordWrap="CJK",
        ),
        "h1": ParagraphStyle(
            "H1CN", fontName=FONT, fontSize=16, leading=24, textColor=BLUE,
            spaceBefore=10, spaceAfter=4, keepWithNext=True, wordWrap="CJK",
        ),
        "h2": ParagraphStyle(
            "H2CN", fontName=FONT, fontSize=14, leading=21, textColor=BLUE,
            spaceBefore=7, spaceAfter=2, keepWithNext=True, wordWrap="CJK",
        ),
        "h3": ParagraphStyle(
            "H3CN", fontName=FONT, fontSize=12, leading=18, textColor=BLUE,
            spaceBefore=5, spaceAfter=0, keepWithNext=True, wordWrap="CJK",
        ),
        "caption": ParagraphStyle(
            "CaptionCN", fontName=FONT, fontSize=BODY_SIZE, leading=LEADING,
            alignment=TA_CENTER, spaceBefore=0, spaceAfter=0, keepWithNext=True,
            wordWrap="CJK",
        ),
        "table": ParagraphStyle(
            "TableCN", fontName=FONT, fontSize=BODY_SIZE, leading=LEADING,
            alignment=TA_CENTER, wordWrap="CJK",
        ),
        "table_left": ParagraphStyle(
            "TableLeftCN", fontName=FONT, fontSize=BODY_SIZE, leading=LEADING,
            alignment=TA_LEFT, wordWrap="CJK",
        ),
    }


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONT, 9)
    canvas.setFillColor(colors.HexColor("#666666"))
    if doc.page > 1:
        canvas.drawCentredString(A4[0] / 2, A4[1] - 1.45 * cm, "机器学习量化交易策略研究")
    canvas.drawCentredString(A4[0] / 2, 1.25 * cm, str(doc.page))
    canvas.restoreState()


def build_table(rows, number, styles):
    ncols = len(rows[0])
    width = A4[0] - 5.08 * cm
    fractions = {
        3: [0.18, 0.42, 0.40],
        4: [0.19, 0.38, 0.22, 0.21],
        5: [0.19, 0.19, 0.19, 0.19, 0.24],
        6: [0.18, 0.16, 0.16, 0.16, 0.17, 0.17],
        7: [0.17, 0.14, 0.14, 0.14, 0.13, 0.14, 0.14],
    }.get(ncols, [1 / ncols] * ncols)
    col_widths = [width * f for f in fractions]
    content = []
    for r_idx, row in enumerate(rows):
        content.append([
            Paragraph(clean_text(value), styles["table_left"] if c_idx == 0 and r_idx else styles["table"])
            for c_idx, value in enumerate(row)
        ])
    table = Table(content, colWidths=col_widths, repeatRows=1, hAlign="CENTER")
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("FONTSIZE", (0, 0), (-1, -1), BODY_SIZE),
        ("LEADING", (0, 0), (-1, -1), LEADING),
        ("BACKGROUND", (0, 0), (-1, 0), LIGHT_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#555555")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return [
        Paragraph(f"表{number}  {TABLE_TITLES[number - 1]}", styles["caption"]),
        table,
    ]


def build_story(styles):
    story = []
    story.extend([
        Spacer(1, 5.2 * cm),
        Paragraph("基于机器学习模型的量化交易策略", ParagraphStyle(
            "CoverTitle", fontName=FONT, fontSize=22, leading=33, alignment=TA_CENTER,
        )),
        Spacer(1, 0.4 * cm),
        Paragraph("TASK6 课程作业", ParagraphStyle(
            "CoverSub", fontName=FONT, fontSize=16, leading=24, alignment=TA_CENTER,
        )),
        Spacer(1, 5.0 * cm),
    ])
    cover_meta = ParagraphStyle("CoverMeta", fontName=FONT, fontSize=12, leading=24, alignment=TA_CENTER)
    story.extend([
        Paragraph("姓名：薛智鸣", cover_meta),
        Paragraph("研究方法：岭回归、决策树与随机森林", cover_meta),
        Paragraph("完成日期：2026年7月", cover_meta),
        PageBreak(),
    ])

    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    table_number = 0
    figure_number = 0
    i = 1
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[-:| ]+\|$", lines[i + 1].strip()):
            rows = [[clean_text(x.strip()) for x in stripped.strip("|").split("|")]]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([clean_text(x.strip()) for x in lines[i].strip().strip("|").split("|")])
                i += 1
            table_number += 1
            story.extend(build_table(rows, table_number, styles))
            continue
        image_match = re.match(r"!\[(.+?)\]\((.+?)\)", stripped)
        if image_match:
            figure_number += 1
            alt, rel_path = image_match.groups()
            img = Image(str(BASE_DIR / rel_path), width=15.5 * cm, height=9.3 * cm)
            img.hAlign = "CENTER"
            story.append(KeepTogether([
                img,
                Paragraph(f"图{figure_number}  {alt}", styles["caption"]),
                Paragraph("图形解读：" + FIGURE_INTERPRETATIONS[alt], styles["body"]),
            ]))
            i += 1
            continue
        heading = re.match(r"^(#{2,4})\s+(.+)$", stripped)
        if heading:
            if heading.group(2) == "六、图形":
                story.append(PageBreak())
            if heading.group(2) in FIGURE_INTERPRETATIONS:
                # The numbered caption beneath the image is the formal title;
                # suppress the duplicate Markdown subheading to avoid orphans.
                i += 1
                continue
            level = min(len(heading.group(1)) - 1, 3)
            story.append(Paragraph(clean_text(heading.group(2)), styles[f"h{level}"]))
            i += 1
            continue
        bullet = re.match(r"^-\s+(.+)$", stripped)
        numbered = re.match(r"^(\d+)\.\s+(.+)$", stripped)
        if bullet:
            story.append(Paragraph("● " + clean_text(bullet.group(1)), styles["list"]))
        elif numbered:
            story.append(Paragraph(numbered.group(1) + ". " + clean_text(numbered.group(2)), styles["list"]))
        else:
            story.append(Paragraph(clean_text(stripped), styles["body"]))
        i += 1
    return story


def main():
    styles = make_styles()
    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4,
        leftMargin=2.54 * cm, rightMargin=2.54 * cm,
        topMargin=2.3 * cm, bottomMargin=2.2 * cm,
        title="基于机器学习模型的量化交易策略",
        author="薛智鸣", subject="TASK6",
    )
    doc.build(build_story(styles), onFirstPage=header_footer, onLaterPages=header_footer)
    print(OUTPUT)


if __name__ == "__main__":
    main()
