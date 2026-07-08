# -*- coding: utf-8 -*-
"""Build the final TASK3 submission PDF.

The report uses Songti, 10.5 pt body text, 1.5 line spacing, zero paragraph
spacing, and justified body paragraphs to match the submission requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
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
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
FIGURE_DIR = BASE_DIR / "figures"
PDF_PATH = BASE_DIR / "薛智鸣TASK3.pdf"

SONGTI_PATH = Path("/System/Library/Fonts/Supplemental/Songti.ttc")
FONT_NAME = "Songti"
BODY_SIZE = 10.5
BODY_LEADING = BODY_SIZE * 1.5


@dataclass(frozen=True)
class FigureChoice:
    ts_code: str
    stock_name: str
    short_window: int
    long_window: int
    title: str


def register_fonts() -> None:
    if SONGTI_PATH.exists():
        pdfmetrics.registerFont(TTFont(FONT_NAME, str(SONGTI_PATH), subfontIndex=0))
    else:
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont

        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        globals()["FONT_NAME"] = "STSong-Light"


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def decimal(value: float) -> str:
    return f"{value:.4f}"


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def build_styles() -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    base = ParagraphStyle(
        "BodyCN",
        parent=styles["Normal"],
        fontName=FONT_NAME,
        fontSize=BODY_SIZE,
        leading=BODY_LEADING,
        firstLineIndent=BODY_SIZE * 2,
        alignment=TA_JUSTIFY,
        spaceBefore=0,
        spaceAfter=0,
        wordWrap="CJK",
    )
    no_indent = ParagraphStyle(
        "BodyNoIndentCN",
        parent=base,
        firstLineIndent=0,
    )
    title = ParagraphStyle(
        "TitleCN",
        parent=base,
        fontSize=16,
        leading=24,
        firstLineIndent=0,
        alignment=TA_CENTER,
    )
    subtitle = ParagraphStyle(
        "SubtitleCN",
        parent=base,
        fontSize=BODY_SIZE,
        leading=BODY_LEADING,
        firstLineIndent=0,
        alignment=TA_CENTER,
    )
    heading1 = ParagraphStyle(
        "Heading1CN",
        parent=base,
        fontSize=12,
        leading=18,
        firstLineIndent=0,
        alignment=TA_LEFT,
    )
    heading2 = ParagraphStyle(
        "Heading2CN",
        parent=base,
        fontSize=BODY_SIZE,
        leading=BODY_LEADING,
        firstLineIndent=0,
        alignment=TA_LEFT,
    )
    caption = ParagraphStyle(
        "CaptionCN",
        parent=base,
        firstLineIndent=0,
        alignment=TA_CENTER,
    )
    table_cell = ParagraphStyle(
        "TableCellCN",
        parent=base,
        fontSize=BODY_SIZE,
        leading=BODY_LEADING,
        firstLineIndent=0,
        alignment=TA_CENTER,
    )
    table_left = ParagraphStyle(
        "TableLeftCN",
        parent=table_cell,
        alignment=TA_LEFT,
        wordWrap="CJK",
    )
    footer = ParagraphStyle(
        "FooterCN",
        parent=no_indent,
        fontSize=9,
        leading=12,
        alignment=TA_CENTER,
    )
    return {
        "body": base,
        "no_indent": no_indent,
        "title": title,
        "subtitle": subtitle,
        "h1": heading1,
        "h2": heading2,
        "caption": caption,
        "table_cell": table_cell,
        "table_left": table_left,
        "footer": footer,
    }


def add_page_number(canvas, doc) -> None:  # noqa: ANN001
    canvas.saveState()
    canvas.setFont(FONT_NAME, 9)
    canvas.drawCentredString(A4[0] / 2, 1.25 * cm, f"{doc.page}")
    canvas.restoreState()


def make_doc() -> BaseDocTemplate:
    doc = BaseDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=2.54 * cm,
        leftMargin=2.54 * cm,
        topMargin=2.54 * cm,
        bottomMargin=2.54 * cm,
        title="薛智鸣TASK3",
        author="薛智鸣",
        subject="双均线策略与量化回测分析",
    )
    frame = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        doc.width,
        doc.height,
        leftPadding=0,
        bottomPadding=0,
        rightPadding=0,
        topPadding=0,
    )
    doc.addPageTemplates([PageTemplate(id="Normal", frames=[frame], onPage=add_page_number)])
    return doc


def load_metrics() -> pd.DataFrame:
    metrics_path = OUTPUT_DIR / "strategy_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(
            "Missing TASK3/output/strategy_metrics.csv. Run analyze_double_ma_strategy.py first."
        )
    return pd.read_csv(metrics_path)


def metric_lookup(metrics: pd.DataFrame, choice: FigureChoice) -> pd.Series:
    row = metrics[
        (metrics["ts_code"] == choice.ts_code)
        & (metrics["short_window"] == choice.short_window)
        & (metrics["long_window"] == choice.long_window)
    ]
    if row.empty:
        raise ValueError(f"Missing metrics for {choice.ts_code} {choice.short_window}/{choice.long_window}")
    return row.iloc[0]


def result_table(metrics: pd.DataFrame, styles: dict[str, ParagraphStyle]) -> Table:
    headers = ["股票", "周期", "策略累计回报", "买入持有回报", "最大回撤", "夏普比率", "信号次数"]
    rows: list[list[Paragraph]] = [[p(item, styles["table_cell"]) for item in headers]]
    sorted_metrics = metrics.sort_values(["stock_name", "short_window", "long_window"])
    for _, row in sorted_metrics.iterrows():
        rows.append(
            [
                p(str(row["stock_name"]), styles["table_cell"]),
                p(f"{int(row['short_window'])}/{int(row['long_window'])}", styles["table_cell"]),
                p(pct(float(row["cumulative_return"])), styles["table_cell"]),
                p(pct(float(row["benchmark_return"])), styles["table_cell"]),
                p(pct(float(row["max_drawdown"])), styles["table_cell"]),
                p(decimal(float(row["sharpe_ratio"])), styles["table_cell"]),
                p(str(int(row["trade_signals"])), styles["table_cell"]),
            ]
        )

    table = Table(
        rows,
        colWidths=[2.1 * cm, 1.6 * cm, 2.3 * cm, 2.3 * cm, 2.0 * cm, 1.9 * cm, 1.8 * cm],
        repeatRows=1,
        hAlign="CENTER",
    )
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
                ("FONTSIZE", (0, 0), (-1, -1), BODY_SIZE),
                ("LEADING", (0, 0), (-1, -1), BODY_LEADING),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.93, 0.93, 0.93)),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def best_choice_rows(metrics: pd.DataFrame) -> list[FigureChoice]:
    rows = (
        metrics.sort_values(["stock_name", "cumulative_return"], ascending=[True, False])
        .groupby("stock_name", as_index=False)
        .head(1)
        .sort_values("ts_code")
    )
    choices = []
    for _, row in rows.iterrows():
        choices.append(
            FigureChoice(
                ts_code=str(row["ts_code"]),
                stock_name=str(row["stock_name"]),
                short_window=int(row["short_window"]),
                long_window=int(row["long_window"]),
                title=f"{row['stock_name']}双均线策略回测图（{int(row['short_window'])}/{int(row['long_window'])}）",
            )
        )
    return choices


def all_figure_choices(metrics: pd.DataFrame) -> list[FigureChoice]:
    choices = []
    rows = metrics.sort_values(["stock_name", "short_window", "long_window"])
    for _, row in rows.iterrows():
        choices.append(
            FigureChoice(
                ts_code=str(row["ts_code"]),
                stock_name=str(row["stock_name"]),
                short_window=int(row["short_window"]),
                long_window=int(row["long_window"]),
                title=f"{row['stock_name']}双均线策略回测图（{int(row['short_window'])}/{int(row['long_window'])}）",
            )
        )
    return choices


def figure_path(choice: FigureChoice) -> Path:
    return FIGURE_DIR / f"{choice.ts_code}_ma_{choice.short_window}_{choice.long_window}_strategy.png"


def figure_interpretation(row: pd.Series, choice: FigureChoice) -> str:
    comparison = "跑赢" if row["excess_return_vs_benchmark"] > 0 else "跑输"
    signal_comment = "信号较频繁，说明短均线对价格波动更敏感" if row["trade_signals"] >= 10 else "信号较少，说明该参数组合更偏向过滤短期噪声"
    return (
        f"解读：红色上三角表示买入信号，黑色下三角表示卖出信号。"
        f"该组合策略累计回报为{pct(float(row['cumulative_return']))}，"
        f"买入持有回报为{pct(float(row['benchmark_return']))}，"
        f"{comparison}基准{pct(abs(float(row['excess_return_vs_benchmark'])))}；"
        f"最大回撤为{pct(float(row['max_drawdown']))}，夏普比率为{decimal(float(row['sharpe_ratio']))}。"
        f"本图共出现{int(row['trade_signals'])}次交易信号，{signal_comment}。"
    )


def append_figure(
    story: list,
    styles: dict[str, ParagraphStyle],
    metrics: pd.DataFrame,
    choice: FigureChoice,
    figure_number: int,
) -> None:
    path = figure_path(choice)
    if not path.exists():
        raise FileNotFoundError(path)
    row = metric_lookup(metrics, choice)

    story.append(PageBreak())
    story.append(p(f"图{figure_number} {choice.title}", styles["caption"]))
    story.append(Spacer(1, 0.15 * cm))
    img = Image(str(path), width=15.5 * cm, height=11.1 * cm)
    img.hAlign = "CENTER"
    story.append(img)
    story.append(Spacer(1, 0.15 * cm))
    story.append(p(figure_interpretation(row, choice), styles["body"]))


def build_story(metrics: pd.DataFrame, styles: dict[str, ParagraphStyle]) -> list:
    story: list = []
    best = metrics.sort_values("cumulative_return", ascending=False).iloc[0]
    worst_mdd = metrics.sort_values("max_drawdown").iloc[0]

    story.append(p("双均线策略与量化回测分析", styles["title"]))
    story.append(p("薛智鸣 TASK3", styles["subtitle"]))
    story.append(Spacer(1, 0.3 * cm))

    story.append(p("一、双均线策略与金叉、死叉", styles["h1"]))
    story.append(
        p(
            "双均线策略是一类典型的趋势跟随策略。它同时计算短周期均线和长周期均线，"
            "短均线对近期价格变化更敏感，长均线更平滑，因此可以用二者的相对位置来判断价格趋势。"
            "当短均线位于长均线上方时，策略认为近期走势偏强；当短均线位于长均线下方时，策略认为近期走势偏弱。",
            styles["body"],
        )
    )
    story.append(
        p(
            "金叉指短均线从下方向上穿过长均线，通常代表短期趋势开始强于长期趋势，"
            "在双均线策略中常被视为买入或开仓信号。死叉指短均线从上方向下穿过长均线，"
            "通常代表短期趋势开始弱于长期趋势，常被视为卖出或离场信号。需要注意的是，均线本身具有滞后性，"
            "在震荡行情中可能频繁出现假信号。",
            styles["body"],
        )
    )

    story.append(p("二、量化策略效果的基础指标", styles["h1"]))
    story.append(
        p(
            "累计回报用于衡量策略从回测开始到结束的总收益，计算公式为："
            "累计回报 = 最终资产 / 初始资产 - 1。它直观反映策略整体是否赚钱，"
            "但不能单独说明策略过程中的风险。",
            styles["body"],
        )
    )
    story.append(
        p(
            "最大回撤（MDD）用于衡量资金曲线从历史高点到之后低点的最大跌幅，计算公式为："
            "回撤 = 当前资产 / 历史最高资产 - 1，最大回撤取所有回撤中的最小值。"
            "该指标越接近0，说明策略历史最大亏损幅度越小；例如-20%表示资金从阶段高点最多回落过20%。",
            styles["body"],
        )
    )
    story.append(
        p(
            "夏普比率用于衡量单位波动下获得的超额收益。本次回测使用日收益率并按252个交易日年化，"
            "默认无风险利率为0。夏普比率越高，说明收益相对波动更有效率，但它依赖历史样本，"
            "也不能完全反映极端行情下的尾部风险。",
            styles["body"],
        )
    )

    story.append(p("三、Python实现方法", styles["h1"]))
    story.append(
        p(
            "本次实现读取项目中已经存储的三份日线行情CSV数据，兼容YYYYMMDD和YYYY-MM-DD两种日期格式。"
            "脚本先计算短周期均线和长周期均线，再根据短均线与长均线的穿越关系生成交易信号。"
            "当短均线上穿长均线时记为买入信号，当短均线下穿长均线时记为卖出信号。",
            styles["body"],
        )
    )
    story.append(
        p(
            "回测采用只做多、不做空的设定，初始资金为100000元，单边交易成本为0.10%。"
            "为避免前视偏差，本次假设当天收盘后确认金叉或死叉，下一交易日才改变持仓。"
            "测试参数包括5/15、10/30和20/60三组均线周期。",
            styles["body"],
        )
    )

    story.append(PageBreak())
    story.append(p("四、不同股票和均线周期的回测结果", styles["h1"]))
    story.append(p("表1 不同股票与均线周期回测指标汇总", styles["caption"]))
    story.append(result_table(metrics, styles))
    story.append(
        p(
            f"从表1可以看到，累计回报最高的是{best['stock_name']}（{best['ts_code']}）"
            f"的{int(best['short_window'])}/{int(best['long_window'])}组合，策略累计回报为"
            f"{pct(float(best['cumulative_return']))}。最大回撤最深的是{worst_mdd['stock_name']}"
            f"（{worst_mdd['ts_code']}）的{int(worst_mdd['short_window'])}/{int(worst_mdd['long_window'])}"
            f"组合，最大回撤为{pct(float(worst_mdd['max_drawdown']))}。整体来看，"
            "不同股票对均线参数较为敏感，同一组参数不能直接套用到所有股票。",
            styles["body"],
        )
    )

    story.append(p("五、图表分析", styles["h1"]))
    story.append(
        p(
            "下面每张图均包含收盘价、短均线、长均线、买入信号、卖出信号、策略资金曲线和回撤曲线。"
            "其中红色上三角代表买入信号，黑色下三角代表卖出信号。",
            styles["body"],
        )
    )
    for index, choice in enumerate(all_figure_choices(metrics), start=1):
        append_figure(story, styles, metrics, choice, index)

    story.append(PageBreak())
    story.append(p("六、双均线策略适用场景与应用心得", styles["h1"]))
    story.append(
        p(
            "双均线策略更适合趋势较明确、价格沿一个方向持续运行的市场环境。"
            "在这类行情中，金叉后价格往往能够继续上涨，策略有机会利用趋势获得收益；"
            "死叉后及时离场，也有助于控制下跌阶段的损失。",
            styles["body"],
        )
    )
    story.append(
        p(
            "短周期组合反应更快，能够更早进入趋势，也会更早离场，但在震荡市中更容易产生反复交易。"
            "长周期组合信号更少，抗噪声能力较强，但进入和退出都更滞后，可能错过趋势初期收益。"
            "因此，均线周期的选择需要结合股票波动特征、市场环境和交易成本综合考虑。",
            styles["body"],
        )
    )
    story.append(
        p(
            "从本次实验看，双均线策略不能简单理解为“金叉必涨、死叉必跌”。"
            "它更像是一种规则清晰、容易实现的趋势过滤工具。实际应用中还应结合成交量、波动率、"
            "止损规则和样本外检验，避免只根据历史收益进行过度调参。",
            styles["body"],
        )
    )
    return story


def main() -> None:
    register_fonts()
    metrics = load_metrics()
    styles = build_styles()
    doc = make_doc()
    story = build_story(metrics, styles)
    doc.build(story)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
