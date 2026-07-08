# -*- coding: utf-8 -*-
"""Build the final TASK4 submission PDF.

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
PDF_PATH = BASE_DIR / "薛智鸣TASK4.pdf"

SONGTI_PATH = Path("/System/Library/Fonts/Supplemental/Songti.ttc")
FONT_NAME = "Songti"
BODY_SIZE = 10.5
BODY_LEADING = BODY_SIZE * 1.5


@dataclass(frozen=True)
class FigureChoice:
    ts_code: str
    stock_name: str
    entry_window: int
    exit_window: int
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
    if pd.isna(value):
        return ""
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
        "caption": caption,
        "table_cell": table_cell,
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
        title="薛智鸣TASK4",
        author="薛智鸣",
        subject="海龟交易法则与通道突破策略回测分析",
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
            "Missing TASK4/output/strategy_metrics.csv. Run analyze_turtle_strategy.py first."
        )
    return pd.read_csv(metrics_path)


def metric_lookup(metrics: pd.DataFrame, choice: FigureChoice) -> pd.Series:
    row = metrics[
        (metrics["ts_code"] == choice.ts_code)
        & (metrics["entry_window"] == choice.entry_window)
        & (metrics["exit_window"] == choice.exit_window)
    ]
    if row.empty:
        raise ValueError(f"Missing metrics for {choice.ts_code} {choice.entry_window}/{choice.exit_window}")
    return row.iloc[0]


def result_table(metrics: pd.DataFrame, styles: dict[str, ParagraphStyle]) -> Table:
    headers = ["股票", "通道", "策略累计回报", "买入持有回报", "最大回撤", "夏普比率", "信号次数"]
    rows: list[list[Paragraph]] = [[p(item, styles["table_cell"]) for item in headers]]
    sorted_metrics = metrics.sort_values(["stock_name", "entry_window", "exit_window"])
    for _, row in sorted_metrics.iterrows():
        rows.append(
            [
                p(str(row["stock_name"]), styles["table_cell"]),
                p(f"{int(row['entry_window'])}/{int(row['exit_window'])}", styles["table_cell"]),
                p(pct(float(row["cumulative_return"])), styles["table_cell"]),
                p(pct(float(row["benchmark_return"])), styles["table_cell"]),
                p(pct(float(row["max_drawdown"])), styles["table_cell"]),
                p(decimal(float(row["sharpe_ratio"])), styles["table_cell"]),
                p(str(int(row["trade_signals"])), styles["table_cell"]),
            ]
        )

    table = Table(
        rows,
        colWidths=[2.1 * cm, 1.6 * cm, 2.4 * cm, 2.4 * cm, 2.0 * cm, 1.9 * cm, 1.6 * cm],
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


def all_figure_choices(metrics: pd.DataFrame) -> list[FigureChoice]:
    choices = []
    rows = metrics.sort_values(["stock_name", "entry_window", "exit_window"])
    for _, row in rows.iterrows():
        choices.append(
            FigureChoice(
                ts_code=str(row["ts_code"]),
                stock_name=str(row["stock_name"]),
                entry_window=int(row["entry_window"]),
                exit_window=int(row["exit_window"]),
                title=(
                    f"{row['stock_name']}海龟策略回测图"
                    f"（{int(row['entry_window'])}/{int(row['exit_window'])}通道）"
                ),
            )
        )
    return choices


def figure_path(choice: FigureChoice) -> Path:
    return FIGURE_DIR / f"{choice.ts_code}_turtle_{choice.entry_window}_{choice.exit_window}_strategy.png"


def figure_interpretation(row: pd.Series) -> str:
    comparison = "跑赢" if row["excess_return_vs_benchmark"] > 0 else "跑输"
    if row["trade_signals"] == 0:
        signal_comment = "该组合在样本期内没有触发突破信号，说明通道设置较长或价格未形成足够强的新高突破。"
    elif row["trade_signals"] >= 10:
        signal_comment = "信号较多，说明该参数对价格波动更敏感，也更容易受到震荡行情影响。"
    else:
        signal_comment = "信号相对较少，说明该参数组合更偏向过滤短期噪声。"

    return (
        f"解读：蓝线为收盘价，绿色线为入场高点通道，橙色线为离场低点通道，紫色虚线为ATR移动止损线；"
        f"红色上三角表示买入信号，黑色下三角表示卖出信号。"
        f"该组合策略累计回报为{pct(float(row['cumulative_return']))}，"
        f"买入持有回报为{pct(float(row['benchmark_return']))}，"
        f"{comparison}基准{pct(abs(float(row['excess_return_vs_benchmark'])))}；"
        f"最大回撤为{pct(float(row['max_drawdown']))}，夏普比率为{decimal(float(row['sharpe_ratio']))}。"
        f"本图共出现{int(row['trade_signals'])}次交易信号，其中ATR止损{int(row['atr_stop_exits'])}次，"
        f"低通道离场{int(row['channel_exits'])}次。{signal_comment}"
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
    story.append(Spacer(1, 0.12 * cm))
    img = Image(str(path), width=15.5 * cm, height=13.3 * cm)
    img.hAlign = "CENTER"
    story.append(img)
    story.append(Spacer(1, 0.12 * cm))
    story.append(p(figure_interpretation(row), styles["body"]))


def build_story(metrics: pd.DataFrame, styles: dict[str, ParagraphStyle]) -> list:
    story: list = []
    best = metrics.sort_values("cumulative_return", ascending=False).iloc[0]
    worst_mdd = metrics.sort_values("max_drawdown").iloc[0]

    story.append(p("海龟交易法则与通道突破策略回测分析", styles["title"]))
    story.append(p("薛智鸣 TASK4", styles["subtitle"]))
    story.append(Spacer(1, 0.3 * cm))

    story.append(p("一、海龟策略的核心思想与关键优势", styles["h1"]))
    story.append(
        p(
            "海龟交易法则是一套经典的趋势跟随策略。它的核心不是预测明天涨跌，"
            "而是用价格突破来确认趋势已经出现：当价格突破过去一段时间的最高价通道时买入；"
            "当价格跌破较短周期的最低价通道，或触及按波动率设置的止损位时离场。",
            styles["body"],
        )
    )
    story.append(
        p(
            "该策略强调顺势而为、截断亏损、让利润奔跑和规则化执行。"
            "它通过高低点通道识别趋势启动，用ATR衡量市场波动并设置止损距离，"
            "从而减少主观判断对交易行为的影响。其关键优势在于方法透明、指标少、易复现，"
            "同时具备较明确的风险控制框架，适合在不同股票或不同参数下进行系统回测。",
            styles["body"],
        )
    )

    story.append(p("二、高低点通道、ATR与止损条件", styles["h1"]))
    story.append(
        p(
            "高低点通道又称Donchian Channel。本次实现中，为避免使用当天价格造成前视偏差，"
            "高点通道和低点通道均使用昨日以前的数据计算。入场高点通道等于过去N个交易日最高价的最大值，"
            "离场低点通道等于过去M个交易日最低价的最小值。经典海龟法则常用20日突破入场、10日低点离场，"
            "或55日突破入场、20日低点离场。",
            styles["body"],
        )
    )
    story.append(
        p(
            "ATR即Average True Range，中文为平均真实波幅，用于衡量股票的真实波动幅度。"
            "真实波幅TR取三个值中的最大者：当日最高价减最低价、当日最高价与昨日收盘价差值的绝对值、"
            "当日最低价与昨日收盘价差值的绝对值。本次默认对TR做20日Wilder平滑得到ATR。",
            styles["body"],
        )
    )
    story.append(
        p(
            "本次策略采用三类交易条件：第一，收盘价向上突破入场高点通道时买入；"
            "第二，收盘价跌破离场低点通道时卖出；第三，收盘价跌破ATR移动止损线时止损卖出。"
            "ATR移动止损线按照持仓以来最高收盘价减去2倍ATR计算，趋势继续上涨时止损线会随之抬高。",
            styles["body"],
        )
    )

    story.append(p("三、Python实现与回测设定", styles["h1"]))
    story.append(
        p(
            "本次实现读取项目中已经存储的三份日线行情CSV数据，兼容YYYYMMDD和YYYY-MM-DD两种日期格式。"
            "脚本先计算高点通道、低点通道、TR和ATR，再根据突破、低通道离场和ATR止损生成买入卖出信号。"
            "随后绘制股价、通道、交易信号、ATR、策略资金曲线和回撤曲线，并输出逐日回测结果与汇总指标。",
            styles["body"],
        )
    )
    story.append(
        p(
            "回测采用只做多、不做空的设定，初始资金为100000元，单边交易成本为0.10%。"
            "为避免前视偏差，本次假设当天收盘后确认信号，下一交易日才改变持仓。"
            "测试参数包括20/10、55/20和10/5三组入场/离场通道，ATR周期为20，ATR止损倍数为2。",
            styles["body"],
        )
    )

    story.append(PageBreak())
    story.append(p("四、不同股票和通道周期的回测结果", styles["h1"]))
    story.append(p("表1 不同股票与通道参数回测指标汇总", styles["caption"]))
    story.append(result_table(metrics, styles))
    story.append(
        p(
            f"从表1可以看到，累计回报最高的是{best['stock_name']}（{best['ts_code']}）"
            f"的{int(best['entry_window'])}/{int(best['exit_window'])}通道组合，策略累计回报为"
            f"{pct(float(best['cumulative_return']))}。最大回撤最深的是{worst_mdd['stock_name']}"
            f"（{worst_mdd['ts_code']}）的{int(worst_mdd['entry_window'])}/{int(worst_mdd['exit_window'])}"
            f"通道组合，最大回撤为{pct(float(worst_mdd['max_drawdown']))}。整体来看，"
            "海龟策略对股票类型和通道周期都比较敏感，同一参数在不同股票上的效果差异明显。",
            styles["body"],
        )
    )

    story.append(p("五、统计图表与结果解读", styles["h1"]))
    story.append(
        p(
            "下面每张图均包含收盘价、高点通道、低点通道、ATR移动止损线、买入信号、卖出信号、"
            "ATR曲线、策略资金曲线、买入持有资金曲线和最大回撤区域。",
            styles["body"],
        )
    )
    for index, choice in enumerate(all_figure_choices(metrics), start=1):
        append_figure(story, styles, metrics, choice, index)

    story.append(PageBreak())
    story.append(p("六、适用场景与使用心得", styles["h1"]))
    story.append(
        p(
            "海龟法则更适合趋势持续性强、突破后容易形成单边行情的标的。"
            "如果股票长期横盘震荡，价格会频繁突破又快速回落，策略容易反复买入卖出，"
            "交易成本和假突破会侵蚀收益。因此，使用海龟策略时需要关注标的是否具有足够的趋势延续性。",
            styles["body"],
        )
    )
    story.append(
        p(
            "较短通道如10/5或20/10反应更快，更早发现行情，但也更容易在震荡中被洗出。"
            "较长通道如55/20信号更少，更重视大趋势，但入场较慢，可能错过趋势早期收益。"
            "ATR止损倍数越小，风险控制越紧，但被正常波动触发的概率也更高；倍数越大，"
            "趋势容忍度更高，但单笔回撤可能扩大。",
            styles["body"],
        )
    )
    story.append(
        p(
            "从本次实验看，海龟法则不是任何市场都赚钱的固定公式，而是一套纪律化的趋势捕捉框架。"
            "实际应用时应结合更长历史样本、样本外测试、成交量与流动性筛选，并控制单笔风险，"
            "避免只根据一段历史收益进行过度调参。",
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
