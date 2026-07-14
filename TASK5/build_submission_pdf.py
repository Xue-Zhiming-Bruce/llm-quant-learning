# -*- coding: utf-8 -*-
"""Build the final TASK5 submission PDF.

Formatting follows the submission requirements: Songti, 10.5 pt body text,
1.5 line spacing, zero paragraph spacing, and justified body paragraphs.
"""

from __future__ import annotations

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
PDF_PATH = BASE_DIR / "薛智鸣TASK5.pdf"

SONGTI_PATH = Path("/System/Library/Fonts/Supplemental/Songti.ttc")
FONT_NAME = "Songti"
BODY_SIZE = 10.5
BODY_LEADING = BODY_SIZE * 1.5


def register_fonts() -> None:
    """Register macOS Songti, with a standard CJK fallback."""

    if SONGTI_PATH.exists():
        # Songti.ttc index 6 is the Simplified Chinese Regular face.
        pdfmetrics.registerFont(TTFont(FONT_NAME, str(SONGTI_PATH), subfontIndex=6))
    else:
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont

        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        globals()["FONT_NAME"] = "STSong-Light"


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def build_styles() -> dict[str, ParagraphStyle]:
    """Create styles that implement the requested typography."""

    styles = getSampleStyleSheet()
    body = ParagraphStyle(
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
        parent=body,
        firstLineIndent=0,
    )
    title = ParagraphStyle(
        "TitleCN",
        parent=body,
        fontSize=16,
        leading=24,
        firstLineIndent=0,
        alignment=TA_CENTER,
    )
    subtitle = ParagraphStyle(
        "SubtitleCN",
        parent=body,
        firstLineIndent=0,
        alignment=TA_CENTER,
    )
    heading1 = ParagraphStyle(
        "Heading1CN",
        parent=body,
        fontSize=12,
        leading=18,
        firstLineIndent=0,
        alignment=TA_LEFT,
        keepWithNext=True,
    )
    heading2 = ParagraphStyle(
        "Heading2CN",
        parent=body,
        firstLineIndent=0,
        alignment=TA_LEFT,
        keepWithNext=True,
    )
    caption = ParagraphStyle(
        "CaptionCN",
        parent=body,
        firstLineIndent=0,
        alignment=TA_CENTER,
        keepWithNext=True,
    )
    table_cell = ParagraphStyle(
        "TableCellCN",
        parent=body,
        firstLineIndent=0,
        alignment=TA_CENTER,
        wordWrap="CJK",
    )
    table_left = ParagraphStyle(
        "TableLeftCN",
        parent=table_cell,
        alignment=TA_LEFT,
    )
    return {
        "body": body,
        "no_indent": no_indent,
        "title": title,
        "subtitle": subtitle,
        "h1": heading1,
        "h2": heading2,
        "caption": caption,
        "table_cell": table_cell,
        "table_left": table_left,
    }


def add_page_number(canvas, doc) -> None:  # noqa: ANN001
    """Draw a centered page number in the footer."""

    canvas.saveState()
    canvas.setFont(FONT_NAME, 9)
    canvas.drawCentredString(A4[0] / 2, 1.25 * cm, str(doc.page))
    canvas.restoreState()


def make_doc() -> BaseDocTemplate:
    doc = BaseDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=2.54 * cm,
        leftMargin=2.54 * cm,
        topMargin=2.54 * cm,
        bottomMargin=2.54 * cm,
        title="薛智鸣TASK5",
        author="薛智鸣",
        subject="分类模型、评价指标与ROC/AUC实验",
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


def load_results(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_path = OUTPUT_DIR / f"model_metrics_{dataset}.csv"
    confusion_path = OUTPUT_DIR / f"confusion_matrices_{dataset}.csv"
    if not metrics_path.exists() or not confusion_path.exists():
        raise FileNotFoundError(
            f"Missing {dataset} result files. Run analyze_classification.py first."
        )
    return pd.read_csv(metrics_path), pd.read_csv(confusion_path)


def styled_table(
    rows: list[list[Paragraph]],
    col_widths: list[float],
    *,
    repeat_rows: int = 1,
) -> Table:
    """Build a consistently styled black-grid table."""

    table = Table(rows, colWidths=col_widths, repeatRows=repeat_rows, hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
                ("FONTSIZE", (0, 0), (-1, -1), BODY_SIZE),
                ("LEADING", (0, 0), (-1, -1), BODY_LEADING),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.92, 0.92, 0.92)),
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


def algorithm_table(styles: dict[str, ParagraphStyle]) -> Table:
    values = [
        ["算法", "基本原理", "主要优点", "主要局限"],
        ["逻辑回归", "线性组合经过Sigmoid函数转为类别1概率", "速度快、概率输出稳定、系数较易解释", "默认是线性决策边界"],
        ["决策树", "用特征阈值递归切分样本，使节点类别更纯", "能拟合非线性关系、无需标准化、规则直观", "单棵树容易过拟合"],
        ["随机森林", "对样本和特征随机抽样，训练多棵树后投票", "降低单棵树方差、泛化通常更稳定", "计算开销较大、整体规则不直观"],
    ]
    rows = [
        [p(value, styles["table_left"] if index else styles["table_cell"]) for index, value in enumerate(row)]
        for row in values
    ]
    return styled_table(rows, [2.2 * cm, 4.2 * cm, 4.5 * cm, 4.2 * cm])


def confusion_definition_table(styles: dict[str, ParagraphStyle]) -> Table:
    values = [
        ["真实类别/预测类别", "预测为0", "预测为1"],
        ["真实为0", "TN：真负例", "FP：假正例"],
        ["真实为1", "FN：假负例", "TP：真正例"],
    ]
    rows = [[p(value, styles["table_cell"]) for value in row] for row in values]
    return styled_table(rows, [4.5 * cm, 4.5 * cm, 4.5 * cm])


def metrics_table(metrics: pd.DataFrame, styles: dict[str, ParagraphStyle]) -> Table:
    headers = ["模型", "Accuracy", "Precision", "Recall", "F1", "AUC"]
    rows = [[p(value, styles["table_cell"]) for value in headers]]
    for _, row in metrics.iterrows():
        rows.append(
            [
                p(str(row["model"]), styles["table_cell"]),
                p(f"{row['accuracy']:.4f}", styles["table_cell"]),
                p(f"{row['precision']:.4f}", styles["table_cell"]),
                p(f"{row['recall']:.4f}", styles["table_cell"]),
                p(f"{row['f1']:.4f}", styles["table_cell"]),
                p(f"{row['auc']:.4f}", styles["table_cell"]),
            ]
        )
    return styled_table(rows, [3.8 * cm, 2.25 * cm, 2.25 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm])


def confusion_result_table(confusions: pd.DataFrame, styles: dict[str, ParagraphStyle]) -> Table:
    headers = ["模型", "TN", "FP", "FN", "TP"]
    rows = [[p(value, styles["table_cell"]) for value in headers]]
    for _, row in confusions.iterrows():
        rows.append(
            [
                p(str(row["model"]), styles["table_cell"]),
                p(str(int(row["TN"])), styles["table_cell"]),
                p(str(int(row["FP"])), styles["table_cell"]),
                p(str(int(row["FN"])), styles["table_cell"]),
                p(str(int(row["TP"])), styles["table_cell"]),
            ]
        )
    return styled_table(rows, [4.0 * cm, 2.25 * cm, 2.25 * cm, 2.25 * cm, 2.25 * cm])


def append_numbered_figure(
    story: list,
    styles: dict[str, ParagraphStyle],
    image_path: Path,
    figure_number: int,
    title: str,
    interpretation: str,
) -> None:
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    story.append(PageBreak())
    image = Image(str(image_path), width=15.2 * cm, height=11.4 * cm)
    image.hAlign = "CENTER"
    story.append(image)
    story.append(Spacer(1, 0.08 * cm))
    story.append(p(f"图{figure_number} {title}", styles["caption"]))
    story.append(p(f"图{figure_number}解读：{interpretation}", styles["body"]))


def build_story(
    cancer_metrics: pd.DataFrame,
    cancer_confusions: pd.DataFrame,
    stock_metrics: pd.DataFrame,
    stock_confusions: pd.DataFrame,
    styles: dict[str, ParagraphStyle],
) -> list:
    story: list = []

    cancer_best = cancer_metrics.sort_values("auc", ascending=False).iloc[0]
    stock_best = stock_metrics.sort_values("auc", ascending=False).iloc[0]

    story.append(p("分类模型、评价指标与ROC/AUC实验", styles["title"]))
    story.append(p("薛智鸣 TASK5", styles["subtitle"]))
    story.append(Spacer(1, 0.30 * cm))

    story.append(p("一、分类型机器学习算法", styles["h1"]))
    story.append(
        p(
            "分类模型用于预测离散类别。本次任务是二分类问题，应变量只有0和1。模型通常先输出样本属于类别1的概率，"
            "再通过分类阈值把概率转换为0或1。默认阈值为0.5时，预测概率不小于0.5就判为类别1。",
            styles["body"],
        )
    )
    story.append(p("（一）逻辑回归", styles["h2"]))
    story.append(
        p(
            "逻辑回归虽然名称中有“回归”，实际是经典分类算法。它先计算特征的线性组合z，再通过Sigmoid函数"
            "P(Y=1|X)=1/(1+exp(-z))将结果压缩到0至1，所得数值可以解释为样本属于类别1的估计概率。"
            "该模型速度快、结果稳定，系数方向也较容易解释，适合作为分类任务的基线模型。主要限制是默认形成线性决策边界，"
            "当特征与类别之间存在复杂非线性关系时，表达能力可能不足。",
            styles["body"],
        )
    )
    story.append(p("（二）决策树", styles["h2"]))
    story.append(
        p(
            "决策树通过一系列“如果……那么……”规则完成分类。每个节点选择一个特征和阈值切分样本，使子节点中的类别更加纯净，"
            "常用切分指标包括基尼不纯度和信息熵。决策树能处理非线性关系和特征交互，不要求标准化，规则也便于展示；"
            "但单棵树容易把训练数据中的噪声当成规律而过拟合，因此本次限制最大深度为5，并要求每个叶节点至少包含5个训练样本。",
            styles["body"],
        )
    )
    story.append(p("（三）随机森林", styles["h2"]))
    story.append(
        p(
            "随机森林是由多棵决策树组成的集成模型。每棵树从训练集中有放回地抽取样本，每个节点只在随机选取的部分特征中寻找切分点，"
            "最后通过投票或平均预测概率形成结果。多棵树犯错的方向不完全相同，组合后通常能降低单棵树的方差并提高泛化稳定性。"
            "本次建立300棵树。随机森林能够拟合复杂非线性关系，但训练开销较大，整体规则也不如单棵树直观。",
            styles["body"],
        )
    )
    story.append(p("表1 三种分类算法的原理、优点与局限", styles["caption"]))
    story.append(algorithm_table(styles))
    story.append(
        p(
            "表1解读：三种算法的复杂度逐步提高。逻辑回归适合建立可解释的线性基准，决策树便于表达非线性规则，随机森林则通过多棵树集成"
            "提高稳定性。模型选择不应只追求复杂度，还要结合样本量、可解释性和样本外表现。",
            styles["body"],
        )
    )

    story.append(PageBreak())
    story.append(p("二、机器学习模型评价指标", styles["h1"]))
    story.append(p("（一）混淆矩阵", styles["h2"]))
    story.append(
        p(
            "混淆矩阵同时展示真实类别和预测类别。TP表示真实为1且预测为1，TN表示真实为0且预测为0，FP表示真实为0却预测为1，"
            "FN表示真实为1却预测为0。四个计数能够直接显示模型犯了哪一种错误。",
            styles["body"],
        )
    )
    story.append(p("表2 二分类混淆矩阵结构", styles["caption"]))
    story.append(confusion_definition_table(styles))
    story.append(
        p(
            "表2解读：混淆矩阵是其他分类指标的基础。准确率Accuracy=(TP+TN)/(TP+TN+FP+FN)；"
            "精确率Precision=TP/(TP+FP)；召回率Recall=TP/(TP+FN)；F1是Precision和Recall的调和平均。"
            "当类别分布不平衡时，模型即使总是预测多数类也可能获得较高准确率，因此必须同时查看FP、FN、Recall、F1和AUC。",
            styles["body"],
        )
    )
    story.append(p("（二）ROC曲线", styles["h2"]))
    story.append(
        p(
            "模型输出概率后，可以从高到低尝试不同分类阈值。每个阈值会产生一个真正例率TPR和假正例率FPR，"
            "将横轴FPR、纵轴TPR的点连接起来就是ROC曲线。曲线越靠近左上角越好，表示在较低误报率下仍能找出较多正类样本。"
            "对角线代表与随机猜测相当的分类器。ROC评价的是所有阈值下的整体表现，而不是只看默认0.5阈值。",
            styles["body"],
        )
    )
    story.append(p("（三）AUC", styles["h2"]))
    story.append(
        p(
            "AUC是ROC曲线下的面积，也可以理解为随机抽取一个正类和一个负类样本时，模型把正类排在负类之前的概率。"
            "AUC等于1表示排序完全正确，等于0.5表示接近随机排序，小于0.5则可能说明预测方向相反。AUC不依赖某个固定阈值，"
            "但不能代替业务阈值选择；如果正类极少，还应结合Precision-Recall曲线以及FP、FN的实际成本判断模型。",
            styles["body"],
        )
    )

    story.append(p("三、Python实现与实验设置", styles["h1"]))
    story.append(
        p(
            "程序文件为analyze_classification.py，支持乳腺癌数据和股票收益分类数据。默认读取model_data_cancer.csv，"
            "也可通过--dataset stock切换到model_data_stock.csv。程序依次完成数据读取、标签0/1转换、训练测试集划分、"
            "模型训练、测试集评价、AUC计算和ROC绘图，并把评价结果、混淆矩阵、分类报告和逐样本预测写入CSV文件。",
            styles["body"],
        )
    )
    story.append(
        p(
            "本次使用分层随机抽样将80%样本作为训练集、20%作为测试集，并固定random_state=42，确保类别比例相近且结果可复现。"
            "缺失值处理中位数只在训练流程内部估计，避免使用测试集信息。逻辑回归前进行标准化，决策树和随机森林保留原始特征尺度。"
            "所有指标均在未参与训练的测试集上计算。",
            styles["body"],
        )
    )
    story.append(
        p(
            "乳腺癌数据共有569个样本和27个数值特征，标签0有212个、标签1有357个。原始定义为0代表恶性、1代表良性，"
            "因此本次ROC和AUC把良性类别1作为正类。股票数据共有20772条记录和17个连续特征，日期和代码只作为样本标识，"
            "没有作为连续解释变量输入模型。",
            styles["body"],
        )
    )

    story.append(PageBreak())
    story.append(p("四、乳腺癌数据实验结果", styles["h1"]))
    story.append(p("表3 乳腺癌测试集模型评价指标", styles["caption"]))
    story.append(metrics_table(cancer_metrics, styles))
    story.append(
        p(
            f"表3解读：测试集共有114个样本。按AUC排序，{cancer_best['model']}表现最好，AUC为{cancer_best['auc']:.4f}，"
            f"Accuracy为{cancer_best['accuracy']:.4f}。随机森林AUC为0.9907，也具有很强的排序能力。决策树在0.5阈值下的"
            "Accuracy与随机森林相同，但AUC只有0.9165，说明固定阈值下的分类正确率与所有阈值下的概率排序能力是两个不同评价角度。",
            styles["body"],
        )
    )
    story.append(p("表4 乳腺癌测试集混淆矩阵计数", styles["caption"]))
    story.append(confusion_result_table(cancer_confusions, styles))
    story.append(
        p(
            "表4解读：逻辑回归正确识别41个类别0样本和70个类别1样本，只出现1个FP和2个FN，总错误数最少。"
            "决策树的FN同样为2，但FP增加到5；随机森林有3个FP和4个FN。由于本数据把良性定义为类别1，"
            "如用于真实医疗场景，应根据漏判恶性病例的实际风险重新定义正类并选择更合适的阈值。",
            styles["body"],
        )
    )

    append_numbered_figure(
        story,
        styles,
        FIGURE_DIR / "roc_curve_cancer.png",
        1,
        "乳腺癌测试集三种分类模型ROC曲线",
        "蓝色逻辑回归曲线和绿色随机森林曲线几乎贴近左上角，AUC分别为0.9954和0.9907，说明两者都能很好地区分类别0与类别1。"
        "橙色决策树曲线更粗且更靠近对角线，AUC为0.9165；这是因为单棵树的叶节点数量有限，输出的概率档位较少。"
        "三条曲线均明显高于随机基准线，说明模型都具有有效的样本区分能力。",
    )

    story.append(PageBreak())
    story.append(p("五、股票收益分类数据补充实验", styles["h1"]))
    story.append(p("表5 股票数据测试集模型评价指标", styles["caption"]))
    story.append(metrics_table(stock_metrics, styles))
    story.append(
        p(
            f"表5解读：股票测试集共有4155条记录。{stock_best['model']}的AUC最高，为{stock_best['auc']:.4f}，"
            f"Accuracy为{stock_best['accuracy']:.4f}，但整体区分能力仍然有限。逻辑回归Accuracy为0.5954，"
            "却几乎把所有样本判为0，类别1的Recall只有0.0060。这说明类别存在一定不平衡时，单看Accuracy会掩盖模型对正类识别失败的问题。",
            styles["body"],
        )
    )
    story.append(p("表6 股票数据测试集混淆矩阵计数", styles["caption"]))
    story.append(confusion_result_table(stock_confusions, styles))
    story.append(
        p(
            "表6解读：逻辑回归仅识别出10个正类，漏掉1668个正类，显示线性边界和默认0.5阈值不适合当前股票标签。"
            "随机森林识别出535个正类，优于决策树的495个，但仍有1143个FN。模型若用于选股，需要进一步改进特征、调整阈值并进行严格样本外验证。",
            styles["body"],
        )
    )

    append_numbered_figure(
        story,
        styles,
        FIGURE_DIR / "roc_curve_stock.png",
        2,
        "股票测试集三种分类模型ROC曲线",
        "三条曲线都只略高于随机基准线。随机森林AUC为0.6131，优于决策树的0.6015和逻辑回归的0.5578，"
        "说明非线性集成模型从现有基本面因子中提取到了一定信息，但预测优势较弱。该结果不能直接视为可交易策略收益，"
        "还需要考虑时间顺序、交易成本、标签构造以及样本外稳定性。",
    )

    story.append(p("六、结论与注意事项", styles["h1"]))
    story.append(
        p(
            "本次实验完成了二分类数据加载、训练测试集划分、逻辑回归、决策树和随机森林训练，以及混淆矩阵、Accuracy、Precision、"
            "Recall、F1、AUC和ROC曲线评价。在乳腺癌数据上，逻辑回归以0.9954的AUC取得最佳表现，说明较简单的线性模型已经能够形成有效边界；"
            "随机森林表现接近，而单棵决策树的概率排序能力相对较弱。",
            styles["body"],
        )
    )
    story.append(
        p(
            "股票数据上随机森林表现最好，但AUC只有0.6131，现有特征对标签的区分能力有限。股票记录具有明显时间结构，本次随机划分只用于统一的"
            "分类算法练习。若开展真实投资研究，应按日期进行样本外划分或滚动验证，确认标签使用未来收益且所有特征在预测时点可得，避免前视偏差和"
            "同一时期信息泄漏，并在加入交易成本后评价策略结果。",
            styles["body"],
        )
    )
    story.append(
        p(
            "综上，模型评价必须同时观察排序能力、固定阈值下的分类错误和实际误判成本。复杂模型并不必然优于简单模型；只有在独立测试集上稳定有效，"
            "并满足业务约束的模型，才具有进一步应用价值。",
            styles["body"],
        )
    )
    return story


def main() -> None:
    register_fonts()
    cancer_metrics, cancer_confusions = load_results("cancer")
    stock_metrics, stock_confusions = load_results("stock")
    styles = build_styles()
    doc = make_doc()
    story = build_story(
        cancer_metrics,
        cancer_confusions,
        stock_metrics,
        stock_confusions,
        styles,
    )
    doc.build(story)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
