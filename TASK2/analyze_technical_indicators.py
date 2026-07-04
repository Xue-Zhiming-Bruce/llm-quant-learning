"""Analyze stored stock price data and calculate technical indicators.

Outputs:
- output/diagnostics_summary.csv
- output/missing_values.csv
- output/descriptive_statistics.csv
- output/<ts_code>_indicators.csv
- figures/<ts_code>_technical_indicators.png
- analysis_report.md
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
FIGURE_DIR = BASE_DIR / "figures"
FONT_CACHE_DIR = BASE_DIR / ".cache"
FONT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(FONT_CACHE_DIR))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_FILES = sorted(BASE_DIR.glob("*行情数据.csv"))
REQUIRED_COLUMNS = [
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount",
]


def load_price_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"{path.name} is missing columns: {missing_columns}")

    df = df[REQUIRED_COLUMNS].copy()
    df["source_file"] = path.name
    df["stock_name"] = path.stem.replace("行情数据", "")
    df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d")

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "change",
        "pct_chg",
        "vol",
        "amount",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


def wilder_smoothing(values: pd.Series, period: int) -> pd.Series:
    smoothed = pd.Series(np.nan, index=values.index, dtype="float64")
    if len(values) <= period:
        return smoothed

    initial_value = values.iloc[1 : period + 1].mean()
    smoothed.iloc[period] = initial_value
    for i in range(period + 1, len(values)):
        smoothed.iloc[i] = (smoothed.iloc[i - 1] * (period - 1) + values.iloc[i]) / period
    return smoothed


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = wilder_smoothing(gain, period)
    avg_loss = wilder_smoothing(loss, period)
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)

    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50)
    return rsi


def calculate_macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast_period, adjust=False, min_periods=fast_period).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False, min_periods=slow_period).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()
    histogram = macd - signal
    return pd.DataFrame(
        {
            "ema_12": ema_fast,
            "ema_26": ema_slow,
            "macd": macd,
            "macd_signal": signal,
            "macd_hist": histogram,
        }
    )


def calculate_bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_multiplier: float = 2.0,
) -> pd.DataFrame:
    middle = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = middle + std_multiplier * std
    lower = middle - std_multiplier * std
    bandwidth = (upper - lower) / middle
    percent_b = (close - lower) / (upper - lower)
    return pd.DataFrame(
        {
            "bb_middle": middle,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_bandwidth": bandwidth,
            "bb_percent_b": percent_b,
        }
    )


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_prev_close = (df["high"] - df["close"].shift(1)).abs()
    low_prev_close = (df["low"] - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    return wilder_smoothing(true_range, period)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["rsi_14"] = calculate_rsi(result["close"], 14)
    result = pd.concat([result, calculate_macd(result["close"])], axis=1)
    result = pd.concat([result, calculate_bollinger_bands(result["close"])], axis=1)
    result["atr_14"] = calculate_atr(result, 14)
    return result


def diagnostics_for(df: pd.DataFrame, path: Path) -> dict[str, object]:
    duplicate_count = int(df.duplicated(subset=["ts_code", "trade_date"]).sum())
    return {
        "source_file": path.name,
        "stock_name": df["stock_name"].iloc[0],
        "ts_code": df["ts_code"].iloc[0],
        "rows": len(df),
        "columns": len(df.columns),
        "start_date": df["trade_date"].min().date().isoformat(),
        "end_date": df["trade_date"].max().date().isoformat(),
        "duplicate_trade_dates": duplicate_count,
        "total_missing_values": int(df.isna().sum().sum()),
    }


def plot_indicators(df: pd.DataFrame, output_path: Path) -> None:
    ts_code = df["ts_code"].iloc[0]
    dates = df["trade_date"]

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(14, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.2, 1.4, 1.2]},
    )

    axes[0].plot(dates, df["close"], label="Close", linewidth=1.5, color="#1f77b4")
    axes[0].plot(dates, df["bb_middle"], label="BB Middle (SMA20)", linewidth=1.0, color="#2ca02c")
    axes[0].plot(dates, df["bb_upper"], label="BB Upper", linewidth=1.0, color="#d62728")
    axes[0].plot(dates, df["bb_lower"], label="BB Lower", linewidth=1.0, color="#d62728")
    axes[0].fill_between(
        mdates.date2num(dates),
        df["bb_lower"].to_numpy(dtype=float),
        df["bb_upper"].to_numpy(dtype=float),
        color="#d62728",
        alpha=0.08,
    )
    axes[0].set_title(f"{ts_code} Close Price and Bollinger Bands")
    axes[0].set_ylabel("Price")
    axes[0].legend(loc="upper left", ncol=4, fontsize=9)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(dates, df["rsi_14"], label="RSI(14)", color="#9467bd", linewidth=1.3)
    axes[1].axhline(70, color="#d62728", linestyle="--", linewidth=0.9)
    axes[1].axhline(30, color="#2ca02c", linestyle="--", linewidth=0.9)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("RSI")
    axes[1].legend(loc="upper left", fontsize=9)
    axes[1].grid(True, alpha=0.25)

    hist_colors = np.where(df["macd_hist"] >= 0, "#2ca02c", "#d62728")
    axes[2].bar(dates, df["macd_hist"], label="Histogram", color=hist_colors, alpha=0.55, width=1.0)
    axes[2].plot(dates, df["macd"], label="MACD", color="#1f77b4", linewidth=1.2)
    axes[2].plot(dates, df["macd_signal"], label="Signal", color="#ff7f0e", linewidth=1.2)
    axes[2].axhline(0, color="#333333", linewidth=0.8)
    axes[2].set_ylabel("MACD")
    axes[2].legend(loc="upper left", ncol=3, fontsize=9)
    axes[2].grid(True, alpha=0.25)

    axes[3].plot(dates, df["atr_14"], label="ATR(14)", color="#8c564b", linewidth=1.3)
    axes[3].set_ylabel("ATR")
    axes[3].set_xlabel("Trade Date")
    axes[3].legend(loc="upper left", fontsize=9)
    axes[3].grid(True, alpha=0.25)
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[3].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def format_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(item) for item in row) + " |")
    return "\n".join(lines)


def build_report(
    diagnostics: pd.DataFrame,
    latest_values: pd.DataFrame,
    figure_paths: list[Path],
) -> str:
    diagnostic_rows = diagnostics[
        [
            "source_file",
            "stock_name",
            "ts_code",
            "rows",
            "start_date",
            "end_date",
            "duplicate_trade_dates",
            "total_missing_values",
        ]
    ].values.tolist()

    latest_rows = latest_values[
        [
            "stock_name",
            "ts_code",
            "trade_date",
            "close",
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_hist",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "atr_14",
        ]
    ].values.tolist()

    figure_lines = "\n".join(f"- `{path.relative_to(BASE_DIR)}`" for path in figure_paths)

    return f"""# TASK2：基础诊断与技术指标分析

## 1. 数据基础诊断

本次分析读取 `TASK2` 目录下已存储的两份日线行情数据，字段包括 `open`、`high`、`low`、`close`、`pre_close`、`pct_chg`、`vol`、`amount` 等。数据按 `trade_date` 升序排列后再计算指标。

{markdown_table(
    [
        "源文件",
        "股票名称",
        "代码",
        "行数",
        "开始日期",
        "结束日期",
        "重复交易日",
        "缺失值总数",
    ],
    diagnostic_rows,
)}

更完整的诊断结果已保存：

- `output/diagnostics_summary.csv`
- `output/missing_values.csv`
- `output/descriptive_statistics.csv`

## 2. RSI、MACD、布林带指标说明

### RSI：相对强弱指标

RSI 是动量震荡指标，衡量最近一段时间上涨力度与下跌力度的相对强弱，常用周期为 14。计算步骤：

```text
Gain_t = max(Close_t - Close_(t-1), 0)
Loss_t = max(Close_(t-1) - Close_t, 0)

AvgGain_t = (AvgGain_(t-1) * 13 + Gain_t) / 14
AvgLoss_t = (AvgLoss_(t-1) * 13 + Loss_t) / 14

RS = AvgGain / AvgLoss
RSI = 100 - 100 / (1 + RS)
```

作用：RSI 通常用于观察超买超卖、短期动能变化和背离。常见参考线是 70 和 30，但强趋势中 RSI 可能长时间处于高位或低位，因此不能单独作为买卖依据。

### MACD：指数平滑异同移动平均线

MACD 用短周期 EMA 与长周期 EMA 的差值观察趋势和动能变化，常用参数为 12、26、9：

```text
EMA_t = alpha * Close_t + (1 - alpha) * EMA_(t-1)
alpha = 2 / (N + 1)

MACD Line = EMA_12 - EMA_26
Signal Line = EMA_9(MACD Line)
Histogram = MACD Line - Signal Line
```

作用：MACD 常用于判断趋势方向、金叉死叉、动能增强或衰减。震荡行情中容易出现反复假信号。

### 布林带：Bollinger Bands

布林带用均线和标准差描述价格的相对高低和波动区间，常用参数为 20 日均线和 2 倍标准差：

```text
Middle Band = SMA_20
Upper Band = SMA_20 + 2 * Std_20
Lower Band = SMA_20 - 2 * Std_20

Bandwidth = (Upper Band - Lower Band) / Middle Band
%B = (Close - Lower Band) / (Upper Band - Lower Band)
```

作用：布林带可以观察价格相对位置、波动率收缩和扩张，也可辅助识别突破或均值回归机会。价格触及上下轨不等于一定反转。

## 3. Python 实现和可视化输出

脚本位置：`analyze_technical_indicators.py`

脚本完成了：

1. 加载已存储的股价 CSV；
2. 检查缺失值、重复交易日，并计算描述性统计量；
3. 计算 RSI(14)、MACD(12,26,9)、布林带(20,2)；
4. 扩展计算 ATR(14)；
5. 输出指标明细 CSV 和技术指标图。

生成图形：

{figure_lines}

最新交易日指标摘要：

{markdown_table(
    [
        "股票名称",
        "代码",
        "日期",
        "收盘价",
        "RSI14",
        "MACD",
        "Signal",
        "Hist",
        "BB上轨",
        "BB中轨",
        "BB下轨",
        "ATR14",
    ],
    latest_rows,
)}

## 4. 其他典型指标与扩展指标 ATR

量化中常见指标还包括：

- 均线 MA / EMA：判断趋势方向和均线交叉。
- 动量 Momentum / ROC：衡量当前价格相对过去价格的涨跌幅。
- KDJ / Stochastic Oscillator：观察收盘价在最近高低区间中的位置。
- ADX：判断趋势强弱，不直接判断方向。
- OBV：结合成交量观察资金流入流出趋势。
- VWAP：成交量加权平均价，常用于日内交易和执行交易评价。
- 波动率 Volatility：用收益率标准差衡量风险。
- MFI：结合价格和成交量的资金流量指标。

本次扩展选取 ATR（Average True Range，平均真实波幅）。ATR 由 Welles Wilder 提出，衡量价格真实波动幅度，不判断涨跌方向。计算方法：

```text
TR_t = max(
  High_t - Low_t,
  abs(High_t - Close_(t-1)),
  abs(Low_t - Close_(t-1))
)

ATR_t = (ATR_(t-1) * 13 + TR_t) / 14
```

作用：ATR 常用于动态止损、仓位控制、识别波动扩大或收缩。例如趋势策略中可以用 `2 * ATR` 作为止损距离，避免固定金额止损无法适应市场波动。

## 参考资料

- Investopedia, Relative Strength Index (RSI): https://www.investopedia.com/terms/r/rsi.asp
- Investopedia, Moving Average Convergence Divergence (MACD): https://www.investopedia.com/terms/m/macd.asp
- Wikipedia, Bollinger Bands: https://en.wikipedia.org/wiki/Bollinger_Bands
- Wikipedia, Average True Range: https://en.wikipedia.org/wiki/Average_true_range
"""


def main() -> None:
    if not DATA_FILES:
        raise SystemExit("No price CSV files found in TASK2.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    diagnostics_records: list[dict[str, object]] = []
    missing_records: list[dict[str, object]] = []
    descriptive_frames: list[pd.DataFrame] = []
    latest_records: list[pd.Series] = []
    figure_paths: list[Path] = []

    for path in DATA_FILES:
        df = load_price_data(path)
        diagnostics_records.append(diagnostics_for(df, path))

        missing = df.isna().sum().reset_index()
        missing.columns = ["column", "missing_count"]
        missing["source_file"] = path.name
        missing["stock_name"] = df["stock_name"].iloc[0]
        missing["ts_code"] = df["ts_code"].iloc[0]
        missing_records.extend(missing.to_dict("records"))

        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "change",
            "pct_chg",
            "vol",
            "amount",
        ]
        desc = df[numeric_columns].describe().T.reset_index().rename(columns={"index": "column"})
        desc.insert(0, "source_file", path.name)
        desc.insert(1, "stock_name", df["stock_name"].iloc[0])
        desc.insert(2, "ts_code", df["ts_code"].iloc[0])
        descriptive_frames.append(desc)

        indicator_df = add_indicators(df)
        ts_code = indicator_df["ts_code"].iloc[0]
        indicator_output_path = OUTPUT_DIR / f"{ts_code}_indicators.csv"
        indicator_df.to_csv(indicator_output_path, index=False, encoding="utf-8-sig")

        figure_path = FIGURE_DIR / f"{ts_code}_technical_indicators.png"
        plot_indicators(indicator_df, figure_path)
        figure_paths.append(figure_path)

        latest_records.append(indicator_df.iloc[-1])

    diagnostics = pd.DataFrame(diagnostics_records)
    diagnostics.to_csv(OUTPUT_DIR / "diagnostics_summary.csv", index=False, encoding="utf-8-sig")

    missing_values = pd.DataFrame(missing_records)
    missing_values = missing_values[
        ["source_file", "stock_name", "ts_code", "column", "missing_count"]
    ]
    missing_values.to_csv(OUTPUT_DIR / "missing_values.csv", index=False, encoding="utf-8-sig")

    descriptive_statistics = pd.concat(descriptive_frames, ignore_index=True)
    descriptive_statistics.to_csv(
        OUTPUT_DIR / "descriptive_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    latest_values = pd.DataFrame(latest_records).copy()
    latest_values["trade_date"] = latest_values["trade_date"].dt.date.astype(str)

    report = build_report(diagnostics, latest_values, figure_paths)
    (BASE_DIR / "analysis_report.md").write_text(report, encoding="utf-8")

    print(f"Analyzed {len(DATA_FILES)} CSV files.")
    print(f"Report: {BASE_DIR / 'analysis_report.md'}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Figure directory: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
