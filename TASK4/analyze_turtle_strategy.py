"""Backtest a Turtle/Donchian channel strategy on stored stock price data.

Outputs:
- output/strategy_metrics.csv
- output/trading_signals.csv
- output/<ts_code>_turtle_<entry>_<exit>_backtest.csv
- figures/<ts_code>_turtle_<entry>_<exit>_strategy.png
- analysis_report.md
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
OUTPUT_DIR = BASE_DIR / "output"
FIGURE_DIR = BASE_DIR / "figures"
CACHE_DIR = Path(tempfile.gettempdir()) / "llm_quant_task4_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR / "xdg"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

DEFAULT_CHANNEL_PAIRS = [(20, 10), (55, 20), (10, 5)]
KNOWN_STOCK_NAMES = {
    "000001.SZ": "平安银行",
    "600031.SH": "三一重工",
    "688017.SH": "绿的谐波",
}


def discover_data_files() -> list[Path]:
    files = [
        *sorted((PROJECT_ROOT / "TASK2").glob("*行情数据.csv")),
        *sorted((PROJECT_ROOT / "TASK1" / "data").glob("*_daily_*.csv")),
    ]
    return [path for path in files if path.is_file()]


def parse_channel_pairs(value: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for raw_pair in value.split(","):
        raw_pair = raw_pair.strip()
        if not raw_pair:
            continue

        delimiter = ":" if ":" in raw_pair else "/"
        if delimiter in raw_pair:
            parts = [part.strip() for part in raw_pair.split(delimiter)]
            if len(parts) != 2:
                raise argparse.ArgumentTypeError(
                    f"Invalid channel pair '{raw_pair}'. Use 20:10 or 20."
                )
            entry_window, exit_window = int(parts[0]), int(parts[1])
        else:
            entry_window = exit_window = int(raw_pair)

        if entry_window <= 0 or exit_window <= 0:
            raise argparse.ArgumentTypeError("Channel windows must be positive.")
        pairs.append((entry_window, exit_window))

    if not pairs:
        raise argparse.ArgumentTypeError("At least one channel setting is required.")
    return pairs


def resolve_data_file(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path

    project_path = PROJECT_ROOT / path
    if project_path.exists():
        return project_path
    return Path.cwd() / path


def display_path(path: Path, base: Path = PROJECT_ROOT) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def parse_trade_date(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip().str.replace("\ufeff", "", regex=False)
    compact_mask = values.str.fullmatch(r"\d{8}")

    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    parsed.loc[compact_mask] = pd.to_datetime(
        values.loc[compact_mask],
        format="%Y%m%d",
        errors="coerce",
    )
    parsed.loc[~compact_mask] = pd.to_datetime(values.loc[~compact_mask], errors="coerce")
    return parsed


def stock_name_from_path(path: Path, ts_code: str) -> str:
    if ts_code in KNOWN_STOCK_NAMES:
        return KNOWN_STOCK_NAMES[ts_code]

    stem = path.stem
    if "行情数据" in stem:
        name = stem.replace("行情数据", "")
        return name or ts_code
    return ts_code


def load_price_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"{path.name} is missing columns: {missing_columns}")

    df = df[REQUIRED_COLUMNS].copy()
    df["trade_date"] = parse_trade_date(df["trade_date"])
    if df["trade_date"].isna().any():
        bad_count = int(df["trade_date"].isna().sum())
        raise ValueError(f"{path.name} has {bad_count} unparseable trade_date values.")

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

    df = df.dropna(subset=["open", "high", "low", "close"]).copy()
    df = df.sort_values(["ts_code", "trade_date"]).drop_duplicates(
        subset=["ts_code", "trade_date"],
        keep="last",
    )
    df = df.reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{path.name} has no usable rows after cleaning.")

    ts_code = str(df["ts_code"].iloc[0])
    df["source_file"] = display_path(path)
    df["stock_name"] = stock_name_from_path(path, ts_code)
    return df


def wilder_smoothing(values: pd.Series, period: int) -> pd.Series:
    smoothed = pd.Series(np.nan, index=values.index, dtype="float64")
    if len(values) <= period:
        return smoothed

    initial_value = values.iloc[1 : period + 1].mean()
    smoothed.iloc[period] = initial_value
    for i in range(period + 1, len(values)):
        smoothed.iloc[i] = (smoothed.iloc[i - 1] * (period - 1) + values.iloc[i]) / period
    return smoothed


def calculate_true_range(df: pd.DataFrame) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_prev_close = (df["high"] - df["close"].shift(1)).abs()
    low_prev_close = (df["low"] - df["close"].shift(1)).abs()
    return pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)


def add_turtle_strategy_columns(
    df: pd.DataFrame,
    entry_window: int,
    exit_window: int,
    atr_period: int,
    atr_stop_multiple: float,
    initial_capital: float,
    fee_rate: float,
) -> pd.DataFrame:
    result = df.copy()
    result["upper_channel"] = (
        result["high"].shift(1).rolling(window=entry_window, min_periods=entry_window).max()
    )
    result["lower_channel"] = (
        result["low"].shift(1).rolling(window=exit_window, min_periods=exit_window).min()
    )
    result["true_range"] = calculate_true_range(result)
    result["atr"] = wilder_smoothing(result["true_range"], atr_period)

    result["trade_signal"] = 0
    result["signal_reason"] = ""
    result["target_position"] = 0.0
    result["entry_price"] = np.nan
    result["highest_close_since_entry"] = np.nan
    result["atr_stop"] = np.nan

    in_position = False
    entry_price = np.nan
    highest_close = np.nan
    trailing_stop = np.nan

    for idx, row in result.iterrows():
        close = float(row["close"])
        upper_channel = row["upper_channel"]
        lower_channel = row["lower_channel"]
        atr = row["atr"]

        if in_position:
            highest_close = max(highest_close, close)
            if pd.notna(atr):
                next_stop = highest_close - atr_stop_multiple * float(atr)
                if pd.isna(trailing_stop):
                    trailing_stop = next_stop
                else:
                    trailing_stop = max(trailing_stop, next_stop)

            result.at[idx, "target_position"] = 1.0
            result.at[idx, "entry_price"] = entry_price
            result.at[idx, "highest_close_since_entry"] = highest_close
            result.at[idx, "atr_stop"] = trailing_stop

            channel_exit = pd.notna(lower_channel) and close < float(lower_channel)
            stop_exit = pd.notna(trailing_stop) and close <= float(trailing_stop)

            if channel_exit or stop_exit:
                result.at[idx, "trade_signal"] = -1
                result.at[idx, "signal_reason"] = "LOW_CHANNEL" if channel_exit else "ATR_STOP"
                result.at[idx, "target_position"] = 0.0
                in_position = False
                entry_price = np.nan
                highest_close = np.nan
                trailing_stop = np.nan
        else:
            breakout_entry = (
                pd.notna(upper_channel)
                and pd.notna(atr)
                and close > float(upper_channel)
            )
            if breakout_entry:
                in_position = True
                entry_price = close
                highest_close = close
                trailing_stop = close - atr_stop_multiple * float(atr)

                result.at[idx, "trade_signal"] = 1
                result.at[idx, "signal_reason"] = "BREAKOUT"
                result.at[idx, "target_position"] = 1.0
                result.at[idx, "entry_price"] = entry_price
                result.at[idx, "highest_close_since_entry"] = highest_close
                result.at[idx, "atr_stop"] = trailing_stop

    result["position"] = result["target_position"].shift(1).fillna(0.0)
    result["market_return"] = result["close"].pct_change().fillna(0.0)
    result["turnover"] = result["position"].diff().abs().fillna(result["position"].abs())
    result["transaction_cost"] = result["turnover"] * fee_rate
    result["strategy_return"] = result["position"] * result["market_return"]
    result["strategy_return"] = result["strategy_return"] - result["transaction_cost"]
    result["strategy_equity"] = initial_capital * (1 + result["strategy_return"]).cumprod()
    result["benchmark_equity"] = initial_capital * (1 + result["market_return"]).cumprod()
    result["running_peak"] = result["strategy_equity"].cummax()
    result["drawdown"] = result["strategy_equity"] / result["running_peak"] - 1.0

    result["entry_window"] = entry_window
    result["exit_window"] = exit_window
    result["atr_period"] = atr_period
    result["atr_stop_multiple"] = atr_stop_multiple
    return result


def safe_annualized_return(cumulative_return: float, trading_days: int) -> float:
    if trading_days <= 0:
        return np.nan
    if cumulative_return <= -1:
        return -1.0
    return (1 + cumulative_return) ** (252 / trading_days) - 1


def safe_sharpe_ratio(strategy_return: pd.Series, risk_free_rate: float) -> float:
    daily_returns = strategy_return.dropna()
    if daily_returns.empty:
        return np.nan

    excess_daily = daily_returns - risk_free_rate / 252
    std = excess_daily.std(ddof=1)
    if pd.isna(std) or std == 0:
        return np.nan
    return np.sqrt(252) * excess_daily.mean() / std


def completed_trade_returns(strategy_df: pd.DataFrame, fee_rate: float) -> list[float]:
    returns: list[float] = []
    entry_price: float | None = None
    for _, row in strategy_df.loc[strategy_df["trade_signal"] != 0].iterrows():
        if row["trade_signal"] == 1:
            entry_price = float(row["close"])
        elif row["trade_signal"] == -1 and entry_price is not None:
            returns.append(float(row["close"]) / entry_price - 1 - 2 * fee_rate)
            entry_price = None
    return returns


def calculate_metrics(
    strategy_df: pd.DataFrame,
    entry_window: int,
    exit_window: int,
    atr_period: int,
    atr_stop_multiple: float,
    initial_capital: float,
    fee_rate: float,
    risk_free_rate: float,
) -> dict[str, object]:
    final_equity = float(strategy_df["strategy_equity"].iloc[-1])
    benchmark_final_equity = float(strategy_df["benchmark_equity"].iloc[-1])
    cumulative_return = final_equity / initial_capital - 1
    benchmark_return = benchmark_final_equity / initial_capital - 1
    trading_days = len(strategy_df)
    round_trips = completed_trade_returns(strategy_df, fee_rate)

    win_rate = np.nan
    if round_trips:
        win_rate = sum(trade_return > 0 for trade_return in round_trips) / len(round_trips)

    return {
        "stock_name": strategy_df["stock_name"].iloc[0],
        "ts_code": strategy_df["ts_code"].iloc[0],
        "source_file": strategy_df["source_file"].iloc[0],
        "start_date": strategy_df["trade_date"].min().date().isoformat(),
        "end_date": strategy_df["trade_date"].max().date().isoformat(),
        "entry_window": entry_window,
        "exit_window": exit_window,
        "atr_period": atr_period,
        "atr_stop_multiple": atr_stop_multiple,
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "cumulative_return": cumulative_return,
        "benchmark_return": benchmark_return,
        "excess_return_vs_benchmark": cumulative_return - benchmark_return,
        "annualized_return": safe_annualized_return(cumulative_return, trading_days),
        "annualized_volatility": strategy_df["strategy_return"].std(ddof=1) * np.sqrt(252),
        "max_drawdown": float(strategy_df["drawdown"].min()),
        "sharpe_ratio": safe_sharpe_ratio(strategy_df["strategy_return"], risk_free_rate),
        "buy_signals": int((strategy_df["trade_signal"] == 1).sum()),
        "sell_signals": int((strategy_df["trade_signal"] == -1).sum()),
        "trade_signals": int((strategy_df["trade_signal"] != 0).sum()),
        "completed_round_trips": len(round_trips),
        "round_trip_win_rate": win_rate,
        "average_round_trip_return": np.mean(round_trips) if round_trips else np.nan,
        "exposure_ratio": float(strategy_df["position"].mean()),
        "atr_stop_exits": int((strategy_df["signal_reason"] == "ATR_STOP").sum()),
        "channel_exits": int((strategy_df["signal_reason"] == "LOW_CHANNEL").sum()),
        "fee_rate": fee_rate,
    }


def signal_events(strategy_df: pd.DataFrame) -> pd.DataFrame:
    events = strategy_df.loc[strategy_df["trade_signal"] != 0].copy()
    if events.empty:
        return pd.DataFrame(
            columns=[
                "stock_name",
                "ts_code",
                "trade_date",
                "action",
                "signal_reason",
                "close",
                "upper_channel",
                "lower_channel",
                "atr",
                "atr_stop",
                "entry_window",
                "exit_window",
                "atr_period",
                "atr_stop_multiple",
            ]
        )

    events["action"] = np.where(events["trade_signal"] == 1, "BUY", "SELL")
    return events[
        [
            "stock_name",
            "ts_code",
            "trade_date",
            "action",
            "signal_reason",
            "close",
            "upper_channel",
            "lower_channel",
            "atr",
            "atr_stop",
            "entry_window",
            "exit_window",
            "atr_period",
            "atr_stop_multiple",
        ]
    ]


def plot_strategy(strategy_df: pd.DataFrame, output_path: Path) -> None:
    dates = strategy_df["trade_date"]
    date_numbers = mdates.date2num(dates)
    buys = strategy_df[strategy_df["trade_signal"] == 1]
    sells = strategy_df[strategy_df["trade_signal"] == -1]
    title = (
        f"{strategy_df['ts_code'].iloc[0]} Turtle "
        f"({strategy_df['entry_window'].iloc[0]}/{strategy_df['exit_window'].iloc[0]})"
    )

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(14, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [2.6, 1.1, 1.3, 1.0]},
    )

    axes[0].plot(dates, strategy_df["close"], label="Close", color="#1f77b4", linewidth=1.3)
    axes[0].plot(
        dates,
        strategy_df["upper_channel"],
        label=f"Upper channel ({strategy_df['entry_window'].iloc[0]})",
        color="#2ca02c",
        linewidth=1.0,
    )
    axes[0].plot(
        dates,
        strategy_df["lower_channel"],
        label=f"Lower channel ({strategy_df['exit_window'].iloc[0]})",
        color="#ff7f0e",
        linewidth=1.0,
    )
    axes[0].plot(
        dates,
        strategy_df["atr_stop"],
        label=f"ATR stop ({strategy_df['atr_stop_multiple'].iloc[0]:.1f}x)",
        color="#9467bd",
        linewidth=1.0,
        linestyle="--",
    )
    axes[0].scatter(
        buys["trade_date"],
        buys["close"],
        marker="^",
        color="#d62728",
        s=70,
        label="Buy signal",
        zorder=4,
    )
    axes[0].scatter(
        sells["trade_date"],
        sells["close"],
        marker="v",
        color="#111111",
        s=70,
        label="Sell signal",
        zorder=4,
    )
    axes[0].set_title(f"{title}: Price, Channels and Signals")
    axes[0].set_ylabel("Price")
    axes[0].legend(loc="upper left", ncol=3, fontsize=9)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(
        dates,
        strategy_df["atr"],
        label=f"ATR ({strategy_df['atr_period'].iloc[0]})",
        color="#8c564b",
        linewidth=1.2,
    )
    axes[1].set_ylabel("ATR")
    axes[1].legend(loc="upper left", fontsize=9)
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(
        dates,
        strategy_df["strategy_equity"],
        label="Strategy equity",
        color="#d62728",
        linewidth=1.3,
    )
    axes[2].plot(
        dates,
        strategy_df["benchmark_equity"],
        label="Buy-and-hold equity",
        color="#1f77b4",
        linewidth=1.1,
    )
    axes[2].set_ylabel("Equity")
    axes[2].legend(loc="upper left", fontsize=9)
    axes[2].grid(True, alpha=0.25)

    axes[3].fill_between(
        date_numbers,
        strategy_df["drawdown"].to_numpy(dtype=float),
        0,
        color="#7f7f7f",
        alpha=0.35,
    )
    axes[3].set_ylabel("Drawdown")
    axes[3].set_xlabel("Trade Date")
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


def format_percent(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value) * 100:.2f}%"


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def relative_output_path(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def build_observation_text(metrics: pd.DataFrame) -> str:
    best_rows = metrics.sort_values(
        ["stock_name", "cumulative_return"],
        ascending=[True, False],
    ).groupby("stock_name", as_index=False).head(1)

    lines = []
    for _, row in best_rows.iterrows():
        comparison = "跑赢" if row["excess_return_vs_benchmark"] > 0 else "跑输"
        lines.append(
            f"- {row['stock_name']}（{row['ts_code']}）表现最好的参数是 "
            f"{int(row['entry_window'])}/{int(row['exit_window'])}，"
            f"策略累计回报 {format_percent(row['cumulative_return'])}，"
            f"同期买入持有回报 {format_percent(row['benchmark_return'])}，"
            f"{comparison}基准 {format_percent(abs(row['excess_return_vs_benchmark']))}。"
        )

    return "\n".join(lines)


def build_report(
    metrics: pd.DataFrame,
    figure_paths: list[Path],
    data_files: list[Path],
    channel_pairs: list[tuple[int, int]],
    atr_period: int,
    atr_stop_multiple: float,
    fee_rate: float,
) -> str:
    metrics_for_table = metrics.sort_values(["stock_name", "entry_window", "exit_window"])
    metric_rows = []
    for _, row in metrics_for_table.iterrows():
        metric_rows.append(
            [
                row["stock_name"],
                row["ts_code"],
                f"{row['start_date']} 至 {row['end_date']}",
                f"{int(row['entry_window'])}/{int(row['exit_window'])}",
                format_percent(row["cumulative_return"]),
                format_percent(row["benchmark_return"]),
                format_percent(row["max_drawdown"]),
                format_value(row["sharpe_ratio"]),
                format_percent(row["annualized_volatility"]),
                int(row["trade_signals"]),
                format_percent(row["exposure_ratio"]),
                int(row["atr_stop_exits"]),
                int(row["channel_exits"]),
            ]
        )

    best = metrics.sort_values("cumulative_return", ascending=False).iloc[0]
    worst_mdd = metrics.sort_values("max_drawdown").iloc[0]
    data_lines = "\n".join(f"- `{display_path(path)}`" for path in data_files)
    channel_text = "、".join(f"{entry}/{exit}" for entry, exit in channel_pairs)
    figure_lines = "\n".join(f"- `{relative_output_path(path)}`" for path in figure_paths)

    return f"""# TASK4：海龟交易法则与通道突破策略回测

## 1. 海龟策略的核心思想与关键优势

海龟交易法则是一套经典的趋势跟随策略。它的核心不是预测明天涨跌，而是用价格突破来确认趋势已经出现：当价格突破过去一段时间的最高价通道时买入；当价格跌破较短周期的最低价通道，或触及按波动率设置的止损位时离场。

核心思想可以概括为：

- 顺势而为：只在价格创出阶段新高后入场，避免提前猜底或主观判断。
- 截断亏损：用低通道和 ATR 止损限制单笔交易亏损。
- 让利润奔跑：趋势延续时继续持仓，不因为短期回调轻易卖出。
- 规则化执行：入场、出场、止损和仓位都由规则决定，减少情绪干扰。
- 波动率适配：ATR 会随市场波动放大或收缩，使止损距离更贴合股票自身特征。

关键优势：

- 方法透明，指标少，易于复现和检验；
- 适合捕捉中长期趋势行情，特别是突破后持续上涨的标的；
- 风险控制明确，避免亏损无限扩大；
- 可跨股票、商品、指数等品种迁移测试；
- 参数直观，便于通过通道周期、ATR 周期和止损倍数进行敏感性分析。

## 2. 高低点通道、ATR 与止损条件

高低点通道又称 Donchian Channel。本次实现中，为避免使用当天价格造成未来函数，通道均使用“昨日以前”的数据计算：

```text
Upper Channel_t = max(High_{{t-N}}, ..., High_{{t-1}})
Lower Channel_t = min(Low_{{t-M}}, ..., Low_{{t-1}})
```

其中 N 是入场通道周期，M 是离场通道周期。经典海龟法则常用 20 日突破入场、10 日低点离场，或 55 日突破入场、20 日低点离场。

ATR（Average True Range，平均真实波幅）衡量股票的真实波动幅度。先计算真实波幅 TR：

```text
TR_t = max(
    High_t - Low_t,
    abs(High_t - Close_{{t-1}}),
    abs(Low_t - Close_{{t-1}})
)
```

再对 TR 做 Wilder 平滑得到 ATR。本次默认 ATR 周期为 {atr_period}。

本次策略的买卖规则：

- 买入：收盘价向上突破 N 日高点通道，生成买入信号；
- 卖出 1：收盘价跌破 M 日低点通道，生成卖出信号；
- 卖出 2：收盘价低于 ATR 移动止损线，生成止损卖出信号；
- ATR 止损线：持仓后记录入场以来最高收盘价，并使用 `最高收盘价 - {atr_stop_multiple:.1f} * ATR` 作为移动止损参考。

## 3. Python 实现

脚本位置：`analyze_turtle_strategy.py`

默认运行：

```bash
python TASK4/analyze_turtle_strategy.py
```

自定义通道周期示例：

```bash
python TASK4/analyze_turtle_strategy.py --channel-pairs 20:10,55:20,30:15
```

只测试某一个数据文件示例：

```bash
python TASK4/analyze_turtle_strategy.py --data-file TASK2/平安集团行情数据.csv --channel-pairs 20:10
```

脚本完成了以下步骤：

1. 加载已存储的股价 CSV，并兼容 `YYYYMMDD` 与 `YYYY-MM-DD` 两种日期格式；
2. 设置高低价格通道周期，本次测试入场/离场参数为 {channel_text}；
3. 计算高点通道、低点通道、TR 和 ATR；
4. 根据突破、低通道离场和 ATR 止损计算买入卖出信号；
5. 绘制股价、高低通道、ATR 止损线、交易信号、ATR、资金曲线和回撤曲线；
6. 模拟交易并计算累计回报、年化收益、最大回撤、夏普比率、胜率、持仓比例等指标。

本次回测假设：

- 初始资金为 100000；
- 只做多，不做空；
- 信号在当天收盘后确认，下一交易日才改变持仓；
- 满仓持有或空仓等待，未实现经典海龟的分批加仓；
- 单边交易成本为 {fee_rate * 100:.2f}%。

读取的数据文件：

{data_lines}

## 4. 不同股票与通道参数结果

{markdown_table(
    [
        "股票",
        "代码",
        "回测区间",
        "入场/离场通道",
        "策略累计回报",
        "买入持有回报",
        "最大回撤",
        "夏普比率",
        "年化波动",
        "信号次数",
        "持仓比例",
        "ATR止损次数",
        "低通道离场次数",
    ],
    metric_rows,
)}

表现最好的组合是 {best['stock_name']}（{best['ts_code']}）的 {int(best['entry_window'])}/{int(best['exit_window'])} 通道，策略累计回报为 {format_percent(best['cumulative_return'])}。最大回撤最深的组合是 {worst_mdd['stock_name']}（{worst_mdd['ts_code']}）的 {int(worst_mdd['entry_window'])}/{int(worst_mdd['exit_window'])} 通道，MDD 为 {format_percent(worst_mdd['max_drawdown'])}。

各股票最优参数观察：

{build_observation_text(metrics)}

生成图形：

{figure_lines}

完整输出文件：

- `output/strategy_metrics.csv`
- `output/trading_signals.csv`
- `output/<ts_code>_turtle_<entry>_<exit>_backtest.csv`
- `figures/<ts_code>_turtle_<entry>_<exit>_strategy.png`

## 5. 适用场景与使用心得

海龟法则更适合趋势持续性强、突破后容易形成单边行情的标的。若股票长期横盘震荡，价格会频繁突破又快速回落，策略容易反复买入卖出，交易成本和假突破会侵蚀收益。

较短通道，例如 10/5 或 20/10，信号更敏感，更早发现行情，也更容易在震荡中被洗出。较长通道，例如 55/20，信号更少，更重视大趋势，但入场更慢，可能错过趋势早期收益。ATR 止损倍数越小，风险控制越紧，但被正常波动触发的概率也更高；倍数越大，趋势容忍度更高，但单笔回撤会变大。

从本次实验可以看到，不同股票对通道周期非常敏感。海龟法则不是“任何市场都赚钱”的公式，它更像一套纪律化趋势捕捉框架。实际使用时应结合更长历史样本、样本外测试、成交量与流动性筛选，并控制单笔风险，避免只根据一段历史收益过度调参。
"""


def run_analysis(
    data_files: list[Path],
    channel_pairs: list[tuple[int, int]],
    atr_period: int,
    atr_stop_multiple: float,
    initial_capital: float,
    fee_rate: float,
    risk_free_rate: float,
) -> tuple[pd.DataFrame, list[Path]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    metrics_records: list[dict[str, object]] = []
    signal_frames: list[pd.DataFrame] = []
    figure_paths: list[Path] = []

    for path in data_files:
        price_df = load_price_data(path)
        ts_code = str(price_df["ts_code"].iloc[0])

        for entry_window, exit_window in channel_pairs:
            strategy_df = add_turtle_strategy_columns(
                price_df,
                entry_window,
                exit_window,
                atr_period,
                atr_stop_multiple,
                initial_capital,
                fee_rate,
            )
            metrics_records.append(
                calculate_metrics(
                    strategy_df,
                    entry_window,
                    exit_window,
                    atr_period,
                    atr_stop_multiple,
                    initial_capital,
                    fee_rate,
                    risk_free_rate,
                )
            )
            signal_frames.append(signal_events(strategy_df))

            backtest_output_path = (
                OUTPUT_DIR / f"{ts_code}_turtle_{entry_window}_{exit_window}_backtest.csv"
            )
            strategy_df.to_csv(backtest_output_path, index=False, encoding="utf-8-sig")

            figure_path = FIGURE_DIR / f"{ts_code}_turtle_{entry_window}_{exit_window}_strategy.png"
            plot_strategy(strategy_df, figure_path)
            figure_paths.append(figure_path)

    metrics = pd.DataFrame(metrics_records)
    metrics.to_csv(OUTPUT_DIR / "strategy_metrics.csv", index=False, encoding="utf-8-sig")

    non_empty_signal_frames = [frame for frame in signal_frames if not frame.empty]
    if non_empty_signal_frames:
        signals = pd.concat(non_empty_signal_frames, ignore_index=True)
    elif signal_frames:
        signals = signal_frames[0]
    else:
        signals = pd.DataFrame()
    signals.to_csv(OUTPUT_DIR / "trading_signals.csv", index=False, encoding="utf-8-sig")

    report = build_report(
        metrics=metrics,
        figure_paths=figure_paths,
        data_files=data_files,
        channel_pairs=channel_pairs,
        atr_period=atr_period,
        atr_stop_multiple=atr_stop_multiple,
        fee_rate=fee_rate,
    )
    (BASE_DIR / "analysis_report.md").write_text(report, encoding="utf-8")
    return metrics, figure_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-file",
        action="append",
        help="Price CSV path. Repeat this option to test multiple stocks.",
    )
    parser.add_argument(
        "--channel-pairs",
        type=parse_channel_pairs,
        default=DEFAULT_CHANNEL_PAIRS,
        help="Comma-separated entry/exit channel settings, for example: 20:10,55:20,30.",
    )
    parser.add_argument("--atr-period", type=int, default=20)
    parser.add_argument("--atr-stop-multiple", type=float, default=2.0)
    parser.add_argument("--initial-capital", type=float, default=100000.0)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.atr_period <= 0:
        raise SystemExit("ATR period must be positive.")
    if args.atr_stop_multiple <= 0:
        raise SystemExit("ATR stop multiple must be positive.")

    if args.data_file:
        data_files = [resolve_data_file(path) for path in args.data_file]
    else:
        data_files = discover_data_files()

    missing_files = [path for path in data_files if not path.exists()]
    if missing_files:
        missing_text = ", ".join(str(path) for path in missing_files)
        raise SystemExit(f"Data file not found: {missing_text}")
    if not data_files:
        raise SystemExit("No stored stock price CSV files found.")

    metrics, figure_paths = run_analysis(
        data_files=data_files,
        channel_pairs=args.channel_pairs,
        atr_period=args.atr_period,
        atr_stop_multiple=args.atr_stop_multiple,
        initial_capital=args.initial_capital,
        fee_rate=args.fee_rate,
        risk_free_rate=args.risk_free_rate,
    )

    print(f"Analyzed {len(data_files)} data files and {len(metrics)} strategy combinations.")
    print(f"Report: {BASE_DIR / 'analysis_report.md'}")
    print(f"Metrics: {OUTPUT_DIR / 'strategy_metrics.csv'}")
    print(f"Figures: {len(figure_paths)} files in {FIGURE_DIR}")


if __name__ == "__main__":
    main()
