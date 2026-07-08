"""Backtest a double moving-average strategy on stored stock price data.

Outputs:
- output/strategy_metrics.csv
- output/trading_signals.csv
- output/<ts_code>_ma_<short>_<long>_backtest.csv
- figures/<ts_code>_ma_<short>_<long>_strategy.png
- analysis_report.md
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
OUTPUT_DIR = BASE_DIR / "output"
FIGURE_DIR = BASE_DIR / "figures"
FONT_CACHE_DIR = BASE_DIR / ".cache"
FONT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(FONT_CACHE_DIR))

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

DEFAULT_WINDOW_PAIRS = [(5, 15), (10, 30), (20, 60)]
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


def parse_window_pairs(value: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for raw_pair in value.split(","):
        raw_pair = raw_pair.strip()
        if not raw_pair:
            continue

        delimiter = ":" if ":" in raw_pair else "/"
        parts = [part.strip() for part in raw_pair.split(delimiter)]
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                f"Invalid window pair '{raw_pair}'. Use a format like 5:15,10:30."
            )

        short_window, long_window = int(parts[0]), int(parts[1])
        if short_window <= 0 or long_window <= 0:
            raise argparse.ArgumentTypeError("Moving-average windows must be positive.")
        if short_window >= long_window:
            raise argparse.ArgumentTypeError(
                f"Short window must be smaller than long window: {raw_pair}"
            )
        pairs.append((short_window, long_window))

    if not pairs:
        raise argparse.ArgumentTypeError("At least one moving-average window pair is required.")
    return pairs


def resolve_data_file(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path

    project_path = PROJECT_ROOT / path
    if project_path.exists():
        return project_path
    return Path.cwd() / path


def parse_trade_date(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip()
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

    df = df.dropna(subset=["close"]).copy()
    df = df.sort_values(["ts_code", "trade_date"]).drop_duplicates(
        subset=["ts_code", "trade_date"],
        keep="last",
    )
    df = df.reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{path.name} has no usable rows after cleaning.")

    ts_code = str(df["ts_code"].iloc[0])
    df["source_file"] = str(path.relative_to(PROJECT_ROOT))
    df["stock_name"] = stock_name_from_path(path, ts_code)
    return df


def add_strategy_columns(
    df: pd.DataFrame,
    short_window: int,
    long_window: int,
    initial_capital: float,
    fee_rate: float,
) -> pd.DataFrame:
    result = df.copy()
    result["short_ma"] = result["close"].rolling(
        window=short_window,
        min_periods=short_window,
    ).mean()
    result["long_ma"] = result["close"].rolling(
        window=long_window,
        min_periods=long_window,
    ).mean()

    ready = result["long_ma"].notna()
    result["target_position"] = np.where(
        ready & (result["short_ma"] > result["long_ma"]),
        1.0,
        0.0,
    )

    previous_target = result["target_position"].shift(1).fillna(0.0)
    result["trade_signal"] = 0
    result.loc[
        (result["target_position"] == 1.0) & (previous_target == 0.0),
        "trade_signal",
    ] = 1
    result.loc[
        (result["target_position"] == 0.0) & (previous_target == 1.0),
        "trade_signal",
    ] = -1

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
    result["short_window"] = short_window
    result["long_window"] = long_window
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
    short_window: int,
    long_window: int,
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
        "short_window": short_window,
        "long_window": long_window,
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
                "close",
                "short_ma",
                "long_ma",
                "short_window",
                "long_window",
            ]
        )

    events["action"] = np.where(events["trade_signal"] == 1, "BUY", "SELL")
    return events[
        [
            "stock_name",
            "ts_code",
            "trade_date",
            "action",
            "close",
            "short_ma",
            "long_ma",
            "short_window",
            "long_window",
        ]
    ]


def plot_strategy(strategy_df: pd.DataFrame, output_path: Path) -> None:
    dates = strategy_df["trade_date"]
    buys = strategy_df[strategy_df["trade_signal"] == 1]
    sells = strategy_df[strategy_df["trade_signal"] == -1]

    title = (
        f"{strategy_df['ts_code'].iloc[0]} MA"
        f"({strategy_df['short_window'].iloc[0]}, {strategy_df['long_window'].iloc[0]})"
    )

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.3, 1.0]},
    )

    axes[0].plot(dates, strategy_df["close"], label="Close", color="#1f77b4", linewidth=1.3)
    axes[0].plot(
        dates,
        strategy_df["short_ma"],
        label=f"Short MA ({strategy_df['short_window'].iloc[0]})",
        color="#ff7f0e",
        linewidth=1.1,
    )
    axes[0].plot(
        dates,
        strategy_df["long_ma"],
        label=f"Long MA ({strategy_df['long_window'].iloc[0]})",
        color="#2ca02c",
        linewidth=1.1,
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
    axes[0].set_title(f"{title}: Price, Moving Averages and Signals")
    axes[0].set_ylabel("Price")
    axes[0].legend(loc="upper left", ncol=5, fontsize=9)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(
        dates,
        strategy_df["strategy_equity"],
        label="Strategy equity",
        color="#d62728",
        linewidth=1.3,
    )
    axes[1].plot(
        dates,
        strategy_df["benchmark_equity"],
        label="Buy-and-hold equity",
        color="#1f77b4",
        linewidth=1.1,
    )
    axes[1].set_ylabel("Equity")
    axes[1].legend(loc="upper left", fontsize=9)
    axes[1].grid(True, alpha=0.25)

    axes[2].fill_between(
        mdates.date2num(dates),
        strategy_df["drawdown"].to_numpy(dtype=float),
        0,
        color="#8c564b",
        alpha=0.35,
    )
    axes[2].set_ylabel("Drawdown")
    axes[2].set_xlabel("Trade Date")
    axes[2].grid(True, alpha=0.25)
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[2].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))

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


def relative_path(path: Path) -> str:
    return str(path.relative_to(BASE_DIR))


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
            f"{int(row['short_window'])}/{int(row['long_window'])}，"
            f"策略累计回报 {format_percent(row['cumulative_return'])}，"
            f"同期买入持有回报 {format_percent(row['benchmark_return'])}，"
            f"{comparison}基准 {format_percent(abs(row['excess_return_vs_benchmark']))}。"
        )

    return "\n".join(lines)


def build_report(
    metrics: pd.DataFrame,
    figure_paths: list[Path],
    data_files: list[Path],
    window_pairs: list[tuple[int, int]],
    fee_rate: float,
) -> str:
    metrics_for_table = metrics.sort_values(["stock_name", "short_window", "long_window"])
    metric_rows = []
    for _, row in metrics_for_table.iterrows():
        metric_rows.append(
            [
                row["stock_name"],
                row["ts_code"],
                f"{row['start_date']} 至 {row['end_date']}",
                f"{int(row['short_window'])}/{int(row['long_window'])}",
                format_percent(row["cumulative_return"]),
                format_percent(row["benchmark_return"]),
                format_percent(row["max_drawdown"]),
                format_value(row["sharpe_ratio"]),
                format_percent(row["annualized_volatility"]),
                int(row["trade_signals"]),
                format_percent(row["exposure_ratio"]),
            ]
        )

    best = metrics.sort_values("cumulative_return", ascending=False).iloc[0]
    worst_mdd = metrics.sort_values("max_drawdown").iloc[0]
    data_lines = "\n".join(f"- `{path.relative_to(PROJECT_ROOT)}`" for path in data_files)
    window_text = "、".join(f"{short}/{long}" for short, long in window_pairs)
    figure_lines = "\n".join(f"- `{relative_path(path)}`" for path in figure_paths)

    return f"""# TASK3：双均线策略与回测分析

## 1. 双均线策略与金叉、死叉

双均线策略是一类典型的趋势跟随策略。它同时计算短周期均线和长周期均线：短均线对近期价格变化更敏感，长均线更平滑，用来过滤一部分短期噪声。

- 金叉：短均线从下方向上穿过长均线，说明近期价格强于较长周期趋势，通常被视为买入或开仓信号。
- 死叉：短均线从上方向下穿过长均线，说明近期价格弱于较长周期趋势，通常被视为卖出或离场信号。
- 策略直觉：上涨趋势中尽量持仓，下跌或弱势阶段尽量空仓。
- 主要缺点：均线是滞后指标，在震荡行情中容易频繁金叉、死叉，产生假信号和交易成本。

## 2. 回测指标说明

累计回报（Cumulative Return）衡量从回测开始到结束总共赚了多少：

```text
Cumulative Return = Final Equity / Initial Equity - 1
```

最大回撤（Maximum Drawdown, MDD）衡量资金曲线从历史高点到之后低点的最大跌幅：

```text
Drawdown_t = Equity_t / max(Equity_0 ... Equity_t) - 1
MDD = min(Drawdown_t)
```

MDD 越接近 0，说明历史最大亏损幅度越小；例如 -20% 表示资金从阶段高点最多回落过 20%。

夏普比率（Sharpe Ratio）衡量单位波动承担下获得的超额收益，本次使用日收益率并按 252 个交易日年化，默认无风险利率为 0：

```text
Sharpe = sqrt(252) * mean(Daily Return - Risk Free Rate / 252)
         / std(Daily Return - Risk Free Rate / 252)
```

夏普比率越高，表示收益相对波动更有效率；但它依赖历史样本，也不能反映所有尾部风险。

## 3. Python 实现

脚本位置：`analyze_double_ma_strategy.py`

默认运行：

```bash
python TASK3/analyze_double_ma_strategy.py
```

自定义均线周期示例：

```bash
python TASK3/analyze_double_ma_strategy.py --window-pairs 5:15,8:21,20:60
```

只测试某一个数据文件示例：

```bash
python TASK3/analyze_double_ma_strategy.py --data-file TASK2/平安集团行情数据.csv
```

脚本完成了以下步骤：

1. 加载已存储的股价 CSV，并兼容 `YYYYMMDD` 与 `YYYY-MM-DD` 两种日期格式；
2. 设置短均线和长均线周期，本次测试参数为 {window_text}；
3. 计算短均线、长均线、金叉买入信号和死叉卖出信号；
4. 绘制股价、长短均线、买入信号、卖出信号、策略资金曲线和回撤曲线；
5. 模拟交易并计算累计回报、最大回撤、夏普比率等指标。

本次回测假设：

- 初始资金为 100000；
- 只做多，不做空；
- 金叉/死叉在当天收盘后确认，下一交易日才改变持仓；
- 满仓持有或空仓等待，不做仓位分层；
- 单边交易成本为 {fee_rate * 100:.2f}%。

读取的数据文件：

{data_lines}

## 4. 不同股票与均线周期结果

{markdown_table(
    [
        "股票",
        "代码",
        "回测区间",
        "均线周期",
        "策略累计回报",
        "买入持有回报",
        "最大回撤",
        "夏普比率",
        "年化波动",
        "信号次数",
        "持仓比例",
    ],
    metric_rows,
)}

表现最好的组合是 {best['stock_name']}（{best['ts_code']}）的 {int(best['short_window'])}/{int(best['long_window'])} 均线，策略累计回报为 {format_percent(best['cumulative_return'])}。最大回撤最深的组合是 {worst_mdd['stock_name']}（{worst_mdd['ts_code']}）的 {int(worst_mdd['short_window'])}/{int(worst_mdd['long_window'])} 均线，MDD 为 {format_percent(worst_mdd['max_drawdown'])}。

各股票最优参数观察：

{build_observation_text(metrics)}

生成图形：

{figure_lines}

完整输出文件：

- `output/strategy_metrics.csv`
- `output/trading_signals.csv`
- `output/<ts_code>_ma_<short>_<long>_backtest.csv`
- `figures/<ts_code>_ma_<short>_<long>_strategy.png`

## 5. 双均线策略适用场景与心得

双均线策略更适合趋势较明确、上涨或下跌持续性较强的市场。因为短均线上穿长均线后，策略需要价格继续沿趋势运行，才能弥补信号滞后和交易成本。

短周期组合，例如 5/15，反应更快，能更早进入趋势，也会更早离场；缺点是信号更频繁，震荡市中容易来回止损。长周期组合，例如 20/60，信号更少、更平滑，抗噪声能力更强；缺点是反应慢，可能错过趋势初期，也可能在趋势结束后较晚卖出。

从本次实验可以看到，不同股票对参数非常敏感，同一组均线在一只股票上表现较好，不代表能直接迁移到另一只股票。实际使用时应结合成交量、波动率、市场环境和风险控制，并保留样本外测试，避免只根据历史表现过度调参。
"""


def run_analysis(
    data_files: list[Path],
    window_pairs: list[tuple[int, int]],
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

        for short_window, long_window in window_pairs:
            strategy_df = add_strategy_columns(
                price_df,
                short_window,
                long_window,
                initial_capital,
                fee_rate,
            )
            metrics_records.append(
                calculate_metrics(
                    strategy_df,
                    short_window,
                    long_window,
                    initial_capital,
                    fee_rate,
                    risk_free_rate,
                )
            )
            signal_frames.append(signal_events(strategy_df))

            backtest_output_path = OUTPUT_DIR / f"{ts_code}_ma_{short_window}_{long_window}_backtest.csv"
            strategy_df.to_csv(backtest_output_path, index=False, encoding="utf-8-sig")

            figure_path = FIGURE_DIR / f"{ts_code}_ma_{short_window}_{long_window}_strategy.png"
            plot_strategy(strategy_df, figure_path)
            figure_paths.append(figure_path)

    metrics = pd.DataFrame(metrics_records)
    metrics.to_csv(OUTPUT_DIR / "strategy_metrics.csv", index=False, encoding="utf-8-sig")

    if signal_frames:
        signals = pd.concat(signal_frames, ignore_index=True)
    else:
        signals = pd.DataFrame()
    signals.to_csv(OUTPUT_DIR / "trading_signals.csv", index=False, encoding="utf-8-sig")

    report = build_report(metrics, figure_paths, data_files, window_pairs, fee_rate)
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
        "--window-pairs",
        type=parse_window_pairs,
        default=DEFAULT_WINDOW_PAIRS,
        help="Comma-separated MA pairs, for example: 5:15,10:30,20:60.",
    )
    parser.add_argument("--initial-capital", type=float, default=100000.0)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        window_pairs=args.window_pairs,
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
