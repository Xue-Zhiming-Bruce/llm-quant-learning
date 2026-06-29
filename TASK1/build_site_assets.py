"""Build static GitHub Pages assets for 688017.SH analysis."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import tushare as ts


BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
DOCS_DIR = REPO_DIR / "docs"
DOCS_ASSETS_DIR = DOCS_DIR / "assets"
DOCS_DATA_DIR = DOCS_DIR / "data"
DATA_PATH = BASE_DIR / "data" / "688017.SH_daily_20250629_20260629.csv"
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


CACHED_FUNDAMENTALS = {
    "stock_basic": {
        "ts_code": "688017.SH",
        "symbol": "688017",
        "name": "绿的谐波",
        "area": "江苏",
        "industry": "机械基件",
        "market": "科创板",
        "list_date": "20200828",
    },
    "daily_basic": {
        "ts_code": "688017.SH",
        "trade_date": "20260629",
        "close": 361.62,
        "turnover_rate": 7.4473,
        "turnover_rate_f": 11.386,
        "volume_ratio": 0.96,
        "pe": 533.0665,
        "pe_ttm": 484.7865,
        "pb": 18.5764,
        "ps": 116.163,
        "ps_ttm": 108.1812,
        "total_share": 18333.0125,
        "float_share": 18333.0125,
        "free_share": 11991.0809,
        "total_mv": 6629583.0762,
        "circ_mv": 6629583.0762,
    },
    "data_status": "使用 Tushare Pro stock_basic / daily_basic 的 2026-06-29 快照；利润表、资产负债表和现金流接口需要更高权限。",
}


def load_dotenv_if_exists(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def clean_number(value: object, digits: int = 2) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return round(value, digits)


def pct(value: float | None, digits: int = 2) -> float | None:
    if value is None:
        return None
    return round(value * 100, digits)


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for window in (5, 20, 60, 120):
        out[f"ma{window}"] = out["close"].rolling(window).mean()

    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    out["macd_diff"] = ema12 - ema26
    out["macd_dea"] = out["macd_diff"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = 2 * (out["macd_diff"] - out["macd_dea"])
    out["rsi14"] = compute_rsi(out["close"], 14)
    out["daily_return"] = out["close"].pct_change()
    out["volatility20"] = out["daily_return"].rolling(20).std() * math.sqrt(252)
    out["drawdown"] = out["close"] / out["close"].cummax() - 1
    out["volume_ma20"] = out["vol"].rolling(20).mean()
    return out


def return_over(df: pd.DataFrame, sessions: int) -> float | None:
    if len(df) <= sessions:
        return None
    return float(df["close"].iloc[-1] / df["close"].iloc[-1 - sessions] - 1)


def latest_change_pct(df: pd.DataFrame) -> float | None:
    if len(df) < 2:
        return None
    return float(df["close"].iloc[-1] / df["close"].iloc[-2] - 1)


def trend_label(last: pd.Series) -> str:
    close = float(last["close"])
    ma20 = clean_number(last.get("ma20"))
    ma60 = clean_number(last.get("ma60"))
    ma120 = clean_number(last.get("ma120"))

    if ma20 and ma60 and close > ma20 > ma60:
        return "中短期偏强"
    if ma20 and ma60 and close < ma20 < ma60:
        return "中短期转弱"
    if ma20 and close < ma20:
        return "短线回撤"
    if ma60 and close > ma60:
        return "趋势修复"
    if ma120 and close > ma120:
        return "长期趋势仍在"
    return "震荡观察"


def rsi_label(rsi: float | None) -> str:
    if rsi is None:
        return "数据不足"
    if rsi >= 70:
        return "高位区间"
    if rsi <= 30:
        return "低位区间"
    return "中性区间"


def macd_label(diff: float | None, dea: float | None, hist: float | None) -> str:
    if diff is None or dea is None or hist is None:
        return "数据不足"
    if diff > dea and hist > 0:
        return "多头动能"
    if diff < dea and hist < 0:
        return "空头动能"
    return "动能切换"


def draw_candlestick_chart(df: pd.DataFrame, output_path: Path) -> None:
    subset = df.tail(120).reset_index(drop=True)
    fig, (ax_price, ax_volume) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.04},
    )

    width = 0.62
    for index, row in subset.iterrows():
        open_price = row["open"]
        close_price = row["close"]
        high = row["high"]
        low = row["low"]
        is_up = close_price >= open_price
        color = "#d84f45" if is_up else "#14865f"
        lower = min(open_price, close_price)
        height = max(abs(close_price - open_price), 0.2)

        ax_price.vlines(index, low, high, color=color, linewidth=1)
        ax_price.add_patch(
            Rectangle(
                (index - width / 2, lower),
                width,
                height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.8,
                alpha=0.88,
            )
        )
        ax_volume.bar(index, row["vol"], color=color, width=width, alpha=0.65)

    for window, color in [(5, "#4e79a7"), (20, "#f2a541"), (60, "#7b6fd6")]:
        column = f"ma{window}"
        ax_price.plot(subset.index, subset[column], color=color, linewidth=1.5, label=f"MA{window}")

    tick_step = max(len(subset) // 8, 1)
    tick_positions = list(range(0, len(subset), tick_step))
    tick_labels = [subset.loc[i, "trade_date"].strftime("%Y-%m-%d") for i in tick_positions]
    ax_volume.set_xticks(tick_positions)
    ax_volume.set_xticklabels(tick_labels, rotation=0)

    ax_price.set_title("688017.SH K-Line with Moving Averages")
    ax_price.set_ylabel("Price")
    ax_price.grid(True, alpha=0.22)
    ax_price.legend(loc="upper left", frameon=False)
    ax_volume.set_ylabel("Volume")
    ax_volume.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def draw_technical_chart(df: pd.DataFrame, output_path: Path) -> None:
    fig, (ax_price, ax_rsi, ax_macd) = plt.subplots(
        3,
        1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2.6, 1, 1.1], "hspace": 0.08},
    )

    ax_price.plot(df["trade_date"], df["close"], color="#1f77b4", linewidth=1.8, label="Close")
    ax_price.plot(df["trade_date"], df["ma20"], color="#f2a541", linewidth=1.4, label="MA20")
    ax_price.plot(df["trade_date"], df["ma60"], color="#2ca58d", linewidth=1.4, label="MA60")
    ax_price.plot(df["trade_date"], df["ma120"], color="#7b6fd6", linewidth=1.4, label="MA120")
    ax_price.set_title("688017.SH Close Price and Technical Indicators")
    ax_price.set_ylabel("Price")
    ax_price.grid(True, alpha=0.22)
    ax_price.legend(loc="upper left", frameon=False, ncol=4)

    ax_rsi.plot(df["trade_date"], df["rsi14"], color="#936639", linewidth=1.4)
    ax_rsi.axhline(70, color="#d84f45", linestyle="--", linewidth=1)
    ax_rsi.axhline(30, color="#14865f", linestyle="--", linewidth=1)
    ax_rsi.set_ylabel("RSI14")
    ax_rsi.set_ylim(0, 100)
    ax_rsi.grid(True, alpha=0.2)

    colors = ["#d84f45" if value >= 0 else "#14865f" for value in df["macd_hist"].fillna(0)]
    ax_macd.bar(df["trade_date"], df["macd_hist"], color=colors, width=1.0, alpha=0.55, label="MACD Hist")
    ax_macd.plot(df["trade_date"], df["macd_diff"], color="#4e79a7", linewidth=1.2, label="DIF")
    ax_macd.plot(df["trade_date"], df["macd_dea"], color="#f2a541", linewidth=1.2, label="DEA")
    ax_macd.set_ylabel("MACD")
    ax_macd.grid(True, alpha=0.2)
    ax_macd.legend(loc="upper left", frameon=False, ncol=3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def query_tushare_fundamentals(token: str | None, ts_code: str) -> dict:
    fallback = CACHED_FUNDAMENTALS.copy()
    if not token:
        return fallback

    pro = ts.pro_api(token)
    try:
        basic = pro.stock_basic(
            ts_code=ts_code,
            fields="ts_code,symbol,name,area,industry,market,list_date",
        )
        daily_basic = pro.daily_basic(ts_code=ts_code, start_date="20250629", end_date="20260629")
    except Exception as exc:
        fallback["data_status"] = f"实时接口读取失败，已使用缓存快照。失败原因：{exc}"
        return fallback

    basic_record = basic.head(1).to_dict(orient="records")[0] if not basic.empty else fallback["stock_basic"]
    daily_record = daily_basic.head(1).to_dict(orient="records")[0] if not daily_basic.empty else {}
    return {
        "stock_basic": basic_record,
        "daily_basic": daily_record,
        "data_status": "Tushare stock_basic 与 daily_basic 已读取；财务报表接口需更高权限。",
    }


def build_analysis_payload(df: pd.DataFrame, fundamentals: dict) -> dict:
    last = df.iloc[-1]
    first = df.iloc[0]
    recent_high_idx = df["close"].idxmax()
    recent_low_idx = df["close"].idxmin()
    high_row = df.loc[recent_high_idx]
    low_row = df.loc[recent_low_idx]
    volume_ratio = None
    if clean_number(last.get("volume_ma20")):
        volume_ratio = float(last["vol"] / last["volume_ma20"])

    latest_daily_basic = fundamentals.get("daily_basic", {})
    market_cap_yi = None
    if latest_daily_basic.get("total_mv") is not None:
        market_cap_yi = clean_number(float(latest_daily_basic["total_mv"]) / 10000, 2)

    rsi_value = clean_number(last.get("rsi14"), 2)
    macd_diff = clean_number(last.get("macd_diff"), 3)
    macd_dea = clean_number(last.get("macd_dea"), 3)
    macd_hist = clean_number(last.get("macd_hist"), 3)

    close = float(last["close"])
    ma20 = clean_number(last.get("ma20"), 2)
    ma60 = clean_number(last.get("ma60"), 2)
    ma120 = clean_number(last.get("ma120"), 2)
    latest_change = latest_change_pct(df)
    one_year_return = float(close / first["close"] - 1)
    from_high = float(close / high_row["close"] - 1)

    technical_summary = [
        f"最新收盘价 {close:.2f} 元，过去一年区间涨幅 {one_year_return * 100:.2f}%。",
        f"当前价格较区间最高收盘价 {high_row['close']:.2f} 元回撤 {abs(from_high) * 100:.2f}%。",
        f"MA20/MA60/MA120 分别为 {ma20:.2f}/{ma60:.2f}/{ma120:.2f}，趋势标签为「{trend_label(last)}」。",
        f"RSI14 为 {rsi_value:.2f}，处于「{rsi_label(rsi_value)}」；MACD 状态为「{macd_label(macd_diff, macd_dea, macd_hist)}」。",
    ]

    kline_summary = [
        f"近 120 个交易日 K 线显示股价经历快速拉升后高位波动，区间最高收盘日为 {high_row['trade_date'].strftime('%Y-%m-%d')}。",
        f"最新交易日涨跌幅为 {latest_change * 100:.2f}%，成交量约为 20 日均量的 {volume_ratio:.2f} 倍。",
        "红色 K 线代表收盘价高于开盘价，绿色 K 线代表收盘价低于开盘价；均线用于观察趋势方向和支撑压力。",
    ]

    fundamental_summary = [
        "公司标签显示其位于江苏，所属行业为机械基件，上市板块为科创板。",
        "主营方向聚焦精密传动部件，谐波减速器与机器人产业链景气度相关性较高。",
        "估值快照显示 PE(TTM)、PB 与 PS(TTM) 均处于较高绝对水平，说明市场对成长性已有较强预期。",
        "基本面跟踪应继续关注收入增速、毛利率、净利率、研发投入、产能利用率和大客户结构。",
    ]

    recent_rows = []
    for _, row in df.tail(12).sort_values("trade_date", ascending=False).iterrows():
        recent_rows.append(
            {
                "trade_date": row["trade_date"].strftime("%Y-%m-%d"),
                "open": clean_number(row["open"], 2),
                "high": clean_number(row["high"], 2),
                "low": clean_number(row["low"], 2),
                "close": clean_number(row["close"], 2),
                "pct_chg": clean_number(row["pct_chg"], 2),
                "vol": clean_number(row["vol"], 2),
            }
        )

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "disclaimer": "本页面仅用于量化学习与数据分析练习，不构成任何投资建议。",
        "stock": {
            **fundamentals.get("stock_basic", {}),
            "display_name": "绿的谐波",
            "business_note": "公司聚焦精密传动与谐波减速器相关产品，应用场景与机器人、自动化设备等产业链相关。",
        },
        "period": {
            "start": first["trade_date"].strftime("%Y-%m-%d"),
            "end": last["trade_date"].strftime("%Y-%m-%d"),
            "sessions": int(len(df)),
        },
        "price": {
            "latest_date": last["trade_date"].strftime("%Y-%m-%d"),
            "latest_close": clean_number(close, 2),
            "latest_change_pct": pct(latest_change),
            "one_year_return_pct": pct(one_year_return),
            "return_20d_pct": pct(return_over(df, 20)),
            "return_60d_pct": pct(return_over(df, 60)),
            "period_high_close": clean_number(high_row["close"], 2),
            "period_high_date": high_row["trade_date"].strftime("%Y-%m-%d"),
            "period_low_close": clean_number(low_row["close"], 2),
            "period_low_date": low_row["trade_date"].strftime("%Y-%m-%d"),
            "from_high_pct": pct(from_high),
            "max_drawdown_pct": pct(float(df["drawdown"].min())),
        },
        "technical": {
            "trend_label": trend_label(last),
            "ma5": clean_number(last.get("ma5"), 2),
            "ma20": ma20,
            "ma60": ma60,
            "ma120": ma120,
            "rsi14": rsi_value,
            "rsi_label": rsi_label(rsi_value),
            "macd_diff": macd_diff,
            "macd_dea": macd_dea,
            "macd_hist": macd_hist,
            "macd_label": macd_label(macd_diff, macd_dea, macd_hist),
            "volatility20_pct": pct(clean_number(last.get("volatility20"), 6)),
            "volume_vs_ma20": clean_number(volume_ratio, 2),
            "summary": technical_summary,
        },
        "kline": {"summary": kline_summary},
        "fundamental": {
            "latest_daily_basic": {
                "trade_date": latest_daily_basic.get("trade_date"),
                "pe_ttm": clean_number(latest_daily_basic.get("pe_ttm"), 2),
                "pe": clean_number(latest_daily_basic.get("pe"), 2),
                "pb": clean_number(latest_daily_basic.get("pb"), 2),
                "ps_ttm": clean_number(latest_daily_basic.get("ps_ttm"), 2),
                "turnover_rate": clean_number(latest_daily_basic.get("turnover_rate"), 2),
                "volume_ratio": clean_number(latest_daily_basic.get("volume_ratio"), 2),
                "total_mv_yi": market_cap_yi,
            },
            "summary": fundamental_summary,
            "data_status": fundamentals.get("data_status"),
            "sources": [
                {"name": "Tushare Pro stock_basic / daily_basic", "url": "https://www.tushare.pro/"},
                {"name": "上海证券交易所 688017 公司信息", "url": "https://www.sse.com.cn/assortment/stock/list/info/company/index.shtml?COMPANY_CODE=688017"},
                {"name": "绿的谐波公司官网", "url": "https://www.leaderdrive.com/"},
            ],
        },
        "recent_rows": recent_rows,
        "assets": {
            "kline": "assets/688017_kline.png",
            "technical": "assets/688017_technical.png",
            "csv": "data/688017_daily.csv",
        },
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build static assets for the GitHub Pages dashboard.")
    parser.add_argument("--token", default=None, help="Optional Tushare token for basic company and valuation data.")
    args = parser.parse_args()

    load_dotenv_if_exists(BASE_DIR / ".env")
    DOCS_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = add_indicators(df.sort_values("trade_date").reset_index(drop=True))

    fundamentals = query_tushare_fundamentals(args.token or os.getenv("TUSHARE_TOKEN"), "688017.SH")
    payload = build_analysis_payload(df, fundamentals)

    df.to_csv(DOCS_DATA_DIR / "688017_daily.csv", index=False, encoding="utf-8-sig")
    write_json(DOCS_DATA_DIR / "analysis.json", payload)
    draw_candlestick_chart(df, DOCS_ASSETS_DIR / "688017_kline.png")
    draw_technical_chart(df, DOCS_ASSETS_DIR / "688017_technical.png")
    print(f"Built site assets for {payload['stock']['display_name']} through {payload['period']['end']}.")


if __name__ == "__main__":
    main()
