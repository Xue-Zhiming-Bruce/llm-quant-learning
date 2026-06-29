"""Fetch daily A-share data from Tushare Pro and plot close prices."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIGURE_DIR = BASE_DIR / "figures"
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd
import tushare as ts


def load_dotenv_if_exists(path: Path) -> None:
    """Load simple KEY=VALUE lines from a local .env file."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def yyyymmdd(value: str) -> str:
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must be in YYYYMMDD format") from exc
    return value


def default_start_date() -> str:
    return (datetime.today() - timedelta(days=365)).strftime("%Y%m%d")


def default_end_date() -> str:
    return datetime.today().strftime("%Y%m%d")


def fetch_daily_data(token: str, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    pro = ts.pro_api(token)
    df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)

    if df.empty:
        raise RuntimeError(
            f"No data returned for {ts_code} from {start_date} to {end_date}. "
            "Please check the stock code, date range, Tushare token, or account permission."
        )

    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    return df.sort_values("trade_date").reset_index(drop=True)


def save_csv(df: pd.DataFrame, ts_code: str, start_date: str, end_date: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / f"{ts_code}_daily_{start_date}_{end_date}.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def plot_close_price(df: pd.DataFrame, ts_code: str, start_date: str, end_date: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURE_DIR / f"{ts_code}_close_{start_date}_{end_date}.png"

    plt.figure(figsize=(12, 6))
    plt.plot(df["trade_date"], df["close"], linewidth=1.8)
    plt.title(f"{ts_code} Daily Close Price")
    plt.xlabel("Trade Date")
    plt.ylabel("Close Price")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch one year of daily A-share data from Tushare Pro."
    )
    parser.add_argument(
        "--ts-code",
        default="688017.SH",
        help="Tushare stock code, such as 688017.SH or 600519.SH.",
    )
    parser.add_argument(
        "--start-date",
        type=yyyymmdd,
        default=default_start_date(),
        help="Start date in YYYYMMDD format. Default: one year before today.",
    )
    parser.add_argument(
        "--end-date",
        type=yyyymmdd,
        default=default_end_date(),
        help="End date in YYYYMMDD format. Default: today.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Tushare token. If omitted, reads TUSHARE_TOKEN from environment or .env.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv_if_exists(BASE_DIR / ".env")
    args = parse_args()

    token = args.token or os.getenv("TUSHARE_TOKEN")
    if not token:
        raise SystemExit(
            "Missing Tushare token. Set TUSHARE_TOKEN, create TASK1/.env, or pass --token."
        )

    df = fetch_daily_data(token, args.ts_code, args.start_date, args.end_date)
    csv_path = save_csv(df, args.ts_code, args.start_date, args.end_date)
    figure_path = plot_close_price(df, args.ts_code, args.start_date, args.end_date)

    print(f"Fetched {len(df)} rows for {args.ts_code}.")
    print(f"CSV saved to: {csv_path}")
    print(f"Figure saved to: {figure_path}")


if __name__ == "__main__":
    main()
