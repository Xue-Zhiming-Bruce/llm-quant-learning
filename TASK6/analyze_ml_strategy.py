"""TASK6: machine-learning stock-selection strategy and backtest.

The bundled model_data.csv is a quarterly cross-sectional data set.  Each row
describes one stock at one quarter end and ``Next_Ret`` is that stock's return
over the following quarter.  The script deliberately splits by date (rather
than randomly) so that test observations are always later than training data.

Outputs are written to TASK6/output and TASK6/figures.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
FIGURE_DIR = BASE_DIR / "figures"
CACHE_DIR = Path(tempfile.gettempdir()) / "llm_quant_task6_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor


RANDOM_STATE = 42
TARGET = "Next_Ret"
ID_COLUMNS = ["Date", "Code"]
MODEL_ORDER = ["Ridge", "Decision Tree", "Random Forest"]


@dataclass(frozen=True)
class BacktestConfig:
    train_fraction: float = 0.70
    selection_fraction: float = 0.20
    transaction_cost: float = 0.001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=BASE_DIR / "model_data.csv")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--selection-fraction", type=float, default=0.20)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    return parser.parse_args()


def load_samples(path: Path) -> pd.DataFrame:
    """Load and validate the stored quarterly model sample."""

    if not path.exists():
        raise FileNotFoundError(f"Sample file not found: {path}")
    data = pd.read_csv(path)
    required = {"Date", "Code", TARGET}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data["Date"] = pd.to_datetime(data["Date"], errors="raise")
    data["Code"] = data["Code"].astype(str).str.zfill(6)
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    if data.duplicated(ID_COLUMNS).any():
        raise ValueError("Date-Code pairs must be unique.")
    data = data.dropna(subset=[TARGET]).sort_values(ID_COLUMNS).reset_index(drop=True)
    if data["Date"].nunique() < 4:
        raise ValueError("At least four distinct dates are required for a time split.")
    return data


def safe_inverse(series: pd.Series) -> pd.Series:
    """Return a bounded reciprocal; near-zero denominators become missing."""

    numeric = pd.to_numeric(series, errors="coerce")
    return 1.0 / numeric.where(numeric.abs() >= 1e-6)


def cross_sectional_zscore(series: pd.Series) -> pd.Series:
    """Winsorize a single-quarter factor and standardize it cross-sectionally."""

    numeric = pd.to_numeric(series, errors="coerce")
    lower, upper = numeric.quantile([0.01, 0.99])
    clipped = numeric.clip(lower, upper)
    std = clipped.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (clipped - clipped.mean()) / std


def derive_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Derive economically interpretable factors using information at each date."""

    frame = data.copy()
    raw_features = [c for c in frame.columns if c not in ID_COLUMNS + [TARGET]]
    frame[raw_features] = frame[raw_features].apply(pd.to_numeric, errors="coerce")

    frame["log_MV"] = np.log1p(frame["MV"].clip(lower=0))
    frame["earnings_yield"] = safe_inverse(frame["市盈率PE(TTM)"])
    frame["book_to_price"] = safe_inverse(frame["市净率PB(MRQ)"])
    frame["sales_yield"] = safe_inverse(frame["市销率PS(TTM)"])
    frame["growth_composite"] = frame[
        ["净利润同比增长率", "营业利润(同比增长率)", "营业总收入(同比增长率)"]
    ].mean(axis=1)
    frame["cashflow_growth_composite"] = frame[
        ["现金净流量同比增长率", "经营活动产生的现金流量净额(同比增长率)"]
    ].mean(axis=1)
    frame["fundamental_growth_gap"] = (
        frame["净利润同比增长率"] - frame["营业总收入(同比增长率)"]
    )

    derived = [
        "log_MV",
        "earnings_yield",
        "book_to_price",
        "sales_yield",
        "growth_composite",
        "cashflow_growth_composite",
        "fundamental_growth_gap",
    ]
    model_features = raw_features + derived

    # Cross-sectional transforms reduce the influence of extreme accounting ratios.
    z_columns: list[str] = []
    for column in model_features:
        z_name = f"z_{column}"
        frame[z_name] = frame.groupby("Date", group_keys=False)[column].transform(
            cross_sectional_zscore
        )
        z_columns.append(z_name)
    return frame, z_columns


def split_by_date(
    data: pd.DataFrame, train_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.Timestamp], list[pd.Timestamp]]:
    """Make a chronological train/test split using complete quarters."""

    if not 0.50 <= train_fraction <= 0.90:
        raise ValueError("train_fraction must be between 0.50 and 0.90")
    dates = sorted(data["Date"].unique())
    cut = max(2, min(len(dates) - 1, int(np.floor(len(dates) * train_fraction))))
    train_dates, test_dates = dates[:cut], dates[cut:]
    train = data[data["Date"].isin(train_dates)].copy()
    test = data[data["Date"].isin(test_dates)].copy()
    return train, test, list(train_dates), list(test_dates)


def build_models() -> dict[str, object]:
    """Build one linear baseline and two requested tree-based regressors."""

    ridge = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", Ridge(alpha=10.0)),
        ]
    )
    # Standardizing the target makes the regularization settings more stable.
    ridge_model = TransformedTargetRegressor(
        regressor=ridge, transformer=StandardScaler()
    )
    tree = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                DecisionTreeRegressor(
                    max_depth=5,
                    min_samples_leaf=100,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    forest = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=30,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return {"Ridge": ridge_model, "Decision Tree": tree, "Random Forest": forest}


def regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    """Calculate prediction metrics, including rank IC for stock selection."""

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
        "Rank_IC": pd.Series(actual).corr(pd.Series(predicted), method="spearman"),
        "Directional_Accuracy": np.mean((np.asarray(predicted) > 0) == (actual.to_numpy() > 0)),
    }


def add_positions(predictions: pd.DataFrame, selection_fraction: float) -> pd.DataFrame:
    """Select the highest predicted-return fraction independently each quarter."""

    if not 0 < selection_fraction <= 1:
        raise ValueError("selection_fraction must be in (0, 1].")
    result = predictions.copy()
    result["Predicted_Percentile"] = result.groupby("Date")["Predicted_Return"].rank(
        pct=True, method="first"
    )
    result["Position"] = (
        result["Predicted_Percentile"] > 1.0 - selection_fraction
    ).astype(int)
    return result


def calculate_quarterly_returns(
    positioned: pd.DataFrame, transaction_cost: float
) -> pd.DataFrame:
    """Calculate equal-weight portfolio returns and turnover-based costs."""

    previous_holdings: set[str] = set()
    rows: list[dict[str, object]] = []
    for date, quarter in positioned.groupby("Date", sort=True):
        selected = quarter.loc[quarter["Position"] == 1]
        holdings = set(selected["Code"])
        gross = selected[TARGET].mean() if holdings else 0.0
        benchmark = quarter[TARGET].mean()
        if not previous_holdings:
            turnover = 1.0
        else:
            union = previous_holdings | holdings
            turnover = 1.0 - len(previous_holdings & holdings) / max(len(union), 1)
        cost = turnover * transaction_cost
        rows.append(
            {
                "Date": date,
                "Selected_Count": len(selected),
                "Universe_Count": len(quarter),
                "Gross_Return": gross,
                "Turnover": turnover,
                "Transaction_Cost": cost,
                "Strategy_Return": gross - cost,
                "Benchmark_Return": benchmark,
                "Excess_Return": gross - cost - benchmark,
            }
        )
        previous_holdings = holdings
    result = pd.DataFrame(rows)
    result["Strategy_NAV"] = (1 + result["Strategy_Return"]).cumprod()
    result["Benchmark_NAV"] = (1 + result["Benchmark_Return"]).cumprod()
    return result


def backtest_metrics(returns: pd.Series) -> dict[str, float]:
    """Calculate core metrics for quarterly returns."""

    returns = pd.Series(returns, dtype=float).dropna()
    nav = (1 + returns).cumprod()
    periods = len(returns)
    total = nav.iloc[-1] - 1 if periods else np.nan
    annualized = nav.iloc[-1] ** (4 / periods) - 1 if periods else np.nan
    volatility = returns.std(ddof=1) * np.sqrt(4) if periods > 1 else np.nan
    sharpe = (returns.mean() * 4) / volatility if volatility and volatility > 0 else np.nan
    # Include the initial capital of 1.0, otherwise a first-period loss is missed.
    nav_with_start = pd.concat([pd.Series([1.0]), nav], ignore_index=True)
    drawdown = nav_with_start / nav_with_start.cummax() - 1
    return {
        "Observations": periods,
        "Total_Return": total,
        "Annualized_Return": annualized,
        "Annualized_Volatility": volatility,
        "Sharpe_Ratio_RF0": sharpe,
        "Max_Drawdown": drawdown.min() if periods else np.nan,
        "Positive_Quarter_Rate": (returns > 0).mean() if periods else np.nan,
        "Average_Quarterly_Return": returns.mean() if periods else np.nan,
    }


def save_plots(
    quarterly_by_model: dict[str, pd.DataFrame], metrics: pd.DataFrame
) -> None:
    """Create net-value, quarterly-return, and model-comparison figures."""

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    first = next(iter(quarterly_by_model.values()))
    ax.plot(first["Date"], first["Benchmark_NAV"], "k--", label="Equal-weight benchmark")
    for name, quarterly in quarterly_by_model.items():
        ax.plot(quarterly["Date"], quarterly["Strategy_NAV"], marker="o", label=name)
    ax.set(title="Test-set cumulative net value", xlabel="Quarter", ylabel="Net value")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "cumulative_nav.png", dpi=180)
    plt.close(fig)

    return_table = pd.DataFrame(
        {name: q.set_index("Date")["Strategy_Return"] for name, q in quarterly_by_model.items()}
    )
    return_table["Benchmark"] = first.set_index("Date")["Benchmark_Return"]
    ax = return_table.plot(kind="bar", figsize=(11, 6), width=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(title="Quarterly returns in the test set", xlabel="Quarter", ylabel="Return")
    ax.set_xticklabels([d.strftime("%Y-Q") + str((d.month - 1) // 3 + 1) for d in return_table.index], rotation=0)
    ax.legend(ncol=2)
    ax.figure.tight_layout()
    ax.figure.savefig(FIGURE_DIR / "quarterly_returns.png", dpi=180)
    plt.close(ax.figure)

    plot_metrics = metrics.set_index("Model")[["Rank_IC", "Directional_Accuracy"]]
    ax = plot_metrics.plot(kind="bar", figsize=(9, 5), rot=0)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(title="Out-of-sample prediction comparison", ylabel="Score", xlabel="Model")
    ax.figure.tight_layout()
    ax.figure.savefig(FIGURE_DIR / "model_comparison.png", dpi=180)
    plt.close(ax.figure)


def feature_importance(model: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    return (
        pd.DataFrame({"Feature": feature_names, "Importance": estimator.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    config = BacktestConfig(
        train_fraction=args.train_fraction,
        selection_fraction=args.selection_fraction,
        transaction_cost=args.transaction_cost,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_samples(args.data)
    data, features = derive_features(raw)
    train, test, train_dates, test_dates = split_by_date(data, config.train_fraction)
    x_train, y_train = train[features], train[TARGET]
    x_test, y_test = test[features], test[TARGET]

    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    backtest_rows: list[dict[str, object]] = []
    quarterly_by_model: dict[str, pd.DataFrame] = {}

    for name, model in build_models().items():
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        row = {"Model": name, **regression_metrics(y_test.reset_index(drop=True), predicted)}
        metric_rows.append(row)

        predictions = test[ID_COLUMNS + [TARGET]].copy()
        predictions["Model"] = name
        predictions["Predicted_Return"] = predicted
        predictions = add_positions(predictions, config.selection_fraction)
        prediction_rows.append(predictions)

        quarterly = calculate_quarterly_returns(predictions, config.transaction_cost)
        quarterly.insert(0, "Model", name)
        quarterly_by_model[name] = quarterly
        strategy_stats = backtest_metrics(quarterly["Strategy_Return"])
        backtest_rows.append({"Model": name, "Portfolio": "Strategy", **strategy_stats})

        if name in {"Decision Tree", "Random Forest"}:
            feature_importance(model, features).to_csv(
                OUTPUT_DIR / f"feature_importance_{name.lower().replace(' ', '_')}.csv",
                index=False,
            )

    benchmark_stats = backtest_metrics(next(iter(quarterly_by_model.values()))["Benchmark_Return"])
    backtest_rows.append({"Model": "Equal Weight", "Portfolio": "Benchmark", **benchmark_stats})

    metrics = pd.DataFrame(metric_rows)
    predictions_all = pd.concat(prediction_rows, ignore_index=True)
    quarterly_all = pd.concat(quarterly_by_model.values(), ignore_index=True)
    backtest = pd.DataFrame(backtest_rows)
    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    predictions_all.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    quarterly_all.to_csv(OUTPUT_DIR / "quarterly_returns.csv", index=False)
    backtest.to_csv(OUTPUT_DIR / "backtest_metrics.csv", index=False)
    pd.DataFrame({"Feature": features}).to_csv(OUTPUT_DIR / "feature_definitions.csv", index=False)
    save_plots(quarterly_by_model, metrics)

    metadata = {
        "source": str(args.data.resolve()),
        "rows": len(data),
        "stocks": int(data["Code"].nunique()),
        "features": len(features),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_dates": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in train_dates],
        "test_dates": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in test_dates],
        "selection_fraction": config.selection_fraction,
        "transaction_cost": config.transaction_cost,
        "target": "Next_Ret: return from the observation date to the next quarter",
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    best = backtest.loc[backtest["Portfolio"] == "Strategy"].sort_values(
        "Total_Return", ascending=False
    ).iloc[0]
    print(f"Train: {train_dates[0]:%Y-%m-%d} to {train_dates[-1]:%Y-%m-%d} ({len(train):,} rows)")
    print(f"Test:  {test_dates[0]:%Y-%m-%d} to {test_dates[-1]:%Y-%m-%d} ({len(test):,} rows)")
    print(metrics.to_string(index=False))
    print(backtest.to_string(index=False))
    print(f"Best test-set strategy by total return: {best['Model']}")


if __name__ == "__main__":
    main()
