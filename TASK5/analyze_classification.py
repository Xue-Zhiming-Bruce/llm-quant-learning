"""Train and evaluate binary classification models for TASK5.

The script supports the two datasets bundled in this directory and produces:

- output/model_metrics_<dataset>.csv
- output/confusion_matrices_<dataset>.csv
- output/classification_report_<dataset>.csv
- output/test_predictions_<dataset>.csv
- figures/roc_curve_<dataset>.png

Examples:
    python TASK5/analyze_classification.py
    python TASK5/analyze_classification.py --dataset stock
"""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
OUTPUT_DIR = BASE_DIR / "output"
FIGURE_DIR = BASE_DIR / "figures"
CACHE_DIR = Path(tempfile.gettempdir()) / "llm_quant_task5_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR / "xdg"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class DatasetBundle:
    """Features, labels, and human-readable dataset metadata."""

    features: pd.DataFrame
    target: pd.Series
    source_path: Path
    description: str
    positive_class: str


def resolve_data_file(raw_path: str | None, dataset: str) -> Path:
    """Resolve a custom file or select the bundled default file."""

    if raw_path is None:
        filename = (
            "model_data_cancer.csv" if dataset == "cancer" else "model_data_stock.csv"
        )
        return BASE_DIR / filename

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path

    project_path = PROJECT_ROOT / path
    if project_path.exists():
        return project_path
    return Path.cwd() / path


def load_cancer_data(path: Path) -> DatasetBundle:
    """Load the breast-cancer dataset, where target 1 means benign."""

    data = pd.read_csv(path)
    if "target" not in data.columns:
        raise ValueError(f"{path.name} must contain a 'target' column.")

    target = pd.to_numeric(data.pop("target"), errors="raise").astype(int)
    features = data.apply(pd.to_numeric, errors="coerce")
    return DatasetBundle(
        features=features,
        target=target,
        source_path=path,
        description="Breast cancer diagnostic data",
        positive_class="1 = benign",
    )


def load_stock_data(path: Path) -> DatasetBundle:
    """Load stock factors and convert the Y label from bool/text to 0/1."""

    data = pd.read_csv(path)
    if "Y" not in data.columns:
        raise ValueError(f"{path.name} must contain a 'Y' column.")

    label_mapping = {
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
        "1.0": 1,
        "0.0": 0,
    }
    target = data.pop("Y").astype(str).str.strip().str.lower().map(label_mapping)
    if target.isna().any():
        invalid = sorted(data.loc[target.isna()].astype(str).head(5).index.tolist())
        raise ValueError(f"Unrecognized Y values near row indices: {invalid}")
    target = target.astype(int)

    # Date and Code identify a sample but are not continuous explanatory factors.
    features = data.drop(columns=["Date", "Code"], errors="ignore")
    features = features.apply(pd.to_numeric, errors="coerce")
    return DatasetBundle(
        features=features,
        target=target,
        source_path=path,
        description="Stock fundamental-factor classification data",
        positive_class="1 = Y is True",
    )


def load_dataset(dataset: str, path: Path) -> DatasetBundle:
    """Load and validate one of the supported binary datasets."""

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    bundle = load_cancer_data(path) if dataset == "cancer" else load_stock_data(path)
    unique_labels = sorted(bundle.target.dropna().unique().tolist())
    if unique_labels != [0, 1]:
        raise ValueError(f"Target must contain both 0 and 1; found {unique_labels}.")
    if bundle.features.empty:
        raise ValueError("No usable feature columns were found.")
    return bundle


def build_models(feature_names: list[str], random_state: int) -> dict[str, Pipeline]:
    """Create reproducible preprocessing-and-model pipelines."""

    numeric_preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                feature_names,
            )
        ],
        remainder="drop",
    )

    scaled_preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_names,
            )
        ],
        remainder="drop",
    )

    return {
        "Logistic Regression": Pipeline(
            [
                ("preprocess", scaled_preprocessor),
                (
                    "model",
                    LogisticRegression(max_iter=2_000, random_state=random_state),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("preprocess", numeric_preprocessor),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=5,
                        min_samples_leaf=5,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocess", numeric_preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate_models(
    models: dict[str, Pipeline],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, tuple]]:
    """Fit every model and collect metrics, predictions, and ROC coordinates."""

    metric_rows: list[dict] = []
    confusion_rows: list[dict] = []
    report_rows: list[dict] = []
    predictions = pd.DataFrame({"row_index": y_test.index, "actual": y_test.to_numpy()})
    roc_data: dict[str, tuple] = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        probability = model.predict_proba(x_test)[:, 1]
        false_positive_rate, true_positive_rate, thresholds = roc_curve(
            y_test, probability
        )
        auc_value = roc_auc_score(y_test, probability)
        true_negative, false_positive, false_negative, true_positive = (
            confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()
        )

        metric_rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predicted),
                "precision": precision_score(y_test, predicted, zero_division=0),
                "recall": recall_score(y_test, predicted, zero_division=0),
                "f1": f1_score(y_test, predicted, zero_division=0),
                "auc": auc_value,
            }
        )
        confusion_rows.append(
            {
                "model": name,
                "TN": int(true_negative),
                "FP": int(false_positive),
                "FN": int(false_negative),
                "TP": int(true_positive),
            }
        )

        model_report = classification_report(
            y_test,
            predicted,
            labels=[0, 1],
            target_names=["class_0", "class_1"],
            output_dict=True,
            zero_division=0,
        )
        for label, values in model_report.items():
            if isinstance(values, dict):
                report_rows.append({"model": name, "label": label, **values})

        safe_name = name.lower().replace(" ", "_")
        predictions[f"predicted_{safe_name}"] = predicted
        predictions[f"probability_1_{safe_name}"] = probability
        roc_data[name] = (false_positive_rate, true_positive_rate, thresholds, auc_value)

    metrics = pd.DataFrame(metric_rows).sort_values("auc", ascending=False)
    confusions = pd.DataFrame(confusion_rows)
    reports = pd.DataFrame(report_rows)
    return metrics, confusions, reports, predictions, roc_data


def plot_roc_curves(
    roc_data: dict[str, tuple], dataset: str, destination: Path
) -> None:
    """Draw all test-set ROC curves on one chart."""

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    colors = {
        "Logistic Regression": "#2563eb",
        "Decision Tree": "#ea580c",
        "Random Forest": "#16a34a",
    }
    for name, (false_positive_rate, true_positive_rate, _, auc_value) in roc_data.items():
        ax.plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=2.2,
            color=colors[name],
            label=f"{name} (AUC = {auc_value:.4f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b", label="Random (AUC = 0.5)")
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1.02),
        xlabel="False Positive Rate (FPR)",
        ylabel="True Positive Rate (TPR)",
        title=f"ROC Curves - {dataset.title()} Test Set",
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train binary classifiers and compare AUC/ROC results."
    )
    parser.add_argument(
        "--dataset",
        choices=["cancer", "stock"],
        default="cancer",
        help="Dataset to analyze (default: cancer).",
    )
    parser.add_argument(
        "--data-file",
        help="Optional CSV path. Defaults to the matching file in TASK5.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction reserved for the test set (default: 0.20).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for splitting and model fitting (default: 42).",
    )
    args = parser.parse_args()
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1.")
    return args


def main() -> None:
    args = parse_args()
    source_path = resolve_data_file(args.data_file, args.dataset)
    bundle = load_dataset(args.dataset, source_path)

    x_train, x_test, y_train, y_test = train_test_split(
        bundle.features,
        bundle.target,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=bundle.target,
    )
    models = build_models(bundle.features.columns.tolist(), args.random_state)
    metrics, confusions, reports, predictions, roc_data = evaluate_models(
        models, x_train, x_test, y_train, y_test
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_DIR / f"model_metrics_{args.dataset}.csv"
    confusion_path = OUTPUT_DIR / f"confusion_matrices_{args.dataset}.csv"
    report_path = OUTPUT_DIR / f"classification_report_{args.dataset}.csv"
    predictions_path = OUTPUT_DIR / f"test_predictions_{args.dataset}.csv"
    figure_path = FIGURE_DIR / f"roc_curve_{args.dataset}.png"

    metrics.to_csv(metrics_path, index=False, float_format="%.6f")
    confusions.to_csv(confusion_path, index=False)
    reports.to_csv(report_path, index=False, float_format="%.6f")
    predictions.to_csv(predictions_path, index=False, float_format="%.6f")
    plot_roc_curves(roc_data, args.dataset, figure_path)

    print(f"Dataset: {bundle.description}")
    print(f"Source: {bundle.source_path}")
    print(f"Positive class: {bundle.positive_class}")
    print(
        f"Rows: {len(bundle.features)} | Features: {bundle.features.shape[1]} | "
        f"Train: {len(x_train)} | Test: {len(x_test)}"
    )
    print("\nTest-set metrics:")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nConfusion matrices:")
    print(confusions.to_string(index=False))
    print(f"\nROC figure: {figure_path}")


if __name__ == "__main__":
    main()
