"""Command-line entry point for the Boston house price workflow."""

import warnings
from pathlib import Path

from boston_house_price_prediction import (
    evaluate_model,
    fit_models,
    load_dataset,
    summarize_coefficients,
)
from plotting import residual_analysis, run_bivariate_analysis, run_univariate_analysis


def main() -> None:
    """Execute the full analysis workflow end-to-end."""
    warnings.filterwarnings("ignore")
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "dataset" / "Boston.csv"
    plot_dir = base_dir / "result"
    df = load_dataset(csv_path)

    run_univariate_analysis(df, plot_dir)
    run_bivariate_analysis(df, plot_dir)

    model, X_train, X_test, y_train, y_test = fit_models(df)
    residual_analysis(model, X_train, y_train, plot_dir)
    evaluate_model(model, X_train, X_test, y_train, y_test)
    summarize_coefficients(model)


if __name__ == "__main__":
    main()
