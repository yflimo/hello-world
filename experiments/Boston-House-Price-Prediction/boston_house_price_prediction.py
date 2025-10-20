
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load the Boston housing dataset and print basic diagnostics."""
    df = pd.read_csv(csv_path)
    print("Dataset preview:")
    print(df.head())
    print("\nDataset info:")
    df.info()
    print("\nDescriptive statistics:")
    print(df.describe().T)
    return df


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into training and test sets with the features used in the notebook."""
    y = df["MEDV_log"]
    X = df.drop(columns={"MEDV", "MEDV_log"})
    X = sm.add_constant(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
    return X_train, X_test, y_train, y_test


def check_vif(features: pd.DataFrame) -> pd.DataFrame:
    """Compute the variance inflation factor for each feature."""
    vif = pd.DataFrame({"feature": features.columns})
    vif["VIF"] = [variance_inflation_factor(features.values, i) for i in range(len(features.columns))]
    return vif


def fit_models(df: pd.DataFrame) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Fit the two OLS models constructed in the notebook and return the second model and split data."""
    X_train, X_test, y_train, y_test = prepare_data(df)
    print("\nInitial VIF check:")
    print(check_vif(X_train))

    print("\nVIF after dropping TAX from X_train:")
    print(check_vif(X_train.drop(columns="TAX")))

    model1 = sm.OLS(y_train, X_train.drop(columns="TAX")).fit()
    print("\nModel 1 summary (TAX dropped from training set):")
    print(model1.summary())

    y = df["MEDV_log"]
    X = df.drop(columns=["ZN", "AGE", "INDUS"])
    X = sm.add_constant(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
    model2 = sm.OLS(y_train, X_train).fit()
    print("\nModel 2 summary (ZN, AGE, INDUS dropped from dataset):")
    print(model2.summary())
    return model2, X_train, X_test, y_train, y_test


def evaluate_model(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Compute regression performance metrics and run cross-validation."""

    def rmse(predictions: pd.Series, targets: pd.Series) -> float:
        return np.sqrt(((targets - predictions) ** 2).mean())

    def mape(predictions: pd.Series, targets: pd.Series) -> float:
        return np.mean(np.abs((targets - predictions)) / targets) * 100

    def mae(predictions: pd.Series, targets: pd.Series) -> float:
        return np.mean(np.abs((targets - predictions)))

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    performance = pd.DataFrame(
        {
            "Data": ["Train", "Test"],
            "RMSE": [rmse(y_pred_train, y_train), rmse(y_pred_test, y_test)],
            "MAE": [mae(y_pred_train, y_train), mae(y_pred_test, y_test)],
            "MAPE": [mape(y_pred_train, y_train), mape(y_pred_test, y_test)],
            "r2": [r2_score(y_train, y_pred_train), r2_score(y_test, y_pred_test)],
        }
    )
    print("\nModel performance:")
    print(performance)

    linear_regression = LinearRegression()
    cv_score_r2 = cross_val_score(linear_regression, X_train, y_train, cv=10)
    cv_score_mse = cross_val_score(
        linear_regression, X_train, y_train, cv=10, scoring="neg_mean_squared_error"
    )
    print(
        "\nCross-validation results:\n"
        f"R-squared: {cv_score_r2.mean():.3f} (+/- {cv_score_r2.std() * 2:.3f})\n"
        f"Mean Squared Error: {-cv_score_mse.mean():.3f} (+/- {cv_score_mse.std() * 2:.3f})"
    )


def summarize_coefficients(model: sm.regression.linear_model.RegressionResultsWrapper) -> None:
    """Print coefficient table and regression equation."""
    coef = model.params
    coef_table = pd.DataFrame({"Feature": coef.index, "Coefficient": coef.values})
    print("\nModel coefficients:")
    print(coef_table)

    pieces = [f"({coef[idx]:.4f}) * {idx}" for idx in coef.index]
    equation = "log(Price) = " + " + ".join(pieces)
    print("\nRegression equation:")
    print(equation)
