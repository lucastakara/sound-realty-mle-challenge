from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import metrics, model_selection, neighbors, pipeline, preprocessing

SALES_PATH = "data/kc_house_data.csv"
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"

SALES_COLUMN_SELECTION = [
    "price",
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "sqft_above",
    "sqft_basement",
    "zipcode",
]


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(SALES_PATH, usecols=SALES_COLUMN_SELECTION, dtype={"zipcode": str})
    demographics = pd.read_csv(DEMOGRAPHICS_PATH, dtype={"zipcode": str})

    merged = data.merge(demographics, how="left", on="zipcode").drop(columns="zipcode")
    y = merged.pop("price")
    x = merged
    return x, y


def fmt(mean: float, std: float) -> str:
    return f"{mean:.6f} ± {std:.6f}"


def main() -> None:
    x, y = load_data()

    k = 5
    kf = model_selection.KFold(n_splits=k, shuffle=True, random_state=42)

    mae_folds: list[float] = []
    rmse_folds: list[float] = []
    r2_folds: list[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(x), start=1):
        x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Base model: RobustScaler + KNN (fresh pipeline each fold)
        model = pipeline.make_pipeline(
            preprocessing.RobustScaler(),
            neighbors.KNeighborsRegressor(),
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)

        # MAE (Mean Absolute Error):
        # "On average, how many dollars off are we?" (linear penalty, robust to outliers)
        mae = metrics.mean_absolute_error(y_val, y_pred)

        # RMSE (Root Mean Squared Error):
        # Penalizes big misses more than MAE, but stays in "$" units.
        mse = metrics.mean_squared_error(y_val, y_pred)
        rmse = float(np.sqrt(mse))

        # R² (Coefficient of Determination):
        # How much better than predicting the mean price? 1=perfect, 0=mean baseline, <0=worse than mean baseline.
        r2 = metrics.r2_score(y_val, y_pred)

        mae_folds.append(mae)
        rmse_folds.append(float(rmse))
        r2_folds.append(r2)

        # Per-fold output (optional)
        print(f"Fold {fold_idx}/{k}")
        print(f"  MAE  (avg $ error):        {mae:.6f}")
        print(f"  RMSE ($, big errors hurt): {rmse:.6f}")
        print(f"  R2   (vs mean baseline):   {r2:.6f}")

    # Aggregate: mean and std across folds (sample std ddof=1)
    mae_mean, mae_std = float(np.mean(mae_folds)), float(np.std(mae_folds, ddof=1))
    rmse_mean, rmse_std = float(np.mean(rmse_folds)), float(np.std(rmse_folds, ddof=1))
    r2_mean, r2_std = float(np.mean(r2_folds)), float(np.std(r2_folds, ddof=1))

    print("\n" + "-" * 70)
    print(f"{k}-Fold Cross-Validation Summary (mean ± std over {k} folds)")
    print(f"MAE   (avg $ error):        {fmt(mae_mean, mae_std)}")
    print(f"RMSE  ($, big errors hurt): {fmt(rmse_mean, rmse_std)}")
    print(f"R2    (vs mean baseline):   {fmt(r2_mean, r2_std)}")


if __name__ == "__main__":
    main()