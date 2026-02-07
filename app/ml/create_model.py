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

    mae_model_folds: list[float] = []
    r2_model_folds: list[float] = []

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

        mae_model = metrics.mean_absolute_error(y_val, y_pred)
        r2_model = metrics.r2_score(y_val, y_pred)

        mae_model_folds.append(mae_model)
        r2_model_folds.append(r2_model)

        # Per-fold output (optional)
        print(f"Fold {fold_idx}/{k}")
        print(f"  MAE model: {mae_model:.6f}")
        print(f"  R2  model: {r2_model:.6f}")

    # Aggregate: mean and std across folds (sample std ddof=1)
    mae_model_mean = float(np.mean(mae_model_folds))
    mae_model_std = float(np.std(mae_model_folds, ddof=1))

    r2_model_mean = float(np.mean(r2_model_folds))
    r2_model_std = float(np.std(r2_model_folds, ddof=1))

    print("\n" + "-" * 70)
    print(f"{k}-Fold Cross-Validation Summary (mean ± std over {k} folds)")
    print(f"Mean Absolute Error (base model): {fmt(mae_model_mean, mae_model_std)}")
    print(f"R-squared Score (base model):     {fmt(r2_model_mean, r2_model_std)}")


if __name__ == "__main__":
    main()