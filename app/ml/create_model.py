from __future__ import annotations

import itertools
import json
import pathlib
import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics, model_selection
from catboost import CatBoostRegressor, Pool

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
    "waterfront",
    "view",
    "condition",
    "grade",
    "yr_built",
    "yr_renovated",
]

# mark these as categorical-ish for CatBoost (works even if they are ints)
CATEGORICAL_FEATURES = ["waterfront", "view", "condition", "grade", "yr_built", "yr_renovated"]

OUTPUT_DIR = "model"


def load_data(
    sales_path: str, demographics_path: str, sales_column_selection: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(sales_path, usecols=sales_column_selection, dtype={"zipcode": str})
    demographics = pd.read_csv(demographics_path, dtype={"zipcode": str})

    merged = data.merge(demographics, how="left", on="zipcode").drop(columns="zipcode")

    if merged.isna().any().any():
        raise ValueError("Found missing values after demographics merge; expected none.")

    y = merged.pop("price")
    X = merged
    return X, y


def main() -> None:
    X, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)

    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    cat_feature_indices = [X.columns.get_loc(c) for c in cat_cols]

    k = 5
    cv = model_selection.KFold(n_splits=k, shuffle=True, random_state=42)

    params_grid = {
        "learning_rate": [0.03, 0.1, 0.3],
        "depth": [4, 6, 8],
        "iterations": [50, 100, 200],
    }

    base_params = dict(
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=42,
        allow_writing_files=False,
        verbose=False,  # keep search quiet
    )

    def eval_5fold(params: dict) -> tuple[float, float, float, float, float, float]:
        maes, rmses, r2s = [], [], []
        for tr_idx, va_idx in cv.split(X):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            train_pool = Pool(X_tr, y_tr, cat_features=cat_feature_indices)
            val_pool = Pool(X_va, y_va, cat_features=cat_feature_indices)

            model = CatBoostRegressor(**params)
            model.fit(train_pool, eval_set=val_pool, use_best_model=False, verbose=False)

            pred = model.predict(X_va)
            maes.append(metrics.mean_absolute_error(y_va, pred))
            rmses.append(float(np.sqrt(metrics.mean_squared_error(y_va, pred))))
            r2s.append(metrics.r2_score(y_va, pred))

        return (
            float(np.mean(maes)), float(np.std(maes, ddof=1)),
            float(np.mean(rmses)), float(np.std(rmses, ddof=1)),
            float(np.mean(r2s)), float(np.std(r2s, ddof=1)),
        )

    combos = list(itertools.product(
        params_grid["learning_rate"],
        params_grid["depth"],
        params_grid["iterations"],
    ))

    print(f"Grid search: {len(combos)} configs, evaluated with {k}-fold CV\n")

    best_params = None
    best_mae = float("inf")

    for lr, depth, iters in combos:
        params = {**base_params, "learning_rate": lr, "depth": depth, "iterations": iters}
        mae_m, mae_s, rmse_m, rmse_s, r2_m, r2_s = eval_5fold(params)

        print(
            f"lr={lr:<4} depth={depth:<2} iters={iters:<3} | "
            f"MAE={mae_m:.2f} ± {mae_s:.2f} | "
            f"RMSE={rmse_m:.2f} ± {rmse_s:.2f} | "
            f"R2={r2_m:.4f} ± {r2_s:.4f}"
        )

        if mae_m < best_mae:
            best_mae = mae_m
            best_params = params

    assert best_params is not None

    print("\n" + "-" * 70)
    print("Best params (by mean CV MAE):")
    print({k: best_params[k] for k in ["learning_rate", "depth", "iterations"]})

    # Final CV summary on best params (mean ± std)
    mae_m, mae_s, rmse_m, rmse_s, r2_m, r2_s = eval_5fold(best_params)
    print("\n" + "-" * 70)
    print(f"{k}-Fold CV Summary (mean ± std)")
    print(f"MAE   (avg $ error):        {mae_m:.6f} ± {mae_s:.6f}")
    print(f"RMSE  ($, big errors hurt): {rmse_m:.6f} ± {rmse_s:.6f}")
    print(f"R2    (vs mean baseline):   {r2_m:.6f} ± {r2_s:.6f}")

    # ---- Train final model on ALL data with best params + export artifacts ----
    final_model = CatBoostRegressor(**best_params)
    final_pool = Pool(X, y, cat_features=cat_feature_indices)
    final_model.fit(final_pool, verbose=False)

    # ---- Top 5 feature importances (CLI) ----
    importances = final_model.get_feature_importance(final_pool)
    feat_names = list(X.columns)
    top_k = 5
    top_idx = np.argsort(importances)[::-1][:top_k]

    print("\n" + "-" * 70)
    print(f"Top {top_k} Feature Importances (CatBoost)")
    for rank, idx in enumerate(top_idx, start=1):
        print(f"{rank}. {feat_names[idx]}: {importances[idx]:.4f}")

    out = pathlib.Path(OUTPUT_DIR)
    out.mkdir(exist_ok=True)

    pickle.dump(final_model, open(out / "model.pkl", "wb"))
    json.dump(list(X.columns), open(out / "model_features.json", "w"))
    json.dump(cat_cols, open(out / "cat_features.json", "w"))

    print("\nSaved artifacts:")
    print(f" - {out / 'model.pkl'}")
    print(f" - {out / 'model_features.json'}")
    print(f" - {out / 'cat_features.json'}  (names used as categorical)")


if __name__ == "__main__":
    main()