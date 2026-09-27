"""Train the final model with the best params from the grid search."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

DATA = Path("data/processed_data")
MODELS = Path("models")


def main() -> None:
    best_params = joblib.load(MODELS / "best_params.pkl")

    X = pd.read_csv(DATA / "X_train_scaled.csv")
    y = pd.read_csv(DATA / "y_train.csv").squeeze("columns")

    model = GradientBoostingRegressor(**best_params).fit(X, y)
    joblib.dump(model, MODELS / "gbr_model.pkl")
    print("trained with:", best_params)


if __name__ == "__main__":
    main()
