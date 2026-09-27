"""GridSearchCV over GradientBoostingRegressor; saves best params."""
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

DATA = Path("data/processed_data")
OUT = Path("models/best_params.pkl")


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))
    gs_cfg, seed = params["grid_search"], params["split"]["random_state"]

    X = pd.read_csv(DATA / "X_train_scaled.csv")
    y = pd.read_csv(DATA / "y_train.csv").squeeze("columns")

    gs = GridSearchCV(
        GradientBoostingRegressor(random_state=seed),
        param_grid=gs_cfg["param_grid"],
        cv=gs_cfg["cv"],
        scoring=gs_cfg["scoring"],
        n_jobs=-1,
    ).fit(X, y)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # random_state saved with the params: training depends on this file only.
    joblib.dump({**gs.best_params_, "random_state": seed}, OUT)
    print("best params:", gs.best_params_, "| cv score:", round(gs.best_score_, 4))


if __name__ == "__main__":
    main()
