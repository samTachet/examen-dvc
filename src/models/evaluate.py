"""Evaluate the trained model on the test set; write predictions and scores."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA = Path("data/processed_data")


def main() -> None:
    model = joblib.load("models/gbr_model.pkl")
    X = pd.read_csv(DATA / "X_test_scaled.csv")
    y = pd.read_csv(DATA / "y_test.csv").squeeze("columns")

    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    scores = {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mean_absolute_error(y, y_pred),
        "r2": r2_score(y, y_pred),
    }

    pd.DataFrame({"y_true": y, "y_pred": y_pred}).to_csv(
        "data/prediction.csv", index=False
    )
    Path("metrics").mkdir(exist_ok=True)
    with open("metrics/scores.json", "w") as f:
        json.dump({k: round(float(v), 4) for k, v in scores.items()}, f, indent=2)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
