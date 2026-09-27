"""Standardize features. Scaler is fitted on train only (no test leakage)."""
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA = Path("data/processed_data")


def main() -> None:
    X_train = pd.read_csv(DATA / "X_train.csv")
    X_test = pd.read_csv(DATA / "X_test.csv")

    scaler = StandardScaler().fit(X_train)
    for name, X in {"X_train_scaled": X_train, "X_test_scaled": X_test}.items():
        pd.DataFrame(scaler.transform(X), columns=X.columns).to_csv(
            DATA / f"{name}.csv", index=False
        )
    print("scaled:", list(X_train.columns))


if __name__ == "__main__":
    main()
