"""Split raw data into train/test sets (target = last column)."""
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

RAW = Path("data/raw_data/raw.csv")
OUT = Path("data/processed_data")


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))["split"]
    df = pd.read_csv(RAW)
    # 'date' is an identifier/timestamp, not a feature: dropped before modelling.
    df = df.drop(columns=["date"], errors="ignore")

    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], random_state=params["random_state"]
    )

    OUT.mkdir(parents=True, exist_ok=True)
    for name, obj in {"X_train": X_train, "X_test": X_test,
                      "y_train": y_train, "y_test": y_test}.items():
        obj.to_csv(OUT / f"{name}.csv", index=False)
    print(f"train={len(X_train)} test={len(X_test)} features={X.shape[1]}")


if __name__ == "__main__":
    main()
