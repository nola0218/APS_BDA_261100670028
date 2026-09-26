import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_data():
    data = load_breast_cancer()

    X = pd.DataFrame(
        data.data,
        columns=data.feature_names
    )

    y = pd.Series(
        data.target,
        name="target"
    )

    return X, y, data


def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
