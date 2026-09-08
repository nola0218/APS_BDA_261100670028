import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def load_data(file_path):
    df = pd.read_csv(file_path)

    print("Shape:", df.shape)

    print("\nNull values before preprocessing:")
    print(df.isnull().sum())


    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

    print("\nNull values after converting TotalCharges:")
    print(df.isnull().sum())

    df = df.dropna(subset=["TotalCharges"])
    print("\nNull values after dropping TotalCharges null values:")
    print(df.isnull().sum())

    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({
        "Yes": 1,
        "No": 0
    })

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    print("\nTrain Test split:")
    print("Training data:", X_train.shape)
    print("Testing data :", X_test.shape)

    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    return X_train, X_test, y_train, y_test, preprocessor, y