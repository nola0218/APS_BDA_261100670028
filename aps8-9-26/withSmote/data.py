import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

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

    numerical_columns = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_columns = X_train.select_dtypes(
        include=["object"]
    ).columns

    # Preprocessing
    numerical_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    # Fit preprocessing ONLY on training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # Transform test data
    X_test_processed = preprocessor.transform(X_test)

    # Apply SMOTE only to training data
    smote = SMOTE(
        random_state=42
    )

    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_processed,
        y_train
    )

    print("\nClass distribution before SMOTE:")
    print(y_train.value_counts())

    print("\nClass distribution after SMOTE:")
    print(y_train_resampled.value_counts())

    return (
        X_train_resampled,
        X_test_processed,
        y_train_resampled,
        y_test,
        y
    )