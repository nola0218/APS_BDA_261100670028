from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def create_model(preprocessor):
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])
    return model


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred

#for smote
def predict_probability(model, X_test):
    y_probability = model.predict_proba(X_test)[:, 1]
    return y_probability