from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def create_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    )


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model
