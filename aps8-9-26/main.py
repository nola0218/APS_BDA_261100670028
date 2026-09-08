from data import load_data
from model import create_model, train_model, predict, predict_probability
from evaluation import evaluate_model
from visualisation import plot_class_distribution, plot_roc_curve, show_plots


def main():
    #load dataset
    file_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        y
    ) = load_data(file_path)

    #train model
    model = create_model(preprocessor)
    model = train_model(
        model,
        X_train,
        y_train
    )

    print("\nModel trained successfully")

    y_pred = predict(
        model,
        X_test
    )

    y_probability = predict_probability(
    model,
    X_test
    )

    accuracy, precision, f1, cm = evaluate_model(
        y_test,
        y_pred
    )

    #display plots
    plot_class_distribution(y)
    plot_roc_curve(
    y_test,
    y_probability
    )
    show_plots()
    

if __name__ == "__main__":
    main()