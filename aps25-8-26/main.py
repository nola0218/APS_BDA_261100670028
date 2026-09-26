from data import load_data, split_data

from model import create_model, train_model

from evaluation import (
    get_probabilities,
    create_results,
    apply_threshold,
    compare_thresholds,
    compare_confusion_matrices,
    calculate_threshold_metrics
)

from visualization import plot_class_distribution


def main():

    X, y, data = load_data()

    print(y.value_counts())
    print("Feature matrix shape:", X.shape)
    print("Target shape:", y.shape)
    print("Class names:", data.target_names)

    class_counts = y.value_counts().sort_index()

    print("\nClass distribution:")

    class_distribution = {
        "Malignant": class_counts[0],
        "Benign": class_counts[1]
    }

    for class_name, count in class_distribution.items():

        probability = count / len(y)

        print(
            f"{class_name}: "
            f"{count} "
            f"({probability:.2%})"
        )

    plot_class_distribution(y)

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    print("\nTraining size:", len(y_train))
    print("Testing size:", len(y_test))

    print("\nTraining proportions:")
    print(
        y_train
        .value_counts(normalize=True)
        .sort_index()
    )

    print("\nTesting proportions:")
    print(
        y_test
        .value_counts(normalize=True)
        .sort_index()
    )

    model = create_model()

    train_model(
        model,
        X_train,
        y_train
    )

    print("\nModel trained successfully.")

    p_malignant, p_benign = get_probabilities(
        model,
        X_test
    )

    print("\nFirst 5 probability predictions:")

    print(
        list(
            zip(
                p_malignant[:5],
                p_benign[:5]
            )
        )
    )

    results = create_results(
        y_test,
        p_malignant,
        p_benign
    )

    print("\nPrediction probabilities:")

    print(
        results[
            [
                "Actual_label",
                "P_malignant",
                "P_benign"
            ]
        ].head(10)
    )

    threshold = 0.50

    results = apply_threshold(
        results,
        threshold
    )

    print(
        f"\nPredictions using threshold = {threshold:.2f}:"
    )

    print(
        results[
            [
                "Actual_label",
                "P_benign",
                "Predicted_label"
            ]
        ].head(10)
    )

    thresholds = [
        0.10,
        0.30,
        0.50,
        0.70,
        0.90
    ]

    print("\nThreshold comparison:")

    compare_thresholds(
        results,
        thresholds
    )

    print("\nConfusion matrices:")

    compare_confusion_matrices(
        y_test,
        p_benign,
        thresholds
    )

    metrics_table = calculate_threshold_metrics(
        y_test,
        p_benign,
        thresholds
    )

    print("\nThreshold Metrics:")

    print(
        metrics_table.round(3).to_string(index=False)
    )


if __name__ == "__main__":
    main()
