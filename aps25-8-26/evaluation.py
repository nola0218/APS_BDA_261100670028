import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def get_probabilities(model, X_test):
    probabilities = model.predict_proba(X_test)

    p_malignant = probabilities[:, 0]
    p_benign = probabilities[:, 1]

    return p_malignant, p_benign


def create_results(y_test, p_malignant, p_benign):
    results = pd.DataFrame({
        "Actual_class": y_test.values,
        "P_malignant": p_malignant,
        "P_benign": p_benign
    })

    results["Actual_label"] = results["Actual_class"].map({
        0: "Malignant",
        1: "Benign"
    })

    return results


def apply_threshold(results, threshold=0.50):
    results = results.copy()

    results["Predicted_class"] = (
        results["P_benign"] >= threshold
    ).astype(int)

    results["Predicted_label"] = results[
        "Predicted_class"
    ].map({
        0: "Malignant",
        1: "Benign"
    })

    return results


def compare_thresholds(results, thresholds):
    for threshold in thresholds:

        predictions = (
            results["P_benign"] >= threshold
        ).astype(int)

        print(
            f"Threshold = {threshold:.2f} | "
            f"Predicted benign cases = {predictions.sum()}"
        )


def calculate_confusion_matrix(
    y_test,
    p_benign,
    threshold
):
    predictions = (
        p_benign >= threshold
    ).astype(int)

    return confusion_matrix(
        y_test,
        predictions
    )


def compare_confusion_matrices(
    y_test,
    p_benign,
    thresholds
):
    for threshold in thresholds:

        cm = calculate_confusion_matrix(
            y_test,
            p_benign,
            threshold
        )

        print(f"\nThreshold = {threshold:.2f}")
        print(cm)


def calculate_threshold_metrics(
    y_test,
    p_benign,
    thresholds
):
    metrics = []

    for threshold in thresholds:

        predictions = (
            p_benign >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions
        ).ravel()

        metrics.append({
            "Threshold": threshold,
            "TN": tn,
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "Accuracy": accuracy_score(
                y_test,
                predictions
            ),
            "Precision": precision_score(
                y_test,
                predictions,
                zero_division=0
            ),
            "Recall": recall_score(
                y_test,
                predictions,
                zero_division=0
            ),
            "F1 Score": f1_score(
                y_test,
                predictions,
                zero_division=0
            )
        })

    return pd.DataFrame(metrics)
