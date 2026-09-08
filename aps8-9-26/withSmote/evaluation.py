from sklearn.metrics import (accuracy_score,precision_score,f1_score,confusion_matrix,classification_report)


def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    print("\nModel Evaluation")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["No Churn", "Churn"],
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(cm)

    return accuracy, precision, f1, cm