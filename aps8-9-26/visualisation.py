import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_class_distribution(y):
    class_counts = y.value_counts()
    labels = ["No Churn", "Churn"]
    plt.figure(figsize=(6, 5))
    plt.bar(
        labels,
        [
            class_counts.get(0, 0),
            class_counts.get(1, 0)
        ]
    )

    plt.xlabel("Class")
    plt.ylabel("Number of Customers")
    plt.title("Customer Churn Class Distribution")


def plot_roc_curve(y_test, y_probability):
    fpr, tpr, thresholds = roc_curve(
        y_test,
        y_probability
    )
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(
        fpr,
        tpr,
        label=f"ROC Curve (AUC = {roc_auc:.3f})"
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-AUC Curve")
    plt.legend(loc="lower right")


def show_plots():
    plt.show()