import matplotlib.pyplot as plt


def plot_class_distribution(y):
    class_counts = y.value_counts().sort_index()

    plt.figure(figsize=(6, 4))

    plt.bar(
        ["Malignant", "Benign"],
        [
            class_counts[0],
            class_counts[1]
        ],
        color=["red", "royalblue"]
    )

    plt.ylabel("Number of observations")
    plt.title("Class Distribution")

    plt.tight_layout()
    plt.show()
