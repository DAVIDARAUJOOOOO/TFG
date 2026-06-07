import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, auc,
    roc_curve, confusion_matrix, ConfusionMatrixDisplay
)


def plot_mlp_results(y_test, y_prob, y_pred):

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # PR curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)

    axes[0].plot(recall, precision)
    axes[0].set_title(f"PR AUC = {pr_auc:.3f}")

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    axes[1].plot(fpr, tpr)
    axes[1].set_title(f"ROC AUC = {roc_auc:.3f}")

    # CM
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot(ax=axes[2], colorbar=False)

    plt.tight_layout()
    plt.show()