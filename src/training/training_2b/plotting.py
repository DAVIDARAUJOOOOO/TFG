from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

def plot_audio_results(y_test_audio, y_probs_audio):
    # ROC
    fpr, tpr, thresholds = roc_curve(y_test_audio, y_probs_audio)
    roc_auc = auc(fpr, tpr)

    # Datos scatter
    y_jitter = np.random.normal(0, 0.02, size=len(y_probs_audio))

    probs = y_probs_audio.values
    labels = y_test_audio.values

    # Figura con 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # =========================
    # ROC Curve
    # =========================
    axes[0].plot(fpr, tpr, color='orange', label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="blue")

    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve per Audio")
    axes[0].legend()

    # =========================
    # Scatter probabilidades
    # =========================

    # Clase 0
    axes[1].scatter(
        probs[labels == 0],
        y_jitter[labels == 0],
        alpha=0.7,
        label="Clase 0"
    )

    # Clase 1
    axes[1].scatter(
        probs[labels == 1],
        y_jitter[labels == 1],
        alpha=0.7,
        label="Clase 1"
    )

    axes[1].axvline(0.39, linestyle="--")

    axes[1].set_xlabel("Probabilitat mitjana de la classe positiva")
    axes[1].set_yticks([])

    axes[1].set_title("Probabilitats per audio")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return fpr, tpr, thresholds, roc_auc