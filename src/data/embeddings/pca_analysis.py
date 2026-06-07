import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def run_pca_analysis(embeddings, components=[10, 15, 20, 30, 50]):
    variances = []

    for c in components:
        pca = PCA(n_components=c)
        embeddings_pca = pca.fit_transform(embeddings)

        var = np.sum(pca.explained_variance_ratio_)
        variances.append(var)

        print(f"Components: {c}, Variance: {var}")

    plt.plot(components, variances)
    plt.xlabel("Components")
    plt.ylabel("Variancia explicada")
    plt.title("Variança explicada segons PCA")
    plt.show()

    return variances