import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def correlation_matrix(df, features_corr):
    corr_matrix = df[features_corr].apply(pd.to_numeric).corr()

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        linewidths=.5,
        cbar_kws={"shrink": .8}
    )

    plt.title("Matriu de Correlació", fontsize=16, pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.show()
