import matplotlib.pyplot as plt
import seaborn as sns

def plot_balance_chunks(df):
    counts = df['label'].value_counts()
    labels = ['Sense Dislèxia', 'Amb Dislèxia']
    colors = sns.color_palette('pastel')[0:2]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    wedges, texts, autotexts = ax.pie(
        counts,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        pctdistance=0.85,
        explode=[0.05, 0],
        textprops={'fontsize': 12, 'fontweight': 'bold'}
    )

    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)

    total = len(df)

    plt.text(
        0, 0,
        f'Total\n{total}\nfragments',
        ha='center',
        va='center',
        fontsize=14,
        fontweight='bold'
    )

    plt.title(
        'Balanç de Fragments (2s): Diagnòstic de Dislèxia',
        fontsize=15,
        pad=20,
        fontweight='bold'
    )

    plt.axis('equal')
    plt.tight_layout()
    plt.show()