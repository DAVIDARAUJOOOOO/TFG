import matplotlib.pyplot as plt
import seaborn as sns



def set_plot_style():
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams['figure.dpi'] = 150


def plot_language_violin(df_plot):
    plt.figure(figsize=(14, 6))
    sns.set_theme(style="whitegrid")

    course_order = sorted(df_plot["curs"].unique())
    order = df_plot["lm"].value_counts().index
    sns.violinplot(x='lm', y='duracio', data=df_plot, order=order, inner=None, color='.9')
    sns.stripplot(x='lm', y='duracio', data=df_plot, hue='curs', hue_order=course_order, order=order, size=6, jitter=True, palette='magma')

    plt.title('Distribució de temps de lectura per Llengua materna i Curs', fontsize=14)
    plt.xlabel('Llengua materna (L1)', fontsize=12)
    plt.ylabel('Duració (segons)', fontsize=12)

    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Curs escolar', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()


def plot_balance_global(df):
    counts = df['dislexia'].value_counts()
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
    plt.text(0, 0, f'Total\n{total}\naudios', ha='center', va='center', fontsize=14, fontweight='bold')

    plt.title('Balanç Global del Dataset: Diagnòstic de Dislèxia', fontsize=15, pad=20, fontweight='bold')

    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def plot_distribution_dislexia(df):
    plt.figure(figsize=(10, 6))
    course_order = sorted(df['curs'].unique())
    ax = sns.countplot(data=df, x='curs', hue='dislexia', order=course_order, edgecolor='0.2')

    plt.title('Distribució de la Mostra per Curs i Diagnòstic de Dislèxia', fontsize=14, pad=20, fontweight='bold')
    plt.xlabel('Curs Escolar', fontsize=12)
    plt.ylabel("Nombre d'àudios (Total)", fontsize=12)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ['Sense Dislèxia', 'Amb Dislèxia'], title='Diagnòstic', loc='upper right')

    total_audios = len(df)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f'{int(height)}\n({(height / total_audios * 100):.1f}%)',
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='center',
                xytext=(0, 12),
                textcoords='offset points',
                fontsize=9, fontweight='bold'
            )

    sns.despine()
    plt.tight_layout()
    plt.show()
