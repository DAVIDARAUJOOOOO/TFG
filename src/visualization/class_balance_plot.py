import matplotlib.pyplot as plt


def plot_class_balance(y_train, y_res):
    original_counts = y_train.value_counts().sort_index()
    resampled_counts = y_res.value_counts().sort_index()

    labels = ['No dislèxia', 'Dislèxia']

    before = [original_counts.get(0, 0), original_counts.get(1, 0)]
    after = [resampled_counts.get(0, 0), resampled_counts.get(1, 0)]

    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 5))

    bars1 = plt.bar([i - width/2 for i in x], before, width, label='Original')
    bars2 = plt.bar([i + width/2 for i in x], after, width, label='SMOTENC')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2,
                height,
                f'{int(height)}',
                ha='center',
                va='bottom'
            )

    plt.xticks(x, labels)
    plt.ylabel('Nombre de mostres')
    plt.title('Distribució abans i després de SMOTENC')
    plt.legend()
    plt.tight_layout()
    plt.show()