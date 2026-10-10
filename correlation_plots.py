"""Графики для части 2: scatter и heatmap (запуск: python correlation_plots.py)."""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FEATURES = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]


def main() -> None:
    # Загружаем данные так же, как в основной функции
    df = pd.read_csv("brainsize.txt", sep="\t", na_values="NA")

    # --- Scatter-графики: каждый признак против MRI_Count, цвет это пол ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i in range(len(FEATURES)):
        feature = FEATURES[i]
        sns.scatterplot(data=df, x=feature, y="MRI_Count", hue="Gender", ax=axes[i])
        axes[i].set_title(feature + " и MRI_Count")
    axes[5].axis("off")                    # шестой график не нужен
    plt.tight_layout()
    plt.savefig("scatter.png")

    # --- Heatmap корреляций отдельно для мужчин и женщин ---
    columns = FEATURES + ["MRI_Count"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, gender in zip(axes, ["Male", "Female"]):
        group = df[df["Gender"] == gender]
        correlation_matrix = group[columns].corr()     # таблица всех корреляций
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f",
                    cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
        ax.set_title("Корреляции: " + gender)
    plt.tight_layout()
    plt.savefig("heatmap.png")

    plt.show()


if __name__ == "__main__":
    main()