import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_combined_scores(df: pd.DataFrame, title="Combined Score vs. Number of Clusters (k)") -> None:
    sns.set(style="whitegrid")

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="k", y="Combined_Score", hue="Model", marker="o")
    plt.title(title)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Combined Score")
    plt.xticks(sorted(df["k"].unique()))
    plt.ylim(0, 1.05)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.show()
