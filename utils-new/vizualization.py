import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def correlation_matrix(df: pd.DataFrame, figsize=(12, 10), annot=False):
    numeric_df = df.select_dtypes(include='number')

    if numeric_df.empty:
        print("⚠️ No numeric columns found for correlation.")
        return

    corr = numeric_df.corr()
    plt.figure(figsize=figsize)
    sns.heatmap(corr, cmap='coolwarm', annot=annot, fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.show()



def plot_feature_distributions(df, columns=None, log_y=True, bins=100, figsize=(8, 4)):
    if columns is None:
        columns = df.select_dtypes(include='number').columns.tolist()

    for col in columns:
        plt.figure(figsize=figsize)
        ax = sns.histplot(
            data=df,
            x=col,
            bins=bins,
            log_scale=(False, log_y),
            kde=False,
            color='steelblue',
            edgecolor='black',
            linewidth=0.7
        )
        ax.set_facecolor('#f9f9f9')
        plt.title(f'Distribution: {col}')
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()