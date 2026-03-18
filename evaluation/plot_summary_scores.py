"""Plot summary comparison of RINet vs refineR from summary_scores.csv."""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


def main():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'summary_scores.csv')

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run evaluate_all.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    required_metrics = {'Accuracy', 'Average Norm. Error', 'Mean |Z-dev|'}
    present_metrics = set(df['Metric'].unique())
    missing = required_metrics - present_metrics
    if missing:
        print(f"Error: summary_scores.csv is missing metrics: {missing}")
        print("Run evaluate_all.py to regenerate it.")
        sys.exit(1)

    grouped = df.groupby(['Metric', 'Dataset', 'Method']).mean().reset_index()
    metrics = grouped['Metric'].unique()

    fig, axes = plt.subplots(len(metrics), 1, figsize=(7, 6), dpi=350)

    color_map = {
        'RINet': 'lightgreen',
        'CNN': 'lightgreen',
        'refineR': 'skyblue',
    }

    for i, metric in enumerate(metrics):
        ax = axes[i]
        metric_data = grouped[grouped['Metric'] == metric]
        metric_data_pivot = metric_data.pivot(index='Dataset', columns='Method', values='Score')
        metric_data_pivot = metric_data_pivot[metric_data_pivot.columns[::-1]]
        dataset_order = ['Simulated', 'RIBench']
        metric_data_pivot = metric_data_pivot.reindex([d for d in dataset_order if d in metric_data_pivot.index])

        metric_data_pivot.plot(
            kind='barh', ax=ax,
            color=[color_map.get(method, 'gray') for method in metric_data_pivot.columns]
        )

        ax.set_title(f'{metric}')

        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            legend = ax.legend(handles[::-1], labels[::-1], title='Method',
                               bbox_to_anchor=(1.25, 0.8), edgecolor=None)
            legend.get_frame().set_alpha(None)
            legend.get_frame().set_facecolor((0, 0, 0, 0))
        else:
            ax.legend().remove()

        if metric == 'Accuracy':
            ax.set_xlim([0, 1])
        ax.set_ylabel('')
        ax.grid(alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'summary_scores.png'),
                bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
