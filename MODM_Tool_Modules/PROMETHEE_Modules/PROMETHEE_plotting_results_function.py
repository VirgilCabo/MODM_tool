import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def plot_bar_chart_promethee(net_flows, weights, user_input, directory):
    # Sort net_flows for better visualization
    sorted_net_flows = net_flows.sort_values(ascending=False)

    # Plot
    df = sorted_net_flows.reset_index()
    df.columns = ['Alternatives', 'Net Outranking Flow']
    order = df['Alternatives'].tolist()
    norm = plt.Normalize(df['Net Outranking Flow'].min(), df['Net Outranking Flow'].max())
    colors = plt.cm.cividis(norm(df['Net Outranking Flow']))
    barplot = sns.barplot(
        x='Alternatives',
        y='Net Outranking Flow',
        data=df,
        palette=list(colors),
        legend=False,
        order=order)
    sm = ScalarMappable(cmap=plt.cm.cividis, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=barplot, orientation='vertical')
    plt.title('Net Outranking Flows of Alternatives')
    plt.ylabel('Net Outranking Flow')
    plt.xlabel('Alternatives')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    """ weights_text = "Criteria weights (1-10):\n\n" + "\n".join(
        [f"{criterion}: {int(weight)}" for criterion, weight in weights.items()])
    plt.text(
        0.75,
        0.95,
        weights_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor='aliceblue',
            edgecolor='black')) """
    if user_input == 'yes':
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                directory,
                'bar_chart.png'),
            dpi=500,
            bbox_inches='tight')
    plt.show()


def show_ranking(ranked_alternatives):
    print(ranked_alternatives)
    return


def results_visualization_promethee(net_flows, weights, user_input, directory, ranked_alternatives):
    plot_bar_chart_promethee(net_flows, weights, user_input, directory)
    show_ranking(ranked_alternatives)
    return
