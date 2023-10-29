import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure as fig
import math
import re
import os
import datetime
import seaborn as sns


def min_max_scaling_for_spider_chart(
        matrix,
        beneficial_criteria,
        normalized_weights):
    scaled_matrix = matrix.copy()

    for criterion in matrix.columns:
        min_val = matrix[criterion].min()
        max_val = matrix[criterion].max()

        if criterion in beneficial_criteria:
            # Standard Min-Max scaling for beneficial criteria
            scaled_matrix[criterion] = 1 + \
                ((matrix[criterion] - min_val) / (max_val - min_val))
        else:
            # Adjusted Min-Max scaling for non-beneficial criteria
            scaled_matrix[criterion] = 2 - \
                (matrix[criterion] - min_val) / (max_val - min_val)
    weighted_scaled_matrix = scaled_matrix.multiply(normalized_weights, axis=1)
    return scaled_matrix, weighted_scaled_matrix


def clean_criteria_names(criteria):
    cleaned = [re.sub(r'\(.*\)', '', criterion).strip()
               for criterion in criteria]
    return cleaned


def plot_bar_chart(scores, weights, user_input, directory):
    # Sort scores for better visualization
    sorted_scores = scores.sort_values(ascending=False)

    # Plot
    df = sorted_scores.reset_index()
    df.columns = ['Alternatives', 'Performance Score']
    colors = sns.color_palette("viridis", len(scores))
    sns.barplot(
        x='Alternatives',
        y='Performance Score',
        data=df,
        palette=colors,
        hue='Alternatives')
    plt.title('Performance Scores of Alternatives')
    plt.ylabel('Performance Score')
    plt.xlabel('Alternatives')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    weights_text = "Criteria weights (1-10):\n\n" + "\n".join(
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
            edgecolor='black'))
    if user_input == 'yes':
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                directory,
                'bar_chart.png'),
            dpi=500,
            bbox_inches='tight')
    plt.show()


def plot_spider_chart(
        matrix,
        weights,
        title,
        decision_matrix,
        user_input,
        directory):
    cleaned_criteria_names = clean_criteria_names(decision_matrix.columns)
    # Number of criteria
    num_vars = len(matrix.columns)

    max_val = matrix.max().max()

    # Split the circle into even parts and save the angles
    angles = [n / float(num_vars) * 2 * math.pi for n in range(num_vars)]
    angles += angles[:1]

    # Set figure & subplot axis
    plt.figure(figsize=(5, 5))
    ax = plt.subplot(111, polar=True)

    # Draw one alternative per loop
    for idx, row in matrix.iterrows():
        values = row.values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=idx)
        ax.fill(angles, values, alpha=0.25)

    # Add a title
    plt.title(title, size=20, y=1.1)

    # Set the first axis on top
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and criteria
    plt.xticks(angles[:-1], cleaned_criteria_names)

    # Draw y-labels
    ax.set_rlabel_position(0)
    if title == 'Spider Chart of Alternatives without weighting':
        y_ticks = [0.4, 0.8, 1.2, 1.6]
        plt.yticks(y_ticks,
                   [f"{tick:.2f}" for tick in y_ticks],
                   color="grey",
                   size=12)
        plt.ylim(0, 2)
    y_ticks = [max_val * 0.2, max_val * 0.4, max_val * 0.6, max_val * 0.8]
    plt.yticks(y_ticks,
               [f"{tick:.2f}" for tick in y_ticks],
               color="grey",
               size=12)
    plt.ylim(0, max_val)
    plt.legend(loc='upper left', bbox_to_anchor=(-0.35, 1))
    if title == 'Spider Chart of Alternatives with weighting':
        # Create a string with the criteria and their weights
        weights_text = "Criteria weights (1-10):\n\n" + "\n".join(
            [f"{criterion}: {int(weight)}" for criterion, weight in weights.items()])
        # Add the text box to the plot
        plt.figtext(
            1,
            1,
            weights_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor='aliceblue',
                edgecolor='black'))

    if user_input == 'yes':
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                directory,
                title + '.png'),
            dpi=500,
            bbox_inches='tight')
    plt.show()


def results_visualization(
        ranked_alternatives,
        weighted_normalized_matrix,
        beneficial_criteria,
        weights,
        normalized_weights,
        S,
        user_input,
        directory,
        decision_matrix):
    print("\nRanked Alternatives:")
    print(ranked_alternatives)
    scaled_matrix, weighted_scaled_matrix = min_max_scaling_for_spider_chart(
        weighted_normalized_matrix, beneficial_criteria, normalized_weights)
    plot_bar_chart(S, weights, user_input, directory)
    plot_spider_chart(
        scaled_matrix,
        weights,
        "Spider Chart of Alternatives without weighting",
        decision_matrix,
        user_input,
        directory)
    plot_spider_chart(
        weighted_scaled_matrix,
        weights,
        "Spider Chart of Alternatives with weighting",
        decision_matrix,
        user_input,
        directory)
