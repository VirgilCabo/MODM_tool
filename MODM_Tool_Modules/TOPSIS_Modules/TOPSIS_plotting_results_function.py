import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure as fig
import math
import re
import os
import datetime
import seaborn as sns
import numpy as np
import plotly.graph_objs as go 
from scipy.spatial import ConvexHull
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D



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
    cleaned_names = [re.sub(r'\(.*\)', '', criterion).strip()
               for criterion in criteria]
    return cleaned_names


def plot_bar_chart_topsis(scores, weights, user_input, directory):
    # Sort scores for better visualization
    sorted_scores = scores.sort_values(ascending=False)

    # Plot
    df = sorted_scores.reset_index()
    df.columns = ['Alternatives', 'Performance Score']
    order = df['Alternatives'].tolist()
    norm = plt.Normalize(df['Performance Score'].min(), df['Performance Score'].max())
    colors = plt.cm.cividis(norm(df['Performance Score']))
    barplot = sns.barplot(
        x='Alternatives',
        y='Performance Score',
        data=df,
        palette=list(colors),
        legend=False,
        order=order)
    sm = ScalarMappable(cmap=plt.cm.cividis, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=barplot, orientation='vertical')
    plt.title('Performance Scores of Alternatives')
    plt.ylabel('Performance Score')
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


def pareto_surface(weighted_normalized_matrix, ideal_best, ideal_worst, user_input, directory):
    criteria = list(weighted_normalized_matrix.columns)

    criteria_names = clean_criteria_names(criteria)
    # Extracting Ideal and Anti-Ideal Points
    utopian_point = [ideal_best[criterion] for criterion in criteria]
    nadir_point = [ideal_worst[criterion] for criterion in criteria]
    # Extract the Pareto optimal points
    pareto_points = weighted_normalized_matrix[criteria].values

    # Use ConvexHull to determine the convex hull of the Pareto points
    hull = ConvexHull(pareto_points)

    # Use the vertices of the convex hull to create a surface
    x_surf = [pareto_points[v, 2] for v in hull.vertices]
    y_surf = [pareto_points[v, 1] for v in hull.vertices]
    z_surf = [pareto_points[v, 0] for v in hull.vertices]

    # Create a 3D plot
    fig = plt.figure(figsize=(8, 8), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1,1,1.1])
    # Plot Pareto optimal alternatives
    for _, row in weighted_normalized_matrix.iterrows():
        ax.scatter(row[criteria[2]], row[criteria[1]], row[criteria[0]], color='b', marker='o', s=10)

    # Add labels to each Pareto point
    for i, txt in enumerate(weighted_normalized_matrix.index):
        x, y, z = pareto_points[i, 2], pareto_points[i, 1], pareto_points[i, 0]
        ax.text(x , y , z , txt, size=10, color='k')
    
    # Plot Ideal and Anti-Ideal Points
    ax.scatter(*utopian_point, color='g', marker='o', s=10)  
    ax.scatter(*nadir_point, color='r', marker='o', s=10)  

    ax.text(*utopian_point, 'Utopian', size=12, color='k')
    ax.text(*nadir_point, 'Nadir', size=12, color='k')

    # Draw lines indicating distance to utopian and nadir points
    for _, row in weighted_normalized_matrix.iterrows():
        ax.plot([row[criteria[2]], utopian_point[2]], [row[criteria[1]], utopian_point[1]], [row[criteria[0]], utopian_point[0]], 'g--', linewidth=0.5)
        ax.plot([row[criteria[2]], nadir_point[2]], [row[criteria[1]], nadir_point[1]], [row[criteria[0]], nadir_point[0]], 'r--', linewidth=0.5)

    # Plot the convex hull as a surface
    ax.plot_trisurf(x_surf, y_surf, z_surf, color='lightblue', alpha=0.5, zorder=0)

    # Labeling
    ax.set_xlabel('Resources')
    ax.set_ylabel('Ecosystems')
    ax.set_zlabel('Profit')
    ax.set_title('3D Pareto Front')

    # Adjust view angle
    ax.view_init(elev=20, azim=-45)

    if user_input == 'yes':
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                directory,
                'pareto_surface.png'),
            dpi=500,
            bbox_inches='tight')
    plt.tight_layout()
    plt.show()


def interactive_pareto_surface(weighted_normalized_matrix, ideal_best, ideal_worst, weights):
    plotly_data = weighted_normalized_matrix.reset_index()  # Reset index to use row labels as hover info
    criteria = list(weighted_normalized_matrix.columns)
    utopian_point = [ideal_best[criterion] for criterion in criteria]
    nadir_point = [ideal_worst[criterion] for criterion in criteria]
    if not 'index' in plotly_data.columns:
        plotly_data.reset_index(inplace=True)
        plotly_data.rename(columns={'index': 'labels'}, inplace=True)
    plotly_data['labels'] = plotly_data['labels'].astype(int)
    plotly_data['labels'] = plotly_data['labels'] + 1
    # Extract the Pareto optimal points
    pareto_points = weighted_normalized_matrix[criteria].values

    # Calculate the convex hull for the surface, if applicable
    """ hull = ConvexHull(pareto_points)
    hull_indices = hull.simplices.flatten()
    hull_points = pareto_points[hull_indices] """
    # Create a scatter3d plot
    pareto_trace = go.Scatter3d(
        x=plotly_data[criteria[2]],
        y=plotly_data[criteria[1]],
        z=plotly_data[criteria[0]],
        mode='markers+text',
        marker=dict(
            size=5,
            color='blue',  # Color of the points
        ),
        text=plotly_data['labels'],  # This will be the hover text
        hoverinfo='text',  # Only show the hover text
        name='Pareto Points'
    )

    nadir_point_trace = go.Scatter3d(
        x=[nadir_point[2]],
        y=[nadir_point[1]],
        z=[nadir_point[0]],
        mode='markers',
        marker=dict(
            size=5,
            color='red',
        ),
        name='Nadir Point'
    )

    utopian_point_trace = go.Scatter3d(
        x=[utopian_point[2]],
        y=[utopian_point[1]],
        z=[utopian_point[0]],
        mode='markers',
        marker=dict(
            size=5,
            color='green',
        ),
        name='Utopian Point'
    )

    """ surface_trace = go.Mesh3d(
        x=hull_points[:, 2],
        y=hull_points[:, 1],
        z=hull_points[:, 0],
        color='lightblue',
        opacity=0.8,
        name='Pareto Surface'
    ) """

    lines_to_nadir = []
    lines_to_utopian = []

    for point in pareto_points:
        lines_to_nadir.append(go.Scatter3d(
            x=[point[2], nadir_point[2]],
            y=[point[1], nadir_point[1]],
            z=[point[0], nadir_point[0]],
            mode='lines',
            line=dict(color='red', width=2, dash='dashdot'),
            showlegend=False
        ))
        
        lines_to_utopian.append(go.Scatter3d(
            x=[point[2], utopian_point[2]],
            y=[point[1], utopian_point[1]],
            z=[point[0], utopian_point[0]],
            mode='lines',
            line=dict(color='green', width=2, dash='dashdot'),
            showlegend=False
        ))


    # Create the layout of the plot, including titles for axes
    layout = go.Layout(
        title='3D Pareto Front',
        scene=dict(
            xaxis=dict(title='Resources'),
            yaxis=dict(title='Ecosystems'),
            zaxis=dict(title='Profit'),
            aspectmode='manual',
            aspectratio=dict(x=weights[criteria[2]], y=weights[criteria[1]], z=weights[criteria[0]])
        ),
        margin=dict(l=0, r=0, b=0, t=0)  # Tight layout
    )

    # Create the figure
    fig = go.Figure(data=[pareto_trace, nadir_point_trace, utopian_point_trace] + lines_to_nadir + lines_to_utopian, layout=layout)

    #fig.write_image("pareto.png", width=1980, height=1080, scale=3)  # Adjust width, height, and scale as needed

    # Show the figure
    fig.show()


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


def results_visualization_topsis(
        ranked_alternatives,
        weighted_normalized_matrix,
        beneficial_criteria,
        weights,
        normalized_weights,
        S,
        user_input,
        directory,
        decision_matrix,
        ideal_best,
        ideal_worst):
    
    """ scaled_matrix, weighted_scaled_matrix = min_max_scaling_for_spider_chart(
        weighted_normalized_matrix, beneficial_criteria, normalized_weights) """
    if len(list(weighted_normalized_matrix.columns)) == 3:
        #pareto_surface(weighted_normalized_matrix, ideal_best, ideal_worst, user_input, directory)
        interactive_pareto_surface(weighted_normalized_matrix, ideal_best, ideal_worst, weights)    
    plot_bar_chart_topsis(S, weights, user_input, directory)
    """ plot_spider_chart(
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
 """
    print("\nRanked Alternatives:")
    print(ranked_alternatives)