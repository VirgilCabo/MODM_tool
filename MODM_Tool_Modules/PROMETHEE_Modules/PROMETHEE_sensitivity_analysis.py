import numpy as np
import pandas as pd
import itertools
import random
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import ScalarMappable
import os
from joypy import joyplot
from tqdm import tqdm
from MODM_Tool_Modules.gathering_data_function import get_integer_input


def get_user_uncertainties(initial_weights):
    uncertainties = {}
    for criterion, weight in initial_weights.items():
        print(f"Initial weight for {criterion}: {int(weight)}")
    for criterion, weight in initial_weights.items():
        while True:
            lower_bound_input = get_integer_input(f"Enter the lower bound of your confidence interval for {criterion}: ")
            if lower_bound_input >= 0 and lower_bound_input <= weight:
                lower_bound = lower_bound_input
                break
            else:
                print(f"The lower bound should be an integer between 0 and {weight}. Please try again.")
        while True:
            upper_bound_input = get_integer_input(f"Enter the upper bound of your confidence interval for {criterion}: ")
            if upper_bound_input >= weight and upper_bound_input <= 10:
                upper_bound = upper_bound_input
                break
            else:
                print(f"The upper bound should be an integer between {weight} and 10. Please try again.")

        # Assuming a 95% confidence interval, we can derive the standard
        # deviation.
        std_dev = (upper_bound - lower_bound) / (2 * 1.96)
        uncertainties[criterion] = std_dev
    return uncertainties


def generate_random_weights_for_criterion(
        initial_weight,
        std_dev,
        num_samples,
        lower_limit,
        upper_limit):
    weights = []
    for _ in range(num_samples):
        weight = np.random.normal(initial_weight, std_dev)
        while weight < lower_limit or weight > upper_limit:
            if weight < lower_limit:
                weight = lower_limit + (lower_limit - weight)
            elif weight > upper_limit:
                weight = upper_limit - (weight - upper_limit)
        weights.append(weight)
    return weights


def generate_all_random_weights(
        initial_weights,
        uncertainties,
        num_samples,
        lower_limit,
        upper_limit):
    all_random_weights = {}
    for criterion, weight in initial_weights.items():
        all_random_weights[criterion] = generate_random_weights_for_criterion(
            weight, uncertainties[criterion], num_samples, lower_limit, upper_limit)
    return all_random_weights


def sample_combinations(all_random_weights, num_samples=1000):
    # Get the Cartesian product of all lists of random weights
    all_combinations = list(itertools.product(*all_random_weights.values()))

    # Randomly sample from the list of all combinations
    sampled_combinations = random.sample(all_combinations, num_samples)

    # Convert the sampled combinations to a list of dictionaries
    sampled_dicts = []
    for combination in sampled_combinations:
        weight_dict = dict(zip(all_random_weights.keys(), combination))
        sampled_dicts.append(weight_dict)

    return sampled_dicts


def efficient_sample_combinations(all_random_weights):
    weight_sets = []
    while True:
            num_sets_input = get_integer_input(
                "Please enter the number of weight sets you wish to generate: ")
            if num_sets_input <= 1000000:
                num_sets = num_sets_input
                break
            else:
                print("The number of weight sets is too high, it will result in unnecessary computational effort. Please choose a number of sets below 1 million.")
    # List of criteria
    criteria = list(all_random_weights.keys())

    for _ in range(num_sets):
        # Randomly select one weight for each criterion
        sampled_weights = [
            random.choice(
                all_random_weights[criterion]) for criterion in criteria]

        # Combine the selected weights into a dictionary
        weight_dict = dict(zip(criteria, sampled_weights))
        weight_sets.append(weight_dict)

    return weight_sets, num_sets


def normalize_weight_sets(weight_sets):
    # Normalize the weights
    normalized_weight_sets = []
    for weights in weight_sets:
        total_weight = sum(weights.values())
        normalized_weights = {
            criterion: weight /
            total_weight for criterion,
            weight in weights.items()}
        normalized_weight_sets.append(normalized_weights)
    return normalized_weight_sets


def plot_histogram(weights, criterion_name):
    plt.hist(weights, bins=30, density=True, alpha=0.75)
    plt.title(f'Distribution of Weights for {criterion_name}')
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()


def run_sensitivity_analysis(
        function,
        decision_matrix,
        weight_sets,
        normalized_matrix,
        preference_functions):
    # Initialize an empty DataFrame to store results
    columns = []
    for alt in decision_matrix.index:
        columns.append('Net_Flow_' + str(alt))
        columns.append('Rank_' + str(alt))
    results_df = pd.DataFrame(columns=columns)

    # Iterate over weight combinations
    for weights in tqdm(weight_sets, colour='green'):
        S, ranked_alternatives, ranks = function(
            decision_matrix, normalized_matrix, weights, preference_functions)

        # Prepare a row to append to the results DataFrame
        row_data = []
        for alt in decision_matrix.index:
            row_data.append(S[alt])
            row_data.append(int(ranks[alt]))

        # Append the row to the results DataFrame
        results_df.loc[len(results_df)] = row_data

    # Separate performance scores
    score_columns = [col for col in results_df.columns if 'Net_Flow_' in col]
    scores_df = results_df[score_columns]

    # Separate ranks
    rank_columns = [col for col in results_df.columns if 'Rank_' in col]
    ranks_df = results_df[rank_columns]

    scores_df.columns = [alt for alt in decision_matrix.index]
    ranks_df.columns = [alt for alt in decision_matrix.index]

    return scores_df, ranks_df


def generate_weight_sets(
        initial_weights,
        num_samples,
        lower_limit,
        upper_limit):
    uncertainties = get_user_uncertainties(initial_weights)
    all_random_weights = generate_all_random_weights(
        initial_weights, uncertainties, num_samples, lower_limit, upper_limit)
    weight_sets, num_sets = efficient_sample_combinations(all_random_weights)
    normalized_weight_sets = normalize_weight_sets(weight_sets)
    return normalized_weight_sets, num_sets, uncertainties


def assess_reliability(S, num_sets, ranks_df):
    """
    Assess the reliability of the initial best solution based on simulation results.

    Parameters:
    - initial_best_solution: The best solution from the initial run.
    - simulation_results: A list of dictionaries where each dictionary represents the results of a simulation.
                          The key is the alternative, and the value is its score.

    Returns:
    - reliability_percentage: The percentage of times the initial best solution remains the best solution in the simulations.
    """

    count = 0
    total_simulations = num_sets
    initial_best_solution = S.idxmax()
    rank_list_initial_best = ranks_df[initial_best_solution].tolist()

    for rank in rank_list_initial_best:
        # Determine the best solution for this simulation
        if rank == 1.0:
            count += 1
    reliability_percentage = round((count / total_simulations) * 100, 2)
    print(initial_best_solution)
    print(reliability_percentage)
    return reliability_percentage, initial_best_solution


def boxplot_sensitivity_results(scores_df, user_input, directory):
    # Transpose the DataFrame so that each row represents an alternative and
    # each column is a performance score from a different weight set
    transposed_scores__df = scores_df.transpose()
    # Melt the scores_df to long-form
    long_form_scores = scores_df.melt(
        var_name="Alternatives",
        value_name="Net Outranking Flow")
    colors = sns.color_palette("viridis", len(scores_df.columns))

    # Create the boxplot
    sns.boxplot(
        x="Alternatives",
        y="Net Outranking Flow",
        data=long_form_scores,
        palette=colors,
        hue="Alternatives")
    plt.title('Distribution of Net Outranking Flows for Each Alternative')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    if user_input == 'yes':
        plt.savefig(os.path.join(directory, 'box_plot.png'), dpi=500)
    plt.show()


def ridgelineplot_sensitivity_results(scores_df, user_input, directory):
    joyplot(
        data=scores_df,
        title='Net Outranking Flow Distributions for Alternatives',
        overlap=2,  # Adjust as needed
        colormap=plt.cm.tab20b,  # Choose a colormap
        grid=False,  # Show grid
        legend=True,  # Show legend
        linecolor='k',
        linewidth=0.5,
        alpha=0.9
    )
    plt.xlabel("Net Outranking Flow")
    if user_input == 'yes':
        plt.savefig(os.path.join(directory, 'ridgeline_plot.png'), dpi=500)
    plt.show()


def top_ranked_percentage(ranks):
    num_sets = len(ranks[1])
    top_ranked_percentage = {}
    for column in ranks.columns:
        counts = ranks[column].value_counts().get(1, 0)
        percentage = round((counts/num_sets)*100, 2)
        top_ranked_percentage[column] = percentage
    filtered_top_rank_percentage = {key: value for key, value in top_ranked_percentage.items() if value >= 0.1}
    filtered_top_serie = pd.Series(filtered_top_rank_percentage)
    top_serie = pd.Series(top_ranked_percentage)
    return filtered_top_serie, top_serie


def plot_bar_chart_topsis(scores, user_input, directory):
    # Sort scores for better visualization
    sorted_scores = scores.sort_values(ascending=False)

    # Plot
    df = sorted_scores.reset_index()
    df.columns = ['Alternatives', 'Top Ranked Percentage']
    order = df['Alternatives'].tolist()
    norm = plt.Normalize(df['Top Ranked Percentage'].min(), df['Top Ranked Percentage'].max())
    colors = plt.cm.cividis(norm(df['Top Ranked Percentage']))
    barplot = sns.barplot(
        x='Alternatives',
        y='Top Ranked Percentage',
        data=df,
        palette=list(colors),
        legend=False,
        order=order)
    sm = ScalarMappable(cmap=plt.cm.cividis, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=barplot, orientation='vertical')
    plt.title('Reliability Percentage for Alternatives')
    plt.ylabel('Reliability Percentage (%)')
    plt.xlabel('Alternatives')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    if user_input == 'yes':
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                directory,
                'top_ranked_percentage_barplot.png'),
            dpi=500,
            bbox_inches='tight')
    plt.show()


def sensitivity_analysis(
        function,
        initial_weights,
        num_samples,
        lower_limit,
        upper_limit,
        decision_matrix,
        normalized_matrix,
        user_input,
        directory,
        preference_functions):
    normalized_weight_sets, num_sets, uncertainties = generate_weight_sets(
        initial_weights, num_samples, lower_limit, upper_limit)
    scores_df, ranks_df = run_sensitivity_analysis(
        function, decision_matrix, normalized_weight_sets, normalized_matrix, preference_functions)
    ridgelineplot_sensitivity_results(scores_df, user_input, directory)
    filtered_top_serie, top_serie = top_ranked_percentage(ranks_df)
    plot_bar_chart_topsis(filtered_top_serie, user_input, directory)
    return uncertainties, scores_df, ranks_df, filtered_top_serie, top_serie
