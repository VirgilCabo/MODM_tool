import numpy as np
import pandas as pd
import itertools
import random
import matplotlib.pyplot as plt
from TOPSIS_main_data_processing import main_data_processing


def get_user_uncertainties(initial_weights):
    uncertainties = {}
    for criterion, weight in initial_weights.items():
        print(f"Initial weight for {criterion}: {int(weight)}")
        lower_bound = float(input(f"Enter the lower bound of your confidence interval for {criterion}: "))
        upper_bound = float(input(f"Enter the upper bound of your confidence interval for {criterion}: "))
        
        # Assuming a 95% confidence interval, we can derive the standard deviation.
        std_dev = (upper_bound - lower_bound) / (2 * 1.96)
        uncertainties[criterion] = std_dev
    return uncertainties


def generate_random_weights_for_criterion(initial_weight, std_dev, num_samples=100):
    weights = []
    for _ in range(num_samples):
        weight = -1  # Initialize with a negative value to enter the while loop
        while weight < 0:  # Keep sampling until we get a non-negative weight
            weight = np.random.normal(initial_weight, std_dev)
        weights.append(weight)
    return weights


def generate_all_random_weights(initial_weights, uncertainties, num_samples=100):
    all_random_weights = {}
    for criterion, weight in initial_weights.items():
        all_random_weights[criterion] = generate_random_weights_for_criterion(weight, uncertainties[criterion], num_samples)
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


def efficient_sample_combinations(all_random_weights, num_samples=1000):
    weight_sets = []
    
    # List of criteria
    criteria = list(all_random_weights.keys())
    
    for _ in range(num_samples):
        # Randomly select one weight for each criterion
        sampled_weights = [random.choice(all_random_weights[criterion]) for criterion in criteria]
        
        # Combine the selected weights into a dictionary
        weight_dict = dict(zip(criteria, sampled_weights))
        weight_sets.append(weight_dict)
    
    return weight_sets


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


def run_sensitivity_analysis(decision_matrix, weight_sets, beneficial_criteria):
    # Initialize an empty DataFrame to store results
    columns = ['Weight_' + crit for crit in decision_matrix.columns]
    for alt in decision_matrix.index:
        columns.append('Score_' + alt)
        columns.append('Rank_' + alt)
    results_df = pd.DataFrame(columns=columns)

    # Iterate over weight combinations
    for weights in weight_sets:
        ranked_alternatives, ranks, weighted_normalized_matrix, S = main_data_processing(decision_matrix, weights, beneficial_criteria)

        # Prepare a row to append to the results DataFrame
        row_data = list(weights.values())  # Start with the weight combination
        for alt in decision_matrix.index:
            row_data.append(S[alt])
            row_data.append(int(ranks[alt]))
        
        # Append the row to the results DataFrame
        results_df.loc[len(results_df)] = row_data
    
    return results_df


def generate_weight_sets(initial_weights, num_samples, num_sets):
    uncertainties = get_user_uncertainties(initial_weights)
    all_random_weights = generate_all_random_weights(initial_weights, uncertainties, num_samples)
    weight_sets = efficient_sample_combinations(all_random_weights, num_sets)
    normalized_weight_sets = normalize_weight_sets(weight_sets)
    return normalized_weight_sets, num_sets


def assess_reliability(S, num_sets, results_df):
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

    rank_list_initial_best = results_df[f'Rank_{initial_best_solution}'].tolist()

    for rank in rank_list_initial_best:
        # Determine the best solution for this simulation
        if rank == 1.0:
            count += 1

    reliability_percentage = (count / total_simulations) * 100
    return reliability_percentage




""" initial_weights = {
    'C1' : 1,
    'C2' : 2,
    'C3' : 3
}

#plot_histogram(all_random_weights['C1'], 'C1')
normalized_weight_sets = generate_weight_sets(initial_weights, num_sample=100, num_sets=100) 
print(normalized_weight_sets) """






