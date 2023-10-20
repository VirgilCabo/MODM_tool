import pandas as pd
import numpy as np
from TOPSIS_main_data_processing import main_data_processing


def get_weight_combinations(original_weights):

    # Prompt user for range and increment
    try:
        variation_range = float(input("Enter the percentage range for weight variation (e.g., 20 for ±20%): ")) / 100
        increment = float(input("Enter the increment value for weight variation (e.g., 0.05): "))
    except ValueError:
        print("Invalid input. Please enter a valid number.")
        return pd.DataFrame()

    combinations = []

    # For each criterion, vary its weight and adjust the others
    for criterion in original_weights:
        min_weight = max(0, original_weights[criterion] - variation_range * original_weights[criterion])
        max_weight = min(1, original_weights[criterion] + variation_range * original_weights[criterion])

        # Create varied weights based on increment
        for varied_weight in np.arange(min_weight, max_weight + increment, increment):
            adjusted_weights = {}
            adjustment_factor = (1 - varied_weight) / (1 - original_weights[criterion])
            
            for crit, weight in original_weights.items():
                if crit == criterion:
                    adjusted_weights[crit] = varied_weight
                else:
                    adjusted_weights[crit] = weight * adjustment_factor

            if all(0 <= w <= 1 for w in adjusted_weights.values()):
                combinations.append(adjusted_weights)

    # Convert the list of dictionaries to a DataFrame
    weight_combinations = pd.DataFrame(combinations)

    return weight_combinations


def run_sensitivity_analysis(decision_matrix, weight_combinations, beneficial_criteria):
    # Initialize an empty DataFrame to store results
    columns = ['Weight_' + crit for crit in decision_matrix.columns]
    for alt in decision_matrix.index:
        columns.append('Score_' + alt)
        columns.append('Rank_' + alt)
    results_df = pd.DataFrame(columns=columns)

    # Iterate over weight combinations
    for index, row in weight_combinations.iterrows():
        weights = row.to_dict()
        ranked_alternatives, ranks, weighted_normalized_matrix, S = main_data_processing(decision_matrix, weights, beneficial_criteria)

        # Prepare a row to append to the results DataFrame
        row_data = list(weights.values())  # Start with the weight combination
        for alt in decision_matrix.index:
            row_data.append(S[alt])
            row_data.append(ranks[alt])
        
        # Append the row to the results DataFrame
        results_df.loc[len(results_df)] = row_data
    
    return results_df