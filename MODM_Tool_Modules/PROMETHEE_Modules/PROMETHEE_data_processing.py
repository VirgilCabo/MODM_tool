import pandas as pd
import numpy as np
import math



def usual_preference_function(d):
    """
    Usual Preference Function.
    
    Parameters:
    - d (float): Difference between evaluations of two alternatives for a criterion.
    
    Returns:
    - float: Preference value.
    """
    if d <= 0:
        return 0
    else:
        return 1


def linear_preference_function(d, q, p):
    """
    Linear Preference Function.
    
    Parameters:
    - d (float): Difference between evaluations of two alternatives for a criterion.
    - q (float): Indifference threshold.
    - p (float): Strict preference threshold.
    
    Returns:
    - float: Preference value.
    """
    if d <= q:
        return 0
    elif q < d <= p:
        return (d - q) / (p - q)
    else:
        return 1


def gaussian_preference_function(d, s):
    """
    Gaussian Preference Function.
    
    Parameters:
    - d (float): Difference between evaluations of two alternatives for a criterion.
    - s (float): Standard deviation of the Gaussian function.
    
    Returns:
    - float: Preference value.
    """
    if d <= 0:
        return 0
    return 1 - math.exp(-d**2 / (2 * s**2))


def compute_d_values(decision_matrix):
    """
    Computes the pairwise differences for each criterion and each pair of alternatives.
    
    Parameters:
    - decision_matrix: A pandas DataFrame containing the normalized decision matrix.
    
    Returns:
    - A pandas DataFrame containing the pairwise differences.
    """
    alternatives = decision_matrix.index
    criteria = decision_matrix.columns
    
    # Create an empty dataframe to store the results
    d_values_df = pd.DataFrame()
    
    # Iterate over each criterion
    for criterion in criteria:
        # Compute pairwise differences for the current criterion
        for alt_i in alternatives:
            for alt_j in alternatives:
                if alt_i != alt_j:
                    d_key = f"d({alt_i}, {alt_j})"
                    d = decision_matrix.loc[alt_i, criterion] - decision_matrix.loc[alt_j, criterion]
                    d_values_df.loc[d_key, criterion] = d
                    
    return d_values_df


def compute_preference_values(d_values_df, preference_functions):
    """
    Compute the preference values for each criterion and each pair of alternatives.
    
    Parameters:
    - differences_df: DataFrame containing the pairwise differences.
    - preference_functions: Dictionary mapping criteria to their corresponding preference functions.
    
    Returns:
    - DataFrame containing the preference values.
    """
    preference_values_df = d_values_df.copy()
    preference_values_df.index = ['p(' + str(idx) + ')'for idx in preference_values_df.index]


    for criterion, pref_func in preference_functions.items():
        preference_values_df[criterion] = preference_values_df[criterion].apply(pref_func)
    
    return preference_values_df


def define_preference_functions(decision_matrix):
    preference_functions = {}
    for criterion in decision_matrix.columns:
        response = input(f"What preference function do you want to use for {criterion}?\nEnter 'u' for usual preference function.\nEnter 'l' for linear preference function.\nEnter 'g' for gaussian preference function.\n").lower()
        while response not in ['u', 'l', 'g']:
            print(
                "Invalid input. Please enter 'u', 'l' or 'g'.")
            response = input(
                f"What preference function do you want to use for {criterion}? ").lower()
        if response == 'u':
            preference_functions[criterion] = lambda d: usual_preference_function(d)   
        elif response == 'l':
            preference_functions[criterion] = lambda d: linear_preference_function(d, q=0.2, p=0.8) 
        else:
            preference_functions[criterion] = lambda d: gaussian_preference_function(d, s=0.3)
    return preference_functions


def compute_global_preference_values(preference_values_df, normalized_weights, decision_matrix):
    """
    Compute the global preference values for pairs of alternatives.

    Parameters:
    - preference_matrix: DataFrame containing preference values for each criterion.
    - weights: Dictionary containing weights for each criterion.

    Returns:
    - DataFrame containing global preference values for pairs of alternatives.
    """
    # Multiply the preference matrix by the weights
    weighted_preference_values_df = preference_values_df.multiply(normalized_weights)
    
    # Sum across the criteria to get the global preference values
    global_preference_values_serie = weighted_preference_values_df.sum(axis=1)

    # Modify the label names
    alternatives = decision_matrix.index
    index_keys = []
    for alt_i in alternatives:
            for alt_j in alternatives:
                if alt_i != alt_j:
                    index_key = f"P({alt_i}, {alt_j})"
                    index_keys.append(index_key)
    global_preference_values_serie.index = [idx for idx in index_keys]

    return global_preference_values_serie


def series_to_matrix(global_preference_values_serie, decision_matrix):
    alternatives = decision_matrix.index
    matrix_df = pd.DataFrame(0.0, index=alternatives, columns=alternatives)

    for alt_i in alternatives:
        for alt_j in alternatives:
            if alt_i != alt_j:
                key = f"P({alt_i}, {alt_j})"
                value = global_preference_values_serie.get(key, None)
                if value is not None:
                    matrix_df.at[alt_i, alt_j] = value
                else:
                    print(f"Key not found: {key}")
    return matrix_df


def calculate_flows(matrix_df):
    """
    Calculate the positive, negative, and net outranking flows.

    Parameters:
    - global_preference_matrix: DataFrame containing global preference values.
    - n: The number of alternatives.

    Returns:
    - Tuple of DataFrames for positive, negative, and net flows.
    """
    n = len(matrix_df)
    # Calculate positive flows
    positive_flows = matrix_df.sum(axis=1) / (n - 1)
    
    # Calculate negative flows
    negative_flows = matrix_df.sum(axis=0) / (n - 1)
    
    # Calculate net flows
    net_flows = positive_flows - negative_flows
    
    ranked_alternatives = net_flows.sort_values(ascending=False)

    ranks = net_flows.rank(ascending=False).astype(int)

    return net_flows, ranked_alternatives, ranks


def PROMETHEE_data_processing(decision_matrix, normalized_matrix, normalized_weights, preference_functions):
    d_values_df = compute_d_values(normalized_matrix)
    preference_values_df = compute_preference_values(d_values_df, preference_functions)
    global_preference_values_serie = compute_global_preference_values(preference_values_df, normalized_weights, decision_matrix)
    matrix_df = series_to_matrix(global_preference_values_serie, decision_matrix)
    net_flows, ranked_alternatives, ranks = calculate_flows(matrix_df)
    return net_flows, ranked_alternatives, ranks


