import pandas as pd
import re
import os
import datetime


def load_data(file_path):
    # Load the Excel file into a DataFrame
    # file_path = 'data_input/mock3_data.xlsx'
    global data_filename
    data_filename = os.path.basename(file_path)
    decision_matrix = pd.read_csv(file_path, index_col=0)
    print(decision_matrix)
    return decision_matrix, data_filename


def define_criteria_nature(criteria_list):
    # Initialize empty lists for beneficial and non-beneficial criteria
    beneficial_criteria = []
    non_beneficial_criteria = []

    # Prompt the user to specify the type for each criterion
    for criterion in criteria_list:
        response = input(
            f"Is '{criterion}' beneficial (maximize) or non-beneficial (minimize)? Enter 'b' for beneficial or 'n' for non-beneficial: ").lower()

        while response not in ['b', 'n']:
            print(
                "Invalid input. Please enter 'b' for beneficial or 'n' for non-beneficial.")
            response = input(
                f"Is '{criterion}' beneficial (maximize) or non-beneficial (minimize)? ").lower()

        if response == 'b':
            beneficial_criteria.append(criterion)
        else:
            non_beneficial_criteria.append(criterion)

    # print("\nBeneficial Criteria:", beneficial_criteria)
    # print("Non-Beneficial Criteria:", non_beneficial_criteria)
    return beneficial_criteria, non_beneficial_criteria


def get_integer_input(prompt_message):
    while True:
        try:
            value = int(input(prompt_message))
            return value
        except ValueError:
            print("Please enter a valid integer.")


def define_weights(criteria_list):
    # Prompt the user for weights for each criterion
    weights = {}
    for criterion in criteria_list:
        while True:
            weight = get_integer_input(
                f"Please assign a weight (1-10) for {criterion} (1 being the least important weight and 10 being the most important weight): ")
            if weight in list(range(11)):
                weights[criterion] = weight
                break
            else:
                print("Weight should be an integer between 0 and 10. Please try again.")
        weights[criterion] = weight
    labels_with_weights = [
        f"{col} ({weights[col]:.2f})" for col in criteria_list]
    return weights


def normalize_weight(weights):
    # Normalize the weights
    total_weight = sum(weights.values())
    normalized_weights = {
        criterion: weight /
        total_weight for criterion,
        weight in weights.items()}
    # print("\nNormalized Weights:")
    # for criterion, weight in normalized_weights.items():
    # print(f"{criterion}: {weight:.2f}")
    return normalized_weights


def min_max_normalization(decision_matrix, beneficial_criteria):
    """
    Normalize the decision matrix using min-max normalization.
    
    Parameters:
    - decision_matrix: DataFrame containing the raw scores for each alternative and criterion.
    
    Returns:
    - normalized_matrix: DataFrame containing the normalized scores.
    """
    normalized_matrix = (decision_matrix - decision_matrix.min()) / (decision_matrix.max() - decision_matrix.min())
    for criterion in decision_matrix.columns:
        if criterion not in beneficial_criteria:
            normalized_matrix[criterion] = 1 - normalized_matrix[criterion]
    return normalized_matrix


def vector_normalization(decision_matrix, beneficial_criteria):
    # Normalize the decision matrix with vector normalization
    normalized_matrix = decision_matrix.div(
        decision_matrix.pow(2).sum(
            axis=0).pow(0.5), axis=1)
    for criterion in decision_matrix.columns:
        if criterion not in beneficial_criteria:
            normalized_matrix[criterion] = 1 - normalized_matrix[criterion]
    # print(normalized_matrix)
    return normalized_matrix


def normalization(decision_matrix, beneficial_criteria):
    response = input(f"What normalization method do you want to use for your dataset?\nEnter 'm' for Min-Max Normalization.\nEnter 'v' for Vector Normalization.\n").lower()
    while response not in ['m', 'v']:
        print(
            "Invalid input. Please enter 'm' or 'v'.")
        response = input(
            f"What normalization method do you want to use? ").lower()
    if response == 'm':
            normalized_matrix = min_max_normalization(decision_matrix, beneficial_criteria)   
    elif response == 'v':
            normalized_matrix = vector_normalization(decision_matrix, beneficial_criteria)
    print(normalized_matrix)
    return normalized_matrix


def gathering_data(file_path):
    decision_matrix, data_filename = load_data(file_path)
    beneficial_criteria, non_beneficial_criteria = define_criteria_nature(
        decision_matrix.columns)
    normalized_matrix = normalization(decision_matrix, beneficial_criteria)
    weights = define_weights(decision_matrix.columns)
    normalized_weights = normalize_weight(weights)
    return decision_matrix, normalized_matrix, data_filename, weights, normalized_weights, beneficial_criteria, non_beneficial_criteria
