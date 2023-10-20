import pandas as pd


def normalize_matrix(decision_matrix):
    # Normalize the decision matrix with vector normalization
    normalized_matrix = decision_matrix.div(
        decision_matrix.pow(2).sum(
            axis=0).pow(0.5), axis=1)
    # print(normalized_matrix)
    return normalized_matrix


def apply_weights(normalized_matrix, normalized_weights):
    # Multiply the normalized decision matrix by the normalized weights
    weighted_normalized_matrix = normalized_matrix.multiply(
        normalized_weights, axis=1)
    # print(weighted_normalized_matrix)
    return weighted_normalized_matrix


def determine_ideal_best_and_worst(
        weighted_normalized_matrix,
        beneficial_criteria):
    # Determine the ideal best and worst values for each criterion
    ideal_best = {}
    ideal_worst = {}

    for criterion in weighted_normalized_matrix.columns:
        if criterion in beneficial_criteria:
            ideal_best[criterion] = weighted_normalized_matrix[criterion].max()
            ideal_worst[criterion] = weighted_normalized_matrix[criterion].min()
        else:
            ideal_best[criterion] = weighted_normalized_matrix[criterion].min()
            ideal_worst[criterion] = weighted_normalized_matrix[criterion].max()

    # print("Ideal Best Values:", ideal_best)
    # print("Ideal Worst Values:", ideal_worst)
    return ideal_best, ideal_worst


def calculate_euclidian_distance(
        weighted_normalized_matrix,
        ideal_best,
        ideal_worst):
    # Calculate the Euclidean distances from the ideal best and worst values
    D_plus = ((weighted_normalized_matrix - ideal_best)
              ** 2).sum(axis=1).pow(0.5)
    D_minus = ((weighted_normalized_matrix - ideal_worst)
               ** 2).sum(axis=1).pow(0.5)
    # print("Distances from Ideal Best (D+):")
    # print(D_plus)
    # print("\nDistances from Ideal Worst (D-):")
    # print(D_minus)
    return D_plus, D_minus


def calculate_performance_score(D_plus, D_minus):
    # Calculate the performance score for each alternative
    S = D_minus / (D_plus + D_minus)
    # print("Performance Scores:")
    # print(S)
    return S


def ranking(S):
    # Rank the alternatives based on their performance scores
    ranked_alternatives = S.sort_values(ascending=False)
    ranks = S.rank(ascending=False).astype(int)
    #print(ranks)
    # print("\nRanked Alternatives:")
    # print(ranked_alternatives)
    return ranked_alternatives, ranks


def main_data_processing(
        decision_matrix,
        normalized_weights,
        beneficial_criteria):
    normalized_matrix = normalize_matrix(decision_matrix)
    weighted_normalized_matrix = apply_weights(
        normalized_matrix, normalized_weights)
    ideal_best, ideal_worst = determine_ideal_best_and_worst(
        weighted_normalized_matrix, beneficial_criteria)
    D_plus, D_minus = calculate_euclidian_distance(
        weighted_normalized_matrix, ideal_best, ideal_worst)
    S = calculate_performance_score(D_plus, D_minus)
    ranked_alternatives, ranks = ranking(S)
    return ranked_alternatives, ranks, weighted_normalized_matrix, S