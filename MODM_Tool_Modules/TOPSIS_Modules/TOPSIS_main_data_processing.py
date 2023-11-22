import pandas as pd


def apply_weights(normalized_matrix, normalized_weights):
    # Multiply the normalized decision matrix by the normalized weights
    weighted_normalized_matrix = normalized_matrix.multiply(
        normalized_weights, axis=1)
    # print(weighted_normalized_matrix)
    return weighted_normalized_matrix


def determine_ideal_best_and_worst(weighted_normalized_matrix):
    # Determine the ideal best and worst values for each criterion
    ideal_best = {}
    ideal_worst = {}

    for criterion in weighted_normalized_matrix.columns:
        ideal_best[criterion] = weighted_normalized_matrix[criterion].max()
        ideal_worst[criterion] = weighted_normalized_matrix[criterion].min()
    return ideal_best, ideal_worst


def calculate_euclidean_distance(
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
    # print(ranks)
    # print("\nRanked Alternatives:")
    # print(ranked_alternatives)
    return ranked_alternatives, ranks


def TOPSIS_main_data_processing(
        normalized_weights,
        normalized_matrix):
    weighted_normalized_matrix = apply_weights(
        normalized_matrix, normalized_weights)
    ideal_best, ideal_worst = determine_ideal_best_and_worst(
        weighted_normalized_matrix)
    D_plus, D_minus = calculate_euclidean_distance(
        weighted_normalized_matrix, ideal_best, ideal_worst)
    S = calculate_performance_score(D_plus, D_minus)
    ranked_alternatives, ranks = ranking(S)
    return ranked_alternatives, ranks, weighted_normalized_matrix, S, ideal_best, ideal_worst
