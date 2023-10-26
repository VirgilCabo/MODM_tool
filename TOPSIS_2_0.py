import sensitivity_analysis_2_0 as sens
from TOPSIS_main_data_processing import main_data_processing
from plotting_results_function import results_visualization
from gathering_data_function import gathering_data
from saving_results_function import prompt_to_save_results


decision_matrix, data_filename, weights, normalized_weights, beneficial_criteria, non_beneficial_criteria = gathering_data(
    'C:/Users/Virgi/OneDrive/Bureau/MODM_tool_project/TOPSIS/data_input/mock4_data.csv')

ranked_alternatives, ranks, weighted_normalized_matrix, S = main_data_processing(
    decision_matrix, normalized_weights, beneficial_criteria)

user_input, directory = prompt_to_save_results(decision_matrix,
        weighted_normalized_matrix,
        ranked_alternatives,
        weights,
        beneficial_criteria,
        non_beneficial_criteria,
        data_filename)

results_visualization(
    ranked_alternatives,
    weighted_normalized_matrix,
    beneficial_criteria,
    weights,
    normalized_weights,
    S,
    user_input,
    directory,
    decision_matrix)

normalized_weight_sets, num_sets = sens.generate_weight_sets(weights, 100000, 1000, 0, 10)

sens_scores_results, sens_ranks_results = sens.run_sensitivity_analysis(decision_matrix, normalized_weight_sets, beneficial_criteria)

reliability_percentage = sens.assess_reliability(S, num_sets, sens_ranks_results)

print(sens_scores_results)
print(sens_ranks_results)
print(reliability_percentage)
sens.visualize_sensitivity_results(sens_scores_results)