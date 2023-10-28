from sensitivity_analysis_2_0 import sensitivity_analysis
from TOPSIS_main_data_processing import main_data_processing
from plotting_results_function import results_visualization
from gathering_data_function import gathering_data
from saving_results_function import directory_creation, save_run_results, save_sensitivity_results


user_input = input(
    "Do you want to save the results of this run? (yes/no): ").strip().lower()

decision_matrix, data_filename, weights, normalized_weights, beneficial_criteria, non_beneficial_criteria = gathering_data(
    'C:/Users/Virgi/OneDrive/Bureau/MODM_tool_project/TOPSIS/data_input/mock3_data.xlsx')

directory = None

if user_input == 'yes':
    directory = directory_creation(data_filename)

ranked_alternatives, ranks, weighted_normalized_matrix, S = main_data_processing(
    decision_matrix, normalized_weights, beneficial_criteria)

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

user_input2 = input(
    "Do you want to run a sensitivity analysis? (yes/no): ").strip().lower()

if user_input2 == 'yes':
    uncertainties, scores_df, ranks_df, reliability_percentage, initial_best_solution = sensitivity_analysis(
        main_data_processing, weights, 10000, 10000, 0, 10, decision_matrix, beneficial_criteria, S, user_input, directory)

if user_input == 'yes':
    save_run_results(
        directory,
        decision_matrix,
        weighted_normalized_matrix,
        ranked_alternatives,
        weights,
        beneficial_criteria,
        non_beneficial_criteria,
    )
    
if user_input2 == 'yes' and user_input == 'yes':
    save_sensitivity_results(
        directory,
        uncertainties,
        scores_df,
        ranks_df,
        reliability_percentage,
        initial_best_solution)
