from MODM_Tool_Modules.TOPSIS_Modules import sensitivity_analysis_TOPSIS as sens
from MODM_Tool_Modules import gathering_data_function as gt
from MODM_Tool_Modules.TOPSIS_Modules import TOPSIS_main_data_processing as tp_process
from MODM_Tool_Modules.TOPSIS_Modules import TOPSIS_plotting_results_function as tp_plot
from MODM_Tool_Modules.TOPSIS_Modules import TOPSIS_saving_results_function as tp_save


decision_matrix, normalized_matrix, data_filename, weights, normalized_weights, beneficial_criteria, non_beneficial_criteria = gt.gathering_data(
    'C:/Users/Virgi/OneDrive/Bureau/MODM_tool_project/Tool/Data/data_input/optimal_pareto_points2.csv')

user_input = input(
    "Do you want to save the results of this run? (yes/no): ").strip().lower()
directory = None
if user_input == 'yes':
    directory = tp_save.directory_creation(data_filename)

ranked_alternatives, ranks, weighted_normalized_matrix, S, ideal_best, ideal_worst = tp_process.TOPSIS_main_data_processing(
    normalized_weights, normalized_matrix)

tp_plot.results_visualization_topsis(
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
    ideal_worst)

user_input2 = input(
    "Do you want to run a sensitivity analysis? (yes/no): ").strip().lower()
if user_input2 == 'yes':
    normalized_weight_sets, num_sets, uncertainties = sens.generate_weight_sets(
        weights, 100000, 0, 10)
    scores_df, ranks_df, filtered_top_serie, top_serie, filtered_top3_serie, top3_serie = sens.sensitivity_analysis(
        tp_process.TOPSIS_main_data_processing, normalized_weight_sets, decision_matrix, normalized_matrix, user_input, directory)

if user_input == 'yes':
    tp_save.save_run_results(
        directory,
        decision_matrix,
        weighted_normalized_matrix,
        ranked_alternatives,
        weights,
        beneficial_criteria,
        non_beneficial_criteria,
    )
    
if user_input2 == 'yes' and user_input == 'yes':
    tp_save.save_sensitivity_results(
        directory,
        uncertainties,
        scores_df,
        ranks_df,
        top_serie)
