from MODM_Tool_Modules import gathering_data_function as gt
from MODM_Tool_Modules.PROMETHEE_Modules import PROMETHEE_data_processing as pm_process
from MODM_Tool_Modules.PROMETHEE_Modules import PROMETHEE_plotting_results_function as pm_plot
from MODM_Tool_Modules.PROMETHEE_Modules import PROMETHEE_saving_results as pm_save
from MODM_Tool_Modules.PROMETHEE_Modules import PROMETHEE_sensitivity_analysis as pm_sens


decision_matrix, data_filename, weights, normalized_weights, beneficial_criteria, non_beneficial_criteria = gt.gathering_data(
    'C:/Users/Virgi/OneDrive/Bureau/MODM_tool_project/Tool/Data/data_input/optimal_pareto_points.csv')

user_input = input(
    "Do you want to save the results of this run? (yes/no): ").strip().lower()
directory = None
if user_input == 'yes':
    directory = pm_save.directory_creation(data_filename)

preference_functions = pm_process.define_preference_functions(decision_matrix)
net_flows, ranked_alternatives, ranks = pm_process.PROMETHEE_data_processing(
    decision_matrix, beneficial_criteria, normalized_weights, preference_functions)

pm_plot.results_visualization_promethee(
    net_flows,
    weights,
    user_input,
    directory,
    ranked_alternatives)

user_input2 = input(
    "Do you want to run a sensitivity analysis? (yes/no): ").strip().lower()
if user_input2 == 'yes':
    uncertainties, net_flows_df, ranks_df, reliability_percentage, initial_best_solution = pm_sens.sensitivity_analysis(
        pm_process.PROMETHEE_data_processing, weights, 10000, 0, 10, decision_matrix, beneficial_criteria, net_flows, user_input, directory, preference_functions)

if user_input == 'yes':
    pm_save.save_run_results(
        directory,
        decision_matrix,
        preference_functions,
        ranked_alternatives,
        weights,
        beneficial_criteria,
        non_beneficial_criteria,
    )

if user_input2 == 'yes' and user_input == 'yes':
    pm_save.save_sensitivity_results(
        directory,
        uncertainties,
        net_flows_df,
        ranks_df,
        reliability_percentage,
        initial_best_solution)
