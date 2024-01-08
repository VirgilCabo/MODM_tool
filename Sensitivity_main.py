from MODM_Tool_Modules import gathering_data_function as gt
from MODM_Tool_Modules.TOPSIS_Modules import TOPSIS_main_data_processing as tp_process
from MODM_Tool_Modules.TOPSIS_Modules import sensitivity_analysis_TOPSIS as tp_sens
from MODM_Tool_Modules.PROMETHEE_Modules import PROMETHEE_data_processing as pm_process
from MODM_Tool_Modules.PROMETHEE_Modules import PROMETHEE_sensitivity_analysis as pm_sens
import pandas as pd
import numpy as np
import os
import sys
import datetime


def lins_ccc(x, y):
    cov_mat = np.cov(x, y)
    cov_xy = cov_mat[0, 1]
    var_x = cov_mat[0, 0]
    var_y = cov_mat[1, 1]
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    ccc = (2 * cov_xy) / (var_x + var_y + (mean_x - mean_y)**2)
    return ccc


def directory_creation(
        data_filename,
        path="C:/Users/Virgi/OneDrive/Bureau/MODM_tool_project/Tool/Data/results/Sensitivity_Analysis"):
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    directory_name = f"{script_name}_{data_filename}_{current_time}"
    directory = os.path.join(path, directory_name)
    os.makedirs(directory, exist_ok=True)
    return directory


decision_matrix, normalized_matrix, data_filename, weights, normalized_weights, beneficial_criteria, non_beneficial_criteria = gt.gathering_data(
    'C:/Users/Virgi/OneDrive/Bureau/MODM_tool_project/Tool/Data/data_input/optimal_pareto_points2.csv')

user_input = input(
    "Do you want to save the results of this run? (yes/no): ").strip().lower()
directory = None
if user_input == 'yes':
    directory = directory_creation(data_filename)

preference_functions = pm_process.define_preference_functions(decision_matrix)

normalized_weight_sets, num_sets, uncertainties = tp_sens.generate_weight_sets(
        weights, 100000, 0, 10)

scores_df, tp_ranks_df, tp_filtered_top_serie, tp_top_serie, tp_filtered_top3_serie, tp_top3_serie = tp_sens.sensitivity_analysis(
        tp_process.TOPSIS_main_data_processing, normalized_weight_sets, decision_matrix, normalized_matrix, user_input, directory)

net_flows_df, pm_ranks_df, pm_filtered_top_serie, pm_top_serie, pm_filtered_top3_serie, pm_top3_serie = pm_sens.sensitivity_analysis(
        pm_process.PROMETHEE_data_processing, normalized_weight_sets, decision_matrix, normalized_matrix, user_input, directory, preference_functions)

ccc_tot = lins_ccc(tp_top_serie, pm_top_serie)
ccc_top5 = lins_ccc(tp_filtered_top_serie, pm_filtered_top_serie)

if user_input =='yes':
    tp_top_serie.to_csv(os.path.join(directory, "TOPSIS_top1_%.csv"))
    pm_top_serie.to_csv(os.path.join(directory, "PROMETHEE_top1_%.csv"))
    tp_top3_serie.to_csv(os.path.join(directory, "TOPSIS_top3_%.csv"))
    pm_top3_serie.to_csv(os.path.join(directory, "PROMETHEE_top3_%.csv"))
    weights_serie = pd.Series(weights)
    weights_serie.to_csv(os.path.join(directory, "weights.csv"))
    uncertainties_serie = pd.Series(uncertainties)
    uncertainties_serie.to_csv(os.path.join(directory, "uncertainties.csv"))
    scores_df.to_csv(os.path.join(directory, "TOPSIS_scores.csv"))
    net_flows_df.to_csv(os.path.join(directory, "PROMETHEE_netflows.csv"))

    with open(os.path.join(directory, "lins_cccs.txt"), "w") as f:
        f.write(f"Lin's Concordance Correlation Coefficient for all Alternatives: {ccc_tot}\n")
        f.write(f"Lin's Concordance Correlation Coefficient for Top 5 Alternatives: {ccc_top5}")

print('\nTotal')
print(ccc_tot)
print('\nTop')
print(ccc_top5)

print('\nTOPSIS Top 1')
print(tp_filtered_top_serie)
print('\nPROMETHEE Top 1')
print(pm_filtered_top_serie)
print('\nTOPSIS Top 3')
print(tp_filtered_top3_serie)
print('\nPROMETHEE Top 3')
print(pm_filtered_top3_serie)