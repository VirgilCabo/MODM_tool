import pandas as pd
import os
import datetime
import sys


def save_run_results(
        directory,
        decision_matrix,
        preference_functions,
        net_flows,
        weights,
        beneficial_criteria,
        non_beneficial_criteria,
):

    # Save decision matrix
    decision_matrix.to_csv(os.path.join(directory, "decision_matrix.csv"))

    preference_functions_series = pd.Series(preference_functions)
    preference_functions_series.to_csv(os.path.join(
        directory, "preference_functions.csv"))

    # Save scores
    net_flows.to_csv(os.path.join(directory, "net_flows.csv"))

    # Save weights
    weights_series = pd.Series(weights)
    weights_series.to_csv(os.path.join(directory, "weights.csv"))

    # Save parameters (for demonstration, saving beneficial criteria list)
    with open(os.path.join(directory, "parameters.txt"), "w") as f:
        f.write("Beneficial Criteria:\n")
        for criterion in beneficial_criteria:
            f.write(f"{criterion}\n")
        f.write("\n\nNon-beneficial Criteria:\n")
        for criterion in non_beneficial_criteria:
            f.write(f"{criterion}\n")

    print(f"Run results saved in {directory}")
    return


def save_sensitivity_results(
        directory,
        uncertainties,
        net_flows_df,
        ranks_df,
        top_serie):
    uncertainties_series = pd.Series(uncertainties)
    uncertainties_series.to_csv(
        os.path.join(
            directory,
            "sensitivity_uncertainties.csv"))

    net_flows_df.to_csv(os.path.join(directory, "sensitivity_net_flows.csv"))

    ranks_df.to_csv(os.path.join(directory, "sensitivity_ranks.csv"))

    top_serie.to_csv(os.path.join(directory, "reliability_percentages.csv"))

    print(f"Sensitivity analysis results saved in {directory}")


def directory_creation(
        data_filename,
        path="C:/Users/Virgi/OneDrive/Bureau/MODM_tool_project/Tool/Data/results/PROMETHEE"):
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    # Create a directory with the current date and time, script name and data
    # filename
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    directory_name = f"{script_name}_{data_filename}_{current_time}"
    directory = os.path.join(path, directory_name)
    os.makedirs(directory, exist_ok=True)
    return directory
