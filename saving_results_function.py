import pandas as pd
import re
import os
import datetime
import sys


def save_results(
        decision_matrix,
        weighted_normalized_matrix,
        scores,
        weights,
        beneficial_criteria,
        non_beneficial_criteria,
        data_filename,
        path="C:/Users/Virgi/OneDrive/Bureau/MODM_tool_project/TOPSIS/results"):
    global directory
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    # Create a directory with the current date and time, script name and data filename 
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    directory_name = f"{script_name}_{data_filename}_{current_time}"
    directory = os.path.join(path, directory_name)
    os.makedirs(directory, exist_ok=True)

    # Save decision matrix
    decision_matrix.to_csv(os.path.join(directory, "decision_matrix.csv"))

    # Save weighted normalized matrix
    weighted_normalized_matrix.to_csv(os.path.join(
        directory, "weighted_normalized_matrix.csv"))

    # Save scores
    scores.to_csv(os.path.join(directory, "scores.csv"))

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

    print(f"Results saved in {directory}")
    return directory


def prompt_to_save_results(
        decision_matrix,
        weighted_normalized_matrix,
        ranked_alternatives,
        weights,
        beneficial_criteria,
        non_beneficial_criteria,
        data_filename
):
    # Ask the user if they want to save the results
    global user_input
    user_input = input(
        "Do you want to save the results? (yes/no): ").strip().lower()

    # If the user's input is 'yes', call the save_results function
    if user_input == 'yes':
        directory = save_results(
            decision_matrix,
            weighted_normalized_matrix,
            ranked_alternatives,
            weights,
            beneficial_criteria,
            non_beneficial_criteria,
            data_filename)
        print("Results saved successfully!")
    else:
        directory = None
        print("Results not saved.")
    return user_input, directory






