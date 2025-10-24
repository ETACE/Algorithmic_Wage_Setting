import Globals, os
from Firm import Firms
from Worker import Workers
from Space import Space
import Neural_Network
import csv
import traceback

def write_firm_to_csv(iteration, firm_list, file_name="firms_output.csv", append=True):
    """
    Write all firm variables to a CSV file.

    Parameters:
        iteration (int): Current iteration or tick of the simulation.
        firm_list (list): List of Firms objects.
        file_name (str): The name of the output CSV file. Defaults to "firms_output.csv".
        append (bool): Whether to append to an existing file. Defaults to True.
    """
    # Define the headers based on the attributes of the first firm
    headers = [
        "iteration", "firm_id", "position", "wage_offer", "greedy_wage",
        "productivity", "profits", "filled_jobs","av_wage_competitors","av_greedy_wage_for_analysis"
    ]

    # Check if the file already exists
    write_header = not append or not os.path.exists(file_name)

    # Open the file in append or write mode
    with open(file_name, mode="a" if append else "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        # Write the header if the file is new or not appending
        if write_header:
            writer.writerow(headers)

        # Write the firm data for each firm in the list
        for firm in firm_list:
            writer.writerow([
                iteration,
                firm.firm_id,
                firm.position,
                firm.wage_offer,
                firm.greedy_wage,
                firm.productivity,
                firm.profits,
                firm.filled_jobs,
                firm.av_wage_competitors,
                firm.av_greedy_wage_for_analysis

            ])

    #print(f"Firm data written to {file_name}")



import csv

def write_worker_aggregates_to_csv(iteration, worker_list, file_name="worker_aggregates.csv", append=True):
    """
    Write aggregated worker data to a CSV file, including metrics like unemployment rate.

    Parameters:
        iteration (int): Current iteration or tick of the simulation.
        worker_list (list): List of Workers objects.
        file_name (str): The name of the output CSV file. Defaults to "worker_aggregates.csv".
        append (bool): Whether to append to an existing file. Defaults to True.
    """
    # Calculate metrics
    total_workers = len(worker_list)
    employed_workers = sum(1 for worker in worker_list if worker.employment_status == 1)
    unemployment_rate = (1 - (employed_workers / total_workers)) * 100 if total_workers > 0 else 0

    # Define the headers
    headers = ["iteration", "total_workers", "employed_workers", "unemployment_rate"]

    # Check if the file already exists
    write_header = not append or not os.path.exists(file_name)

    # Open the file in append or write mode
    with open(file_name, mode="a" if append else "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        # Write the header if the file is new or not appending
        if write_header:
            writer.writerow(headers)

        # Write the aggregated data
        writer.writerow([
            iteration,
            total_workers,
            employed_workers,
            unemployment_rate
        ])

    #print(f"Worker aggregate data written to {file_name}")


import csv
import os
import numpy as np

def save_q_values_over_time(iteration, firm_list, file_name="q_values_over_time.csv", append=True):
    """
    Save Q-values for each candidate wage over iterations to a CSV file.

    Args:
        iteration (int): Current simulation iteration.
        firm_list (list): List of Firm objects.
        file_name (str): Name of the output CSV file.
        append (bool): Whether to append to an existing file or overwrite it.
    """

    file_exists = os.path.isfile(file_name)

    with open(file_name, mode="a" if append else "w", newline="") as file:
        writer = csv.writer(file)

        # Write header if file is new
        if not file_exists or append is False:
            writer.writerow(["Iteration", "Firm_ID", "Candidate_Wage", "Q_Value"])

        # Collect data for each firm
        for firm in firm_list:
            q_values = firm.policy_net(firm.current_state).numpy().ravel()  # Get Q-values from policy network
            for idx, q_value in enumerate(q_values):
                candidate_wage = firm.wage_list[idx]
                writer.writerow([
                    iteration,
                    firm.firm_id,
                    candidate_wage,
                    q_value
                ])

def save_firm_performance(iteration, firm_list, file_name="firm_performance.csv", append=True):
    """
    Save firm performance data (wage offer and profit) over iterations to a CSV file.

    Args:
        iteration (int): Current simulation iteration.
        firm_list (list): List of Firm objects.
        file_name (str): Name of the output CSV file.
        append (bool): Whether to append to an existing file or overwrite it.
    """

    file_exists = os.path.isfile(file_name)

    with open(file_name, mode="a" if append else "w", newline="") as file:
        writer = csv.writer(file)

        # Write header if file is new
        if not file_exists or append is False:
            writer.writerow(["Iteration", "Firm_ID", "Wage_Offer", "Greedy_Wage","Profit"])

        # Collect data for each firm
        for firm in firm_list:
            writer.writerow([
                iteration,
                firm.firm_id,
                firm.wage_offer,
                firm.greedy_wage,
                firm.profits
            ])





def save_policy_models(firms, iteration="final", save_format="weights"):
    """
    Saves each firm's policy network into <current_output_dir>/checkpoints/.
    save_format: 'weights' -> .weights.h5 (fast); 'keras' -> .keras (full model)
    """
  

    for firm in firms:
        model = getattr(firm, "policy_net", None)
        if model is None:
            continue
        try:
            if save_format == "weights":
                out = f"firm{firm.firm_id}_iter{iteration}.weights.h5"
                model.save_weights(out)
            else:  # 'keras'
                out = f"firm{firm.firm_id}_iter{iteration}.keras"
                model.save(out)
        except Exception:
            print(f"[WARN] Failed to save policy for firm {getattr(firm, 'firm_id', '?')} at iter {iteration}")
            traceback.print_exc()
