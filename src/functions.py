# =======================================================================
# FUNCTIONS
# Define reusable helper functions in this file.
# =======================================================================

#------------------------
#Helper functions
#------------------------
import sys
import numpy as np
#------------------------
#For economic calculations
#------------------------

from src.constants import (Cost_Glucose, Cost_Glutamine, Cost_VCD, Value_mAB, Downtime_Days)

#Calc total cell culture cost per litre
#[$/L]
def calculate_cell_culture_cost(initial_glucose, initial_glutamine, initial_vcd):
    cost = (Cost_Glucose*initial_glucose)+(Cost_Glutamine*initial_glutamine)+(Cost_VCD*initial_vcd)
    return cost

#Calc value mAB produced per litre
#[$/L]
def calculate_culture_value(mAB_concentration):
    value = mAB_concentration*Value_mAB
    return value

#Calc profit
#[$/L]
def calculate_profit(mAB_concentration, initial_glucose, initial_glutamine, initial_vcd):
    value=calculate_culture_value(mAB_concentration)
    cost=calculate_cell_culture_cost(initial_glucose, initial_glutamine, initial_vcd)
    profit = value-cost
    return profit

#Calc spacetime profit
#[$/L⋅d]    Note: Harvest Time in Days
def calculate_space_time_profit(profit, harvest_time_hours):
    harvest_days = harvest_time_hours/24.0
    total_time = harvest_days+Downtime_Days
    return profit/total_time

#Calc spacetime yield
#[mg/L⋅d]    Note: Harvest Time in Days
def calculate_space_time_yield(mAB_concentration, harvest_time_hours):
    harvest_days = harvest_time_hours/24.0
    total_time = harvest_days+Downtime_Days
    return mAB_concentration/total_time

#------------------------
#time-series modelling (Unit 5 Notes)
#------------------------
#progress bar
def progress_bar(c_iter, n_iters, n_updates=1000, length_bar=20):
    update_every = max(1, round(n_iters / n_updates))
    if c_iter % update_every == 0:
        progress = c_iter / n_iters
        length_fill = round(length_bar * progress)
        length_void = length_bar - length_fill
        sys.stdout.write(f"\r{progress:6.0%} [" + length_fill * "=" + length_void * " " + "]")


#x lagged vector autoregressive curve
def generate_x_lag_cur_var(p, batch_ids, df, features, col_batch="Batch ID", col_time="Time [h]"):
    #Generate lagged vector autoregressive features and corresponding target values.
    x_lag = []
    x_cur = []
    t_cur = []
    b_cur = []
    for batch_id in batch_ids:
        mask = df.loc[:, col_batch] == batch_id
        x_batch = df.loc[mask, features].to_numpy()
        t_batch = df.loc[mask, col_time].to_numpy()
        b_batch = df.loc[mask, col_batch].to_numpy()
        current_values = x_batch[p:]
        current_times = t_batch[p:]
        current_batches = b_batch[p:]
        lagged_values = []
        for i in range(1, p + 1):
            lagged_values.append(x_batch[p - i:-i])
        lagged_values = np.concatenate(lagged_values, axis=1)
        x_lag.append(lagged_values)
        x_cur.append(current_values)
        t_cur.append(current_times)
        b_cur.append(current_batches)
    x_lag = np.concatenate(x_lag, axis=0)
    x_cur = np.concatenate(x_cur, axis=0)
    t_cur = np.concatenate(t_cur, axis=0)
    b_cur = np.concatenate(b_cur, axis=0)
    return x_lag, x_cur, t_cur, b_cur
