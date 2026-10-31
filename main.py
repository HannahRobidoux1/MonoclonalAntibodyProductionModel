# =====================================================================
# MAIN PROGRAM
# Your code starts here. This is the file that will be run.
# =====================================================================
# Note for User: n_simulations is set to 1000 for final results in constants.py.
# Reduce this value (e.g., to 50 or 100) if you want a faster execution time.
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import src.functions as sf
from src.classes import Batch  # This allows you to import individual Classes from classes.py
from src.constants import *  # This imports all constants from constants.py
from src.functions import generate_x_lag_cur_var
from src.functions import progress_bar
from src.classes import StandardScalerVAR

#_________
#LOAD DATA
#_________
#Load initial conditions for all the batches
initial_conditions=pd.read_csv("datasets/initial_conditions.csv")
#Find batch files
batch_files = sorted([file for file in os.listdir("datasets") if file.startswith("batch_")and file.endswith(".csv")])
#dictionary to store batch objects
batches={}
for file in batch_files:
    #extract batch ID from the filename
    batch_id=int(file.replace("batch_","").replace(".csv", ""))
    #need initial condition for this batch
    initial_row = initial_conditions[initial_conditions["Batch ID"] == batch_id].iloc[0]
    #Get time-series data
    batch_data = pd.read_csv(os.path.join("datasets", file))
    batches[batch_id] = Batch(batch_id, initial_row, batch_data)
print("Monoclonal Antibody Production")
print("-----------------------------")
print("Number of batches loaded:", len(batches))

#_________
#VECTOR AUTOREGRESSION (VAR)
#_________
#(im using column names here)
features=["Glucose [mM]","Glutamine [mM]","Lactate [mM]","Ammonia [mM]","VCD [cells/mL]","TCD [cells/mL]","mAb [mg/L]"]
data=[]
for batch_id, batch in batches.items():
    for i in range(len(batch.time)):
        row={"Batch ID":batch_id,"Time [h]":batch.time[i],"Glucose [mM]":batch.glucose[i],"Glutamine [mM]":batch.glutamine[i],"Lactate [mM]":batch.lactate[i],"Ammonia [mM]":batch.ammonia[i],"VCD [cells/mL]":batch.vcd[i],"TCD [cells/mL]":batch.tcd[i],"mAb [mg/L]":batch.mab[i]}
        data.append(row)
df=pd.DataFrame(data)
print("Total time-series measurements:", len(df))
print("Features used in VAR:", len(features))
#I use Unit05 VAR as a reference here
#(from constants.py)
n_simulations = N_Simulations
prediction_interval = Prediction_Interval
p_max =3
alpha=1-prediction_interval
#from constants
rng=np.random.default_rng(Random_seed)
# Extract the unique batch IDs
batch_ids=np.unique(df["Batch ID"])
# Select the variable that will be plotted and evaluated (in this case mAB)
feature_plot="mAb [mg/L]"
idx_fp=features.index(feature_plot)
#Dynamic grid sizing (5 rows, 6 cols for 30 batches)
nrows, ncols = 5,6
n_iters=p_max*batch_ids.shape[0]*n_simulations
c_iter=0
records=[]

os.makedirs("figures", exist_ok=True)
os.makedirs("tables", exist_ok=True)

for p in range(1, p_max + 1):
    fig,axes=plt.subplots(nrows,ncols,figsize=(18,12),layout="constrained",dpi=300)
    rmse_forecasts=[]
    y_min = 0
    y_max = np.max(df.loc[:, feature_plot])
    # Leave-one-batch-out cross-validation
    for batch_id_test in batch_ids:
        #training batches have all batches except test batch
        batch_ids_train=np.array([i for i in batch_ids if i!=batch_id_test])
        # Generate the lagged value dataset used to train VAR(p)
        x_lag_train, x_cur_train, _, _ = generate_x_lag_cur_var(p=p, batch_ids=batch_ids_train, df=df, features=features)
        # this is for finding the index of the test batch for plotting in the subplot grid
        k = np.where(batch_ids == batch_id_test)[0][0]

        # Create and train the model
        model = make_pipeline(StandardScalerVAR(p=p), LinearRegression())
        model.fit(x_lag_train, x_cur_train)
        #residual errors from training batches
        x_cur_train_pred=model.predict(x_lag_train)
        x_cur_train_res=x_cur_train_pred-x_cur_train
        #mask so pandas only grabs true values (only the batch im evaluating)
        mask= (df["Batch ID"] == batch_id_test)
        batch_test=df.loc[mask].sort_values("Time [h]")
        t_test = batch_test["Time [h]"].to_numpy()
        x_test = batch_test[features].to_numpy()
        #use first measured state as initial and dont want to mess up x_intial so stores copy
        x_initial=x_test[0].copy()
        # Perform multiple forecast simulations to estimate uncertainty
        x_forecasts=[]
        for simulation_id in range(n_simulations):
            x_forecast=[x_initial]
            while len(x_forecast)<len(t_test):
                x_lag=x_forecast[::-1][:p].copy()

                while len(x_lag)<p:
                    x_lag.append(x_lag[-1])
                # Concatenate all lagged vectors into a single feature vector expected by the VAR model.
                x_lag = np.concatenate(x_lag, axis=0)
                x_lag = np.expand_dims(x_lag, axis=0)

                #assume residuals behave like white noise
                white_noise=[]
                for j in range(x_cur_train_res.shape[1]):white_noise.append(rng.choice(x_cur_train_res[:, j]))
                x_cur = model.predict(x_lag)[0] + white_noise
                x_cur = np.clip(x_cur, a_min=0, a_max=None)
                x_forecast.append(x_cur)
            x_forecast = np.array(x_forecast)
            x_forecasts.append(x_forecast)
            c_iter += 1
            progress_bar(c_iter, n_iters)
        x_forecasts = np.array(x_forecasts)

        # Median forecast and 95% prediction interval
        x_forecast_pi_lower = np.quantile(x_forecasts, q=alpha / 2, axis=0)
        x_forecast_median = np.quantile(x_forecasts, q=0.5, axis=0)
        x_forecast_pi_higher = np.quantile(x_forecasts, q=1-alpha / 2, axis=0)
        # Calculate RMSE for the selected plotting variable
        rmse_forecast = root_mean_squared_error(x_test[:, idx_fp], x_forecast_median[:, idx_fp])
        rmse_forecasts.append(rmse_forecast)
        #PLOT!
        ax = axes[k // ncols, k % ncols]
        ax.scatter(t_test, x_test[:, idx_fp],s=10, color="tab:red", edgecolors="black", label="Measured")
        ax.plot(t_test, x_forecast_median[:, idx_fp],color="black", label=f"VAR({p})")
        ax.fill_between(t_test, x_forecast_pi_lower[:, idx_fp], x_forecast_pi_higher[:, idx_fp],label=f"{prediction_interval:.0%} PI", color="tab:cyan", edgecolors="none", alpha=0.3, zorder=2)
        ax.set_title(f"Batch {batch_id_test} | RMSE: {rmse_forecast:.3f}")
        ax.set_xlabel("Time [h]")
        ax.set_ylabel("mAb [mg/L]")
        # Track the maximum value needed for consistent subplot limits
        y_max = max(y_max, np.max(x_test[:, idx_fp]), np.max(x_forecast_pi_higher[:, idx_fp]))
        if k == 0:
            ax.legend(fontsize=7)
    fig.savefig(f"figures/VAR_p{p}.png", dpi=300)
    plt.close(fig)
    records.append((p, n_simulations,np.mean(rmse_forecasts),np.std(rmse_forecasts)))

# Export comparison of lag orders
df_results = pd.DataFrame(records,columns=["p", "N simulations", "RMSE mean", "RMSE SD"])
df_results.to_csv("tables/VAR_results.csv", index=False)
print("---------------------")
print("VAR model comparison:")
print(df_results.to_string(index=False))

#_________
#NONLINEAR VECTOR AUTOREGRESSION (NVAR)
#_________
#(from constants.py)
n_simulations = N_Simulations
prediction_interval = Prediction_Interval
alpha=1-prediction_interval
# Define training parameters
p_max = 3
cv_splits = 3
poly_degree_max = 3
knn_k_max = 5
model_types = ["poly", "knn"]

param_grid_poly = dict(polynomialfeatures__degree=[i for i in range(2, poly_degree_max + 1)])
param_grid_knn = dict(kneighborsregressor__n_neighbors=[i for i in range(2, knn_k_max + 1)],kneighborsregressor__weights=["uniform"])


#(from constants) random number generator
rng = np.random.default_rng(Random_seed)
# Extract the unique batch IDs
batch_ids=np.unique(df["Batch ID"])
# Select the variable that will be plotted and evaluated (in this case mAB)
feature_plot="mAb [mg/L]"
idx_fp=features.index(feature_plot)
#Dynamic grid sizing (5 rows, 6 cols for 30 batches)
nrows, ncols = 5,6
n_iters=len(model_types) * p_max * len(batch_ids) * n_simulations
c_iter=0
records=[]
forecast_results={} #save the forecasts for economics

#for figures:
os.makedirs("figures", exist_ok=True)
os.makedirs("tables", exist_ok=True)

for model_type in model_types:
    for p in range(1, p_max + 1):
        fig,axes=plt.subplots(nrows,ncols,figsize=(18,12),layout="constrained",dpi=300)
        rmse_forecasts=[]
        y_max = np.max(df[feature_plot])

        # Leave-one-batch-out cross-validation
        for batch_id_test in batch_ids:
            #training batches have all batches except test batch
            batch_ids_train=np.array([i for i in batch_ids if i!=batch_id_test])
            # Generate the lagged training data
            x_lag_train, x_cur_train, _, b_cur_train = generate_x_lag_cur_var(p=p, batch_ids=batch_ids_train, df=df, features=features)
            #added for different scales of process variables
            a_min = np.min(x_cur_train, axis=0)
            a_max = np.max(x_cur_train,axis=0)

            # Create and train the model
            if model_type == "poly":
                model = make_pipeline(StandardScalerVAR(p=p), PolynomialFeatures(include_bias=False), LinearRegression())
                param_grid=param_grid_poly
            elif model_type == "knn":
                model = make_pipeline(StandardScalerVAR(p=p),KNeighborsRegressor())
                param_grid = param_grid_knn
            # Keep the batches separate in GridSearchCV
            gkf = GroupKFold(n_splits=cv_splits)
            model_gscv = GridSearchCV(model, param_grid=param_grid, cv=gkf, scoring="neg_root_mean_squared_error",return_train_score=True)
            model_gscv.fit(x_lag_train,x_cur_train,groups=b_cur_train)
            #residual errors from training batches
            x_cur_train_pred = model_gscv.predict(x_lag_train)
            x_cur_train_res = x_cur_train_pred - x_cur_train
            #mask so pandas only grabs true values (only the batch im evaluating)
            mask= (df["Batch ID"] == batch_id_test)
            batch_test=df.loc[mask].sort_values("Time [h]")
            t_test = batch_test["Time [h]"].to_numpy()
            x_test = batch_test[features].to_numpy()
            #use first measured state as initial and dont want to mess up x_intial so stores copy
            x_initial=x_test[0].copy()
            # Find the subplot for this batch
            k = np.where(batch_ids == batch_id_test)[0][0]
            ax = axes[k // ncols, k % ncols]
            # Perform multiple forecast simulations to estimate uncertainty
            x_forecasts=[]
            for simulation_id in range(n_simulations):
                x_forecast=[x_initial]
                while len(x_forecast)<len(t_test):
                    x_lag=x_forecast[::-1][:p].copy()

                    while len(x_lag)<p:
                        x_lag.append(x_lag[-1])
                    # Concatenate all lagged vectors into a single feature vector expected by the VAR model.
                    x_lag = np.concatenate(x_lag, axis=0)
                    x_lag = np.expand_dims(x_lag, axis=0)

                    #assume residuals behave like white noise
                    white_noise=[]
                    for j in range(x_cur_train_res.shape[1]):white_noise.append(rng.choice(x_cur_train_res[:, j]))
                    white_noise=np.array([white_noise])
                    # Predict the next process state
                    x_cur_baseline = model_gscv.predict(x_lag)
                    # Keep baseline predictions within the training-data range
                    x_cur_baseline = np.clip(x_cur_baseline,a_min=a_min,a_max=a_max)
                    # Add sampled residual noise
                    x_cur = x_cur_baseline + white_noise
                    # Prevent negative concentrations
                    x_cur = np.clip(x_cur[0], a_min=0, a_max=None)
                    x_forecast.append(x_cur)
                x_forecast = np.array(x_forecast)
                x_forecasts.append(x_forecast)
                c_iter += 1
                progress_bar(c_iter, n_iters)
            x_forecasts = np.array(x_forecasts)

            # Median forecast and 95% prediction interval
            x_forecast_pi_lower = np.quantile(x_forecasts, q=alpha / 2, axis=0)
            x_forecast_median = np.quantile(x_forecasts, q=0.5, axis=0)
            x_forecast_pi_higher = np.quantile(x_forecasts, q=1-alpha / 2, axis=0)
            # Save the median forecast for each model, lag order, and batch
            forecast_results[(model_type, p, batch_id_test)] = (t_test.copy(), x_forecast_median.copy())
            # Calculate RMSE for the selected plotting variable
            rmse_forecast = root_mean_squared_error(x_test[:, idx_fp], x_forecast_median[:, idx_fp])
            rmse_forecasts.append(rmse_forecast)
            #PLOT!
            ax = axes[k // ncols, k % ncols]
            ax.scatter(t_test, x_test[:, idx_fp],s=10, color="tab:pink", edgecolors="black", label="Measured")
            ax.plot(t_test, x_forecast_median[:, idx_fp],color="black", label=f"NVAR-{model_type}({p})")
            ax.fill_between(t_test, x_forecast_pi_lower[:, idx_fp], x_forecast_pi_higher[:, idx_fp],label=f"{prediction_interval:.0%} PI", color="tab:green", edgecolors="none", alpha=0.3, zorder=2)
            ax.set_title(f"Batch {batch_id_test} | RMSE: {rmse_forecast:.3f}")
            ax.set_xlabel("Time [h]")
            ax.set_ylabel("mAb [mg/L]")
            # Track the maximum value needed for consistent subplot limits
            y_max = max(y_max, np.max(x_test[:, idx_fp]), np.max(x_forecast_pi_higher[:, idx_fp]))
            if k == 0:
                ax.legend(fontsize=7)
        fig.savefig(f"figures/NVAR_{model_type}_p{p}.png", dpi=300)
        plt.close(fig)
        records.append((model_type,p, n_simulations,np.mean(rmse_forecasts),np.std(rmse_forecasts)))

# Export comparison of lag orders
df_results = pd.DataFrame(records,columns=["Model","p", "N simulations", "RMSE mean", "RMSE SD"])
df_results.to_csv("tables/NVAR_results.csv", index=False)
print("---------------------")
print("NVAR model comparison:")
print(df_results.to_string(index=False))

#_________
#ECONOMICS
#_________
print("---------------------")
print("Economic Optimization")
#selects NVAR model with the lowest mean RMSE
best_model_row=df_results.loc[df_results["RMSE mean"].idxmin()]
best_model_type=best_model_row["Model"]
best_p=int(best_model_row["p"])
print("Best NVAR Model: ",best_model_type)
print("Best Lag Order: ",best_p)
print("Best RMSE: ",best_model_row["RMSE mean"])
#store best economic results for each batch
economic_records=[]
for batch_id in batch_ids:
    t_forecast, x_forecast = forecast_results[(best_model_type, best_p, batch_id)]

    initial_row=initial_conditions[initial_conditions["Batch ID"] == batch_id].iloc[0]
    i_gluc=initial_row["Glucose [mM]"]
    i_glut=initial_row["Glutamine [mM]"]
    i_vcd=initial_row["VCD [cells/mL]"]

    #CALC INITIAL CULTURE COST (FROM FUNCTIONS.PY)
    culture_cost = sf.calculate_cell_culture_cost(i_gluc, i_glut, i_vcd)

    # Calculate space time profit at each harvest time (FUNCTIONS IN FUNCTIONS.PY
    space_time_profits = []
    for i in range(len(t_forecast)):
        harvest_time=t_forecast[i]
        mab_concentration = x_forecast[i, idx_fp]
        profit = sf.calculate_profit(mab_concentration, i_gluc, i_glut, i_vcd)
        space_time_profit = sf.calculate_space_time_profit(profit, harvest_time)
        space_time_profits.append(space_time_profit)

    # Find the harvest time with the highest space-time profit (aka the best)
    best_index=np.argmax(space_time_profits)
    best_harvest_time=t_forecast[best_index]
    best_mab=x_forecast[best_index, idx_fp]
    best_profit=sf.calculate_profit(best_mab, i_gluc, i_glut, i_vcd)

    #RESULTS:
    economic_records.append({"Batch ID": batch_id, "Harvest time [h]": best_harvest_time, "Predicted mAb [mg/L]": best_mab, "Profit [$]": best_profit, "Space time profit [$/L]": space_time_profits[best_index]})

#TABLE RESULTS
df_economics=pd.DataFrame(economic_records)
df_economics.to_csv("tables/NEconomic_results.csv", index=False)
print("---------------------")
print("Economic optimization results:")
print(df_economics.to_string(index=False))



