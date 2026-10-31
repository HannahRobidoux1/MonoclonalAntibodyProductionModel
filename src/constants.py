# =======================================================================
# CONSTANTS
# Define constant values that do not change throughout your program.
# =======================================================================

# Economic Constants
Cost_Glucose=10.0 #Cost per litre for each mM of initial glucose
Cost_Glutamine=20.0 #Cost per litre for each mM of initial glutamine
Cost_VCD=0.0005 #Cost per litre of initial VCD
Value_mAB=1000.0 #Product value per litre for each mg/L of mAB

#The 1 L bioreactors used require 1 day of downtime between batch runs.
Downtime_Days=1.0

#Model settings and optimization
Random_seed=0
# Number of forecast simulations
N_Simulations=1000
# Fraction used to calculate prediction intervals
Prediction_Interval=0.95
