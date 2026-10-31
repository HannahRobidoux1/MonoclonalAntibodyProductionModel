# =======================================================================
# CLASSES
# Define custom classes in this file.
# =======================================================================
import numpy as np

#--------
#Unit05 StandardScalarVAR
#--------
# Standard scaler designed for vector autoregressive (VAR) models.
class StandardScalerVAR:
    def __init__(self, p):
        self.p = p
        self.mean_ = None
        self.scale_ = None
        self.var_ = None
        self.n_samples_seen_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None

    def fit(self, x, y=None):
        n_var_features = x.shape[1] // self.p
        assert n_var_features * self.p == x.shape[1]
        # Separate the lag blocks.
        x_sf = []
        for i in range(self.p):
            idx_start = i * n_var_features
            idx_end = idx_start + n_var_features
            x_sf.append(x[:, idx_start:idx_end])
            x_sf = np.concatenate(x_sf, axis=0)
            x_sf_mean = np.mean(x_sf, axis=0)
            x_sf_sd = np.std(x_sf, axis=0)
            x_sf_var = np.var(x_sf, axis=0)
            self.mean_ = np.tile(x_sf_mean, (1, self.p))
            self.scale_ = np.tile(x_sf_sd, (1, self.p))
            self.var_ = np.tile(x_sf_var, (1, self.p))
            self.n_samples_seen_ = x.shape[0]
            self.n_features_in_ = x.shape[1]
            return self
    def transform(self, x):
        x_tr = (x - self.mean_) / self.scale_
        return x_tr
    def inverse_transform(self, x):
        x_original = self.scale_ * x + self.mean_
        return x_original
    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x)
    def get_params(self, deep=True):
        return {"p": self.p}
    def set_params(self, **params):
        if "p" in params:
            self.p = params["p"]
        return self


#for representing a single mAB production batch:
class Batch:
    def __init__(self,batch_id, initial_conditions, time_series_df):
        # batch id
        self.batch_id = batch_id
        #initial op conditions
        self.initial_glucose=initial_conditions["Glucose [mM]"]
        self.initial_glutamine = initial_conditions["Glutamine [mM]"]
        self.initial_vcd =initial_conditions["VCD [cells/mL]"]

# Time series data (from individual batch file)
        self.time=time_series_df["Time [h]"].to_numpy()
        self.glucose = time_series_df["Glucose [mM]"].to_numpy()
        self.glutamine = time_series_df["Glutamine [mM]"].to_numpy()
        self.vcd = time_series_df["VCD [cells/mL]"].to_numpy()
        self.lactate=time_series_df["Lactate [mM]"].to_numpy()
        self.ammonia= time_series_df["Ammonia [mM]"].to_numpy()
        self.tcd = time_series_df["TCD [cells/mL]"].to_numpy()
        self.mab =time_series_df["mAb [mg/L]"].to_numpy()

    #combine measurements into 2D array
        def get_feature_matrix(self):
            return np.column_stack([self.glucose, self.glutamine, self.lactate, self.vcd, self.ammonia, self.tcd, self.mab])

