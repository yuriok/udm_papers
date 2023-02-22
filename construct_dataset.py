import pickle

import numpy as np
import pandas as pd
from QGrain.models import DistributionType, ArtificialDataset
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt


classes = np.logspace(0, 5, 101) * 0.02
classes_phi = -np.log2(classes / 1000.0)
n_classes = len(classes)

# load data
df = pd.read_excel("./sheets/Construct_Dataset.xlsx", sheet_name=0)
MGS_ages = df.iloc[:, 0].dropna().to_numpy()
MGS_values = df.iloc[:, 1].dropna().to_numpy()
d18O_ages = df.iloc[:, 2].dropna().to_numpy()
d18O_values = df.iloc[:, 3].dropna().to_numpy()

# construct three signal series
n_samples = 500
n_components = 4
ages = np.linspace(4.5, 144.0, n_samples)
series_1 = interp1d(MGS_ages, -MGS_values)(ages)
series_1 = (series_1 - np.min(series_1)) / np.std(series_1)
b, a = butter(8, 0.05, "lowpass")
transformed_signal = filtfilt(b, a, d18O_values)
series_2 = interp1d(d18O_ages, -transformed_signal)(ages)
series_2 = (series_2 - np.min(series_2)) / np.std(series_2)
median = np.median(series_2)
upper_group = series_2[np.greater(series_2, median)]
lower_group = series_2[np.less(series_2, median)]
value_1_4 = np.median(lower_group)
value_3_4 = np.median(upper_group)
series_3 = np.copy(series_2)
series_3[np.less_equal(series_3, median)] = np.min(series_3)

# prepare the parameters
C1_mean = np.random.random(n_samples) * 0.05 + 10.2
C1_std = np.random.random(n_samples) * 0.05 + 0.55
C1_weight = np.random.random(n_samples) * 0.01
C2_mean = series_1 * 0.1 + 7.8 + np.random.random(n_samples) * 0.01
C2_std = np.random.random(n_samples) * 0.04 + 0.8
C2_weight = series_1 * 0.2 + 1.0 + np.random.random(n_samples) * 0.01
C3_mean = series_1 * 0.2 + 5.5 + np.random.random(n_samples) * 0.01
C3_std = np.random.random(n_samples) * 0.04 + 0.7
C3_weight = series_1 * 0.5 + 1.0 + np.random.random(n_samples) * 0.01
C4_mean = series_2 * 0.4 + 2.5 + np.random.random(n_samples) * 0.01
C4_std = np.random.random(n_samples) * 0.1 + 0.6
C4_weight = series_3 + 0.1 + np.random.random(n_samples) * 0.01

# pack parameters
parameters = np.ones((len(ages), 3, 4))
parameters[:, 0, 0] = C1_mean
parameters[:, 1, 0] = C1_std
parameters[:, 2, 0] = C1_weight
parameters[:, 0, 1] = C2_mean
parameters[:, 1, 1] = C2_std
parameters[:, 2, 1] = C2_weight
parameters[:, 0, 2] = C3_mean
parameters[:, 1, 2] = C3_std
parameters[:, 2, 2] = C3_weight
parameters[:, 0, 3] = C4_mean
parameters[:, 1, 3] = C4_std
parameters[:, 2, 3] = C4_weight

# construct and save
dataset = ArtificialDataset(parameters, DistributionType.Normal)
with open("./dump/dataset.dump", "wb") as f:
    pickle.dump(dataset, f)
array = np.array([ages, series_1, series_2, series_3]).T
df = pd.DataFrame(array, columns=["Age (ka)", "Series 1", "Series 2", "Series 3"])
df.to_excel("./sheets/Series.xlsx")
