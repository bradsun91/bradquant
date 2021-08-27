"""
    Kapman Project
    Input file: SPY.csv
    Date,Open,High,Low,Close,Adj Close,Volume

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pykalman import KalmanFilter

# Read
initial_values = pd.read_csv("SPY.csv")['Adj Close'].values

kf = KalmanFilter(transition_matrices = [1],
                  observation_matrices = [1],
                  initial_state_mean = initial_values[0],
                  initial_state_covariance = 1,
                  observation_covariance=1,
                  transition_covariance=.01)

state_means,_ = kf.filter(initial_values)
new_values = state_means.flatten()


# Plot

figure = plt.figure(figsize = (11, 8))
plt.plot(initial_values)
plt.plot(new_values)
plt.legend(['Initial Mean', 'New State Mean'], loc = 'upper left')
plt.xlabel("Date")
plt.ylabel("Values")
plt.show()
