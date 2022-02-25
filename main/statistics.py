# Program som  beräknar medelvärden och std
import numpy as np
import matplotlib.pyplot as plt

simulation_time = 0
time_step = 0
n = 3  # Antal observationer
data = []
t = []
for i in range(n):
    exec(open('several_predators.py').read())
    x = np.load('fish_eaten.npy')
    if i == 0:
        time_step = x[1, 1] - x[0, 1]
        simulation_time = x.shape[0]
        data = np.zeros((simulation_time, n))
        t = x[:, 1]
    data[:, i] = x[:, 0]
fish_eaten_mean = np.mean(data, 1)
plt.plot(t, fish_eaten_mean)