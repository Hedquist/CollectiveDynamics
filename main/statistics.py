# Program som  beräknar medelvärden och std
import numpy as np
import matplotlib.pyplot as plt
import several_predators as sp

n = 5  # Antal observationer

data = np.zeros((sp.simulation_iterations, n))
t = np.linspace(0, sp.simulation_iterations*sp.time_step, sp.simulation_iterations)

def process(i):
    sp.main()
    x = np.load('fish_eaten.npy')
    global data
    data[:, i] = x[:, 0]

for i in range(n):
    process(i)
fish_eaten_mean = np.mean(data, axis=1)
fish_eaten_std = np.std(data, axis=1)
plt.errorbar(t, fish_eaten_mean, yerr=fish_eaten_std, color='blue')
plt.show()
