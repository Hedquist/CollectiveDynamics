# Program som  beräknar medelvärden och std
import numpy as np
import matplotlib.pyplot as plt
import several_predators

n = 5  # Antal observationer

several_predators.main()
x = np.load('fish_eaten.npy')
simulation_iterations = x.shape[0]
data = np.zeros((simulation_iterations, n))
data[:, 0] = x[:, 0]
t = x[:, 1]

def process(i):
    several_predators.main()
    x = np.load('fish_eaten.npy')
    global data
    data[:, i] = x[:, 0]

for i in range(1, n):
    process(i)
fish_eaten_mean = np.mean(data, axis=1)
fish_eaten_std = np.std(data, axis=1)
plt.errorbar(t, fish_eaten_mean, yerr=fish_eaten_std, color='blue')
plt.show()
