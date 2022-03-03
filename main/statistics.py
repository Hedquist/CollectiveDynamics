# Program som  beräknar medelvärden och std
import numpy as np
import matplotlib.pyplot as plt
import several_predators as sp

n = 10  # Antal observationer

data = np.zeros((sp.simulation_iterations, n))
t = np.linspace(0, sp.simulation_iterations*sp.time_step, sp.simulation_iterations)

for i in range(n):
    sp.main()
    x = np.load('fish_eaten.npy')
    data[:, i] = x[:, 0]

fish_eaten_mean = np.mean(data, axis=1)  # Beräkna medelvärdet längs rader
fish_eaten_std = np.std(data, axis=1)  # Beräkna standardavvikelsen längs rader
fig, ax = plt.subplots()
markers, caps, bars = ax.errorbar(t, fish_eaten_mean, yerr=fish_eaten_std, fmt='bo')
[bar.set_alpha(0.3) for bar in bars]  # Gör errorbars mer genomskinliga
[cap.set_alpha(0.3) for cap in caps]
plt.show()
