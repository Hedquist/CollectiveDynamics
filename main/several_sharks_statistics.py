# Program som  beräknar medelvärden och std samt plottar en figur
import numpy as np
import matplotlib.pyplot as plt
import several_predators as sp

n = 3  # Antal observationer
simulation_iterations = sp.simulation_iterations
t = np.linspace(0, sp.simulation_iterations*sp.time_step, sp.simulation_iterations)
fish_eaten_all_sim = []
for i in range(n):
    sp.main()
    x = np.load('fish_eaten_this_sim.npy')
    fish_eaten_all_sim.append(x)


fish_eaten_mean = np.mean(np.array(fish_eaten_all_sim), axis=0)
fish_eaten_std = np.std(np.array(fish_eaten_all_sim), axis=0)  # Beräkna standardavvikelsen längs rader
fig, ax = plt.subplots()
markers, caps, bars = ax.errorbar(np.linspace(1,simulation_iterations,simulation_iterations), fish_eaten_mean, yerr=fish_eaten_std, fmt='b-')
[bar.set_alpha(0.3) for bar in bars]  # Gör errorbars mer genomskinliga
[cap.set_alpha(0.3) for cap in caps]
plt.show()