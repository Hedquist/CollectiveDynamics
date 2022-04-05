# Program som  beräknar medelvärden och std samt plottar en figur
import numpy as np
import matplotlib.pyplot as plt
import Obstacle_Avoidance as oa

num_times_run = 3  # Antal observationer
seed = [n + 100 for n in range(num_times_run)]

simulation_iterations = oa.simulation_iterations
t = np.linspace(0, oa.simulation_iterations * oa.time_step, oa.simulation_iterations)
fish_eaten_all_sim = []

for N in range(num_times_run):
    oa.main('circles', 8, 8, 8, True, seed[N])
    x = np.load('fish_eaten_this_sim.npy')
    fish_eaten_all_sim.append(x)

fish_eaten_mean = np.mean(np.array(fish_eaten_all_sim), axis=0)
fish_eaten_std = np.std(np.array(fish_eaten_all_sim), axis=0)  # Beräkna standardavvikelsen längs rader
fig, ax = plt.subplots()
markers, caps, bars = ax.errorbar(np.linspace(1,simulation_iterations,simulation_iterations), fish_eaten_mean, yerr=fish_eaten_std, fmt='b-')
[bar.set_alpha(0.3) for bar in bars]  # Gör errorbars mer genomskinliga
[cap.set_alpha(0.3) for cap in caps]
plt.show()