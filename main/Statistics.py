# Program som  beräknar medelvärden och std samt plottar en figur
import numpy as np
import matplotlib.pyplot as plt
import time
from timeit import default_timer as timer
import Obstacle_Avoidance as oa

num_times_run = 1  # Antal observationer
seed = [n for n in range(num_times_run)]

simulation_iterations = oa.simulation_iterations
t = np.linspace(0, simulation_iterations-1 , simulation_iterations)
fish_eaten_all_sim = []
new_simulation = True

if new_simulation:
    start = timer()
    for N in range(num_times_run):
        oa.main('circles', 8, 8, 16, True, seed[N])
        x = np.load('fish_eaten_this_sim.npy')
        fish_eaten_all_sim.append(x)
        print('Progress =', round(((N+1) / num_times_run * 100) , 2), '%')
    np.save('fish_eaten_all_sim.npy', fish_eaten_all_sim)
    print("Time:", timer() - start)  # Skriver hur lång tid simulationen tog
else:
    fish_eaten_all_sim.extend(np.load('fish_eaten_all_sim.npy'))


fish_eaten_mean = np.mean(np.array(fish_eaten_all_sim), axis=0)
fish_eaten_std = np.std(np.array(fish_eaten_all_sim), axis=0)  # Beräkna standardavvikelsen längs rader
fig, ax = plt.subplots(num = 'Meand and std')
markers, caps, bars = ax.errorbar(t, fish_eaten_mean, yerr=fish_eaten_std, fmt='b-')
[bar.set_alpha(0.3) for bar in bars]  # Gör errorbars mer genomskinliga
[cap.set_alpha(0.3) for cap in caps]

plt.figure('All sim graph')
for i in range(np.shape(fish_eaten_all_sim)[0]):
    plt.plot(t , np.array(fish_eaten_all_sim[i]))  # Plotta
plt.xlabel('Tid')
plt.ylabel('Antal fiskar ätna')
plt.show()