# Program som beräknar medelvärden och std samt plottar en figur
import numpy as np
import matplotlib.pyplot as plt
import several_predators as sp
import time
from timeit import default_timer as timer

start = timer()  # Timer startas
mean_fish_eaten_per_shark_count = []
std_fish_eaten_per_shark_count = []
shark_counts = []
for j in range(5):
    sp.shark_count = (j+1)
    n = 3  # Antal observationer
    simulation_iterations = sp.simulation_iterations # Hämta antalet iterationer i simulationen
    fish_eaten_all_sim = []
    for i in range(n):
        sp.seed = i+j # Välj seed
        sp.main()   # Startar simulationen
        x = np.load('fish_eaten_this_sim.npy')
        fish_eaten_all_sim.append(x)    # Sparar ner resultat för varje simulation som körs

    fig, ax = plt.subplots()

    fish_eaten_mean = np.mean(np.array(fish_eaten_all_sim), axis=0) # Beräkna medelvärde
    fish_eaten_std = np.std(np.array(fish_eaten_all_sim), axis=0)  # Beräkna standardavvikelsen längs rader
    #markers, caps, bars = ax.errorbar(np.linspace(1,simulation_iterations,simulation_iterations), fish_eaten_mean, yerr=fish_eaten_std, fmt='b-')
    #[bar.set_alpha(0.3) for bar in bars]  # Gör errorbars mer genomskinliga
    #[cap.set_alpha(0.3) for cap in caps]

    mean_fish_eaten_per_shark_count.append(fish_eaten_mean[-1]/sp.shark_count)
    std_fish_eaten_per_shark_count.append(fish_eaten_std[-1]/sp.shark_count)
    shark_counts.append(sp.shark_count)

fig, ax = plt.subplots()
markers, caps, bars = ax.errorbar(shark_counts, mean_fish_eaten_per_shark_count, yerr=std_fish_eaten_per_shark_count, fmt='b-')
[bar.set_alpha(0.3) for bar in bars]  # Gör errorbars mer genomskinliga
[cap.set_alpha(0.3) for cap in caps]
#ax.bar(shark_counts, mean_fish_eaten_per_shark_count)
print("Time:", timer() - start)  # Skriver hur lång tid simulationen tog
plt.show()