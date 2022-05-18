# Program som  beräknar medelvärden och std samt plottar en figur
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import time
from timeit import default_timer as timer
import simple_torque as st
import data_capture as dc

num_times_run = dc.num_times_run  # Antal observationer
seed = [n for n in range(num_times_run)]

simulation_iterations = st.simulation_iterations
num = dc.num_of_points
t = np.linspace(1, num , num)*0.01
# fish_eaten_all_sim = []
# new_simulation = True
#
# if new_simulation:
#     start = timer()
#     for N in range(num_times_run):
#         oa.main('circles', 8, 8, 16, True, seed[N])
#         x = np.load('fish_eaten_this_sim.npy')
#         fish_eaten_all_sim.append(x)
#         print('Progress =', round(((N+1) / num_times_run * 100) , 2), '%')
#     np.save('fish_eaten_all_sim.npy', fish_eaten_all_sim)
#     print("Time:", timer() - start)  # Skriver hur lång tid simulationen tog
# else:
#     fish_eaten_all_sim.extend(np.load('fish_eaten_all_sim.npy'))


# fish_eaten_mean = np.mean(np.array(fish_eaten_all_sim), axis=0)
# fish_eaten_std = np.std(np.array(fish_eaten_all_sim), axis=0)  # Beräkna standardavvikelsen längs rader
fish_eaten_mean = np.load('result_100_1.npy')
fish_eaten_mean = fish_eaten_mean[0:2, :]/200*100
fish_eaten_mean = np.flip(fish_eaten_mean)
fish_eaten_std = np.load('result_std_100_1.npy')
fish_eaten_std = fish_eaten_std[0:2, :]/200*100
fish_eaten_std = np.flip(fish_eaten_std)

# plt.figure('All sim graph')
# fig, ax = plt.subplots(1, 2, num='Medelvärden och standardavvikelser', figsize=(7.5, 6.5), sharex=True, sharey=True)
# ax[1].set_ylabel('.', color=(0, 0, 0, 0))
# fig.text(0.5, 0.04, 'Vinkelhastighet rovdjur [\u03C0 rad/tidssteg]', va='center', ha='center', fontsize=16)
# fig.text(0.03, 0.5, 'Andel fångade byten [%]', va='center', ha='center', rotation='vertical', fontsize=16)

# plt.axes(ax[i])
plt.figure(0, figsize=(7.5, 6.5))
markers, caps, bars = plt.errorbar(t, fish_eaten_mean[1, :], yerr=fish_eaten_std[1, :], fmt='b-')
plt.tick_params(axis='both', labelsize=16, right=True, top=True)
[bar.set_alpha(0.3) for bar in bars]  # Gör errorbars mer genomskinliga
[cap.set_alpha(0.3) for cap in caps]
plt.xlabel('Vinkelhastighet rovdjur [\u03C0 rad/\u0394t]', labelpad=1, fontsize=16)
plt.ylabel('Medelvärde av andel fångade bytesdjur [%]', labelpad=1, fontsize=16)
plt.axis([0, 0.105, 3.5, 23])
plt.show()

# plt.axes(ax[i])
plt.figure(1, figsize=(7.5, 6.5))
plt.tick_params(axis='both', labelsize=16, right=True, top=True)
markers, caps, bars = plt.errorbar(t, fish_eaten_mean[0, :], yerr=fish_eaten_std[0, :], fmt='b-')
[bar.set_alpha(0.3) for bar in bars]  # Gör errorbars mer genomskinliga
[cap.set_alpha(0.3) for cap in caps]
plt.xlabel('Vinkelhastighet rovdjur [\u03C0 rad/\u0394t]', labelpad=1, fontsize=16)
plt.ylabel('Medelvärde av andel fångade bytesdjur [%]', labelpad=1, fontsize=16)
plt.axis([0, 0.105, 3.5, 23])
plt.show()