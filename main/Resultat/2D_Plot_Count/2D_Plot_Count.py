import matplotlib.pyplot as plt
import numpy as np
import time
import Obstacle_Avoidance as oa


obstacle_type = 'circles'
num_times_run = 100
seed = [n for n in range(num_times_run)]

start_obst_count = 5
end_obst_count = 8


start = time.time()
obstacle_radius = 25 # Hålls konstant
obstacle_count = np.linspace(start_obst_count, end_obst_count, 2) # Ger start till end
num_of_points = len(obstacle_count)
print(obstacle_count,'obstacle count')
print(obstacle_radius, 'obstacle radius')

fish_eaten_matrix = np.zeros( (len(obstacle_count), num_times_run) )
print(fish_eaten_matrix, 'initial fish eaten matrix eaten')

i = 0
new_simulation = False
if new_simulation:
    print('Simulation initiated')
    for obst_count in obstacle_count:
        res = 0.0
        for k in range(num_times_run):
            temp = oa.main('circles', obst_count, obst_count, obstacle_radius, True, seed[k]) # Fiskar ätna
            print('Fiskar ätna:  ', temp)
            fish_eaten_matrix[i,k] = temp
        i += 1
        np.save('../../fish_eaten_matrix.npy', fish_eaten_matrix)
        # Print how far through the simulation we are currently
        print('Progress =', round((i / num_of_points * 100), 2), '%')
    # Print how much time has passed since simulation start
    duration = divmod(time.time() - start, 60)
    print('Time passed:', int(duration[0]), 'm,', round(duration[1], 1), 's')
    np.save('../../fish_eaten_matrix.npy', fish_eaten_matrix)

    np.save('../../fish_eaten_matrix.npy', fish_eaten_matrix)
    # Print total time taken
    duration = divmod(time.time() - start, 60)
    print('Total time:', int(duration[0]), 'm,', round(duration[1], 1), 's')
else:
    fish_eaten_matrix = np.load('../../fish_eaten_matrix.npy')

print(fish_eaten_matrix)

fish_eaten_mean = np.mean(np.array(fish_eaten_matrix), axis=1)
print(fish_eaten_mean)
fish_eaten_std = np.std(np.array(fish_eaten_matrix), axis=1)  # Beräkna standardavvikelsen längs rader
fig, ax = plt.subplots(num = 'Meand and std')
markers, caps, bars = ax.errorbar(obstacle_count, fish_eaten_mean, yerr=fish_eaten_std, fmt='b-')
[bar.set_alpha(0.3) for bar in bars]  # Gör errorbars mer genomskinliga
[cap.set_alpha(0.3) for cap in caps]

plt.figure('All sim graph')
for i in range(np.shape(fish_eaten_matrix)[0]):
    plt.plot(obstacle_count , np.array(fish_eaten_matrix[i,:]))  # Plotta
plt.xlabel('Antal hinder ')
plt.ylabel('Medelantalet fångade byten')
plt.show()