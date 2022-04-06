# Skapa en heatmap för antal fiskar ätna för olika turning speeds

import matplotlib.pyplot as plt
import numpy as np
import time
import simple_torque as st

start = time.time()
num_of_points = 20  # Amount of points in each array, main point of running time adjustment
N = 3

fish_turning_speed = np.linspace(0, 0.1, num_of_points)  # Fiskarnas turning speed
shark_turning_speed = np.linspace(0, 0.1, num_of_points)  # Hajarnas turning speed
fish_eaten_matrix = np.zeros((num_of_points, num_of_points))  # Allokera minne

i = 0
flag = False
if flag:
    print('Simulation initiated')
    for fts in fish_turning_speed:
        j = 0
        for sts in shark_turning_speed:
            res = 0.0
            for k in range(N):
                temp = st.main(fts, sts, False)
                res += temp  # Anropa simulationen med olika turning speed
            res /= N
            fish_eaten_matrix[j, i] = res  # Medelvärde av antal ätna fiskar
            j = j + 1
            # Print how far through the simulation we are currently
            print('Progress =', round((i / num_of_points * 100) + ((j / num_of_points * 100) / num_of_points), 2), '%')
        i = i + 1
        # Print how much time has passed since simulation start
        duration = divmod(time.time() - start, 60)
        print('Time passed:', int(duration[0]), 'm,', round(duration[1], 1), 's')

    np.save('fish_eaten_matrix.npy', fish_eaten_matrix)
    # Print total time taken
    duration = divmod(time.time() - start, 60)
    print('Total time:', int(duration[0]), 'm,', round(duration[1], 1), 's')
else:
    fish_eaten_matrix = np.load('fish_eaten_matrix.npy')

x_, y_ = np.meshgrid(fish_turning_speed, shark_turning_speed)
fig = plt.figure()
ax1 = plt.pcolormesh(x_,y_,fish_eaten_matrix)
heatmap = plt.imshow(fish_eaten_matrix, origin='lower')
plt.xlabel('Fish turning speed')
plt.ylabel('Shark turning speed')
plt.xticks(np.arange(0, num_of_points, step=4), [round(fish_turning_speed[i], 4) for i in range(0, len(fish_turning_speed), 4)])  # Set label locations.
plt.yticks(np.arange(0, num_of_points, step=4), [round(shark_turning_speed[i], 4) for i in range(0, len(shark_turning_speed), 4)])  # Set label locations.
cbar = plt.colorbar(heatmap)
cbar.set_label('Average fish eaten', rotation=270, labelpad=15)
plt.show()
