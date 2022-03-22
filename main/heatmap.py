# Skapa en heatmap för antal fiskar ätna för olika turning speeds

import matplotlib.pyplot as plt
import numpy as np
import time
import simple_torque as st

start = time.time()
num_of_points= 5
N = 3

fish_turning_speed = np.linspace(0.4, 0.6, num_of_points)  # Fiskarnas turning speed
shark_turning_speed = np.linspace(0.4, 0.6, num_of_points)  # Hajarnas turning speed
fish_eaten_matrix = np.zeros((num_of_points, num_of_points))  # Allokera minne

i = 0
for fts in fish_turning_speed:
    j = 0
    print('Progress = ', i/num_of_points*100, '%')
    for sts in shark_turning_speed:
        res = 0.0
        for k in range(N):
            temp = st.main(fts, sts)
            res += temp  # Anropa simulationen med olika turning speed
        res /= N
        fish_eaten_matrix[j, i] = res  # Medelvärde av antal ätna fiskar
        j = j + 1
    i = i + 1

duration = time.time() - start
print('Time [s]: ', duration)
plt.imshow(fish_eaten_matrix, interpolation='spline16', origin='lower')
plt.colorbar()
plt.show()
