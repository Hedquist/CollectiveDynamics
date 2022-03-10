# Skapa en heatmap för antal fiskar ätna för olika turning speeds

import numpy as np
import matplotlib.pyplot as plt
import simple_torque as st

fish_turning_speed = np.linspace(0.01, 1, 100)
shark_turning_speed = np.linspace(0.01, 1, 100)

n = len(fish_turning_speed)
fish_eaten_matrix = np.zeros((n, n))

i = 0
for fts in fish_turning_speed:
    j = 0
    for sts in shark_turning_speed:
        res = 0
        for k in range(10):
            res += st.main(fts, sts)
        res /= 10
        fish_eaten_matrix[i, j] = res
        j = j + 1
    i = i + 1

plt.imshow(fish_eaten_matrix)
plt.show()