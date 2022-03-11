# Skapa en heatmap för antal fiskar ätna för olika turning speeds

import numpy as np
import matplotlib.pyplot as plt
import simple_torque as st

fish_turning_speed = np.linspace(0.01, 1, 10)  # Fiskarnas turning speed
shark_turning_speed = np.linspace(0.01, 1, 10)  # Hajarnas turning speed
N = 1

n = len(fish_turning_speed)
fish_eaten_matrix = np.zeros((n, n))  # Allokera minne

i = 0
for fts in fish_turning_speed:
    j = 0
    for sts in shark_turning_speed:
        res = 0.0
        for k in range(N):
            temp = st.main(fts, sts)
            res += temp  # Anropa simulationen med olika turning speed
        res /= N
        fish_eaten_matrix[i, j] = res  # Medelvärde av antal ätna fiskar
        j = j + 1
    i = i + 1

# rng = np.random.default_rng()
# fish_eaten_matrix = rng.integers(low=0, high=st.fish_count-4, size=n*n).reshape((n, n))
plt.imshow(fish_eaten_matrix, interpolation='nearest', origin='lower')
plt.colorbar()
plt.show()
