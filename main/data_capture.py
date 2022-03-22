import matplotlib.pyplot as plt
import numpy as np
import time
import Obstacle_Avoidance as oa

obstacle_count_array = np.array([2, 3, 4, 5])
obstacle_type = 'circles'
obstacle_radius = 5

fish_eaten_array = np.zeros(np.obstacle_count_array.size())

for i in range(np.obstacle_count_array.size()):
    fish_eaten_array[i] = oa(obstacle_count_array[i], obstacle_type, obstacle_radius)

plt.plot(obstacle_count_array, fish_eaten_array)
