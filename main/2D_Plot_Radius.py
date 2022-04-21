import matplotlib.pyplot as plt
import numpy as np
import time
import Obstacle_Avoidance as oa


obstacle_type = 'circles'
num_times_run = 100
seed = [n for n in range(num_times_run)]

start_obst_count = 4
end_obst_count = 4
start_obst_radius = 7
end_obst_radius = 28

start = time.time()
obstacle_count = 8
obstacle_radius = np.linspace(start_obst_radius, end_obst_radius, 8) # Ger start till end
num_of_points = len(obstacle_count)
print(obstacle_count,'obstacle count')
print(obstacle_radius, 'obstacle radius')

fish_eaten_matrix = np.zeros((len(obstacle_radius), len(num_times_run)))
print(fish_eaten_matrix, 'initial fish eaten matrix eaten')

i = 0
new_simulation = True
if new_simulation:
    print('Simulation initiated')
    for obst_rad in obstacle_radius:
        res = 0.0
        for k in range(num_times_run):
            temp = oa.main('circles', obstacle_count, obstacle_count, obst_rad, True, seed[k])
            print('Fiskar ätna:  ', temp)
            res += temp
        res /= num_times_run
        fish_eaten_matrix[j, i] = res  # Medelvärde av antal ätna fiskar
        j = j + 1
        np.save('fish_eaten_matrix.npy', fish_eaten_matrix)
        # Print how far through the simulation we are currently
        print('Progress =', round((i / num_of_points * 100) + ((j / num_of_points * 100) / num_of_points), 2), '%')
    i = i + 1
    # Print how much time has passed since simulation start
    duration = divmod(time.time() - start, 60)
    print('Time passed:', int(duration[0]), 'm,', round(duration[1], 1), 's')
    np.save('fish_eaten_matrix.npy', fish_eaten_matrix)

    np.save('fish_eaten_matrix.npy', fish_eaten_matrix)
    # Print total time taken
    duration = divmod(time.time() - start, 60)
    print('Total time:', int(duration[0]), 'm,', round(duration[1], 1), 's')
else:
    fish_eaten_matrix = np.load('fish_eaten_matrix.npy')