import matplotlib.pyplot as plt
import numpy as np
import time
import Obstacle_Avoidance as oa

obstacle_type = 'circles'
num_times_run = 2

start_obst_count = 3
end_obst_count = 4
start_obst_radius = 5
end_obst_radius = 6

start = time.time()
obstacle_count = [i for i in range(start_obst_count,end_obst_count+1)] # Ger start till end
obstacle_radius = [i for i in range(start_obst_radius,end_obst_radius+1)] # Ger start till end
num_of_points = len(obstacle_count)
print(obstacle_count,'óbstacle count')
print(obstacle_radius, 'óbstacle radius')


fish_eaten_matrix = np.zeros((len(obstacle_count), len(obstacle_radius)))
print(fish_eaten_matrix, 'initial fish eaten matrix eaten')

i = 0
flag = False
if flag:
    print('Simulation initiated')
    for obst_count in obstacle_count:
        j = 0
        for obst_rad in obstacle_radius:
            res = 0.0
            for k in range(num_times_run):
                temp = oa.main('circles', obst_count, obst_count,obst_rad, True)
                res += temp  # Anropa simulationen med olika turning speed
            res /= num_times_run
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

print(fish_eaten_matrix, 'final fish eaten matrix')
heatmap = plt.imshow(fish_eaten_matrix, interpolation='spline16', origin='lower')
plt.xlabel('Obstacle count')
plt.ylabel('Obstacle size')
cbar = plt.colorbar(heatmap)
cbar.set_label('Average fish eaten', rotation=270, labelpad=15)
plt.show()

x_, y_ = np.meshgrid(obstacle_count, obstacle_radius)
fig = plt.figure()
ax1 = plt.pcolormesh(x_,y_,fish_eaten_matrix)
plt.xlabel('Obstacle count')
plt.ylabel('Obstacle size')
plt.xticks(np.arange(start_obst_count, end_obst_count+1, step=1))  # Set label locations.
plt.yticks(np.arange(start_obst_radius, end_obst_radius+1, step=1))  # Set label locations.
plt.show()