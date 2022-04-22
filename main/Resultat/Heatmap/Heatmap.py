import matplotlib.pyplot as plt
import numpy as np
import time

# Q1 = np.load('fish_eaten_matrix_Q1.npy')
# Q2 = np.load('fish_eaten_matrix_Q2.npy')
# Q3 = np.load('fish_eaten_matrix_Q3.npy')
# Q4 = np.load('fish_eaten_matrix_Q4.npy')


# Q1Q2 = np.concatenate( (Q1, Q2),axis=1)
# Q3Q4 = np.concatenate( (Q3, Q4),axis=1)
# Q = np.concatenate( (Q1Q2,Q3Q4), axis= 0)

start_obst_count = 1
end_obst_count = 8
start_obst_radius = 7
end_obst_radius = 28

start = time.time()
obstacle_count = np.linspace(start_obst_count, end_obst_count, 8, dtype=int) # Ger start till end
obstacle_radius = np.linspace(start_obst_radius, end_obst_radius, 8) # Ger start till end
print(obstacle_count,'obstacle count')
print(obstacle_radius, 'obstacle radius')

x_, y_ = np.meshgrid(obstacle_count, obstacle_radius)
fig = plt.figure()
ax1 = plt.pcolormesh(x_,y_,Q)
plt.xlabel('Obstacle count')
plt.ylabel('Obstacle size')
plt.xticks(np.arange(start_obst_count, end_obst_count+1, step=1))  # Set label locations.
plt.yticks(np.arange(start_obst_radius, end_obst_radius+1, step=1))  # Set label locations.
cbar = plt.colorbar(ax1)
cbar.set_label('Average fish eaten', rotation=270, labelpad=15)
plt.show()


