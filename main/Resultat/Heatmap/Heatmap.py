import matplotlib.pyplot as plt
import numpy as np
import time

Q1 = np.load('fish_eaten_matrix_Q1.npy')[2:, 2:]
Q2 = np.load('fish_eaten_matrix_Q2.npy')[2:, :]
Q3 = np.load('fish_eaten_matrix_Q3.npy')[:, 2:]
Q4 = np.load('fish_eaten_matrix_Q4.npy')


Q1Q2 = np.concatenate( (Q1, Q2),axis=1)
Q3Q4 = np.concatenate( (Q3, Q4),axis=1)
Q = np.concatenate( (Q1Q2,Q3Q4), axis= 0)

start_obst_count = 1
end_obst_count = 8
start_obst_radius = 7
end_obst_radius = 28

start = time.time()
obstacle_count = np.linspace(start_obst_count-1, end_obst_count, 9, dtype=int) # Ger start till end
obstacle_radius = np.linspace(start_obst_radius-1, end_obst_radius, 9) # Ger start till end
print(obstacle_count,'obstacle count')
print(obstacle_radius, 'obstacle radius')

x_, y_ = np.meshgrid([2, 3, 4, 5, 6, 7, 8], [10, 13, 16, 19, 22, 25, 28])
#[4, 9, 16, 25, 36, 49, 64]
fig = plt.figure()
ax1 = plt.pcolormesh(x_+0.5,y_+1.5,Q)
plt.xlabel('Kvadratroten av totala antalet hinder')
plt.ylabel('Radie på hinder')
plt.yticks([13, 16, 19, 22, 25, 28])  # Set label locations.
plt.xticks([3, 4, 5, 6, 7, 8])  # Set label locations.
# plt.title('Antal fångade bytesdjur som funktion av antalet hinder och dess storlek ')
cbar = plt.colorbar(ax1)
cbar.set_label('Medelvärde av antal fångade bytesdjur', rotation=270, labelpad=15)
plt.show()


