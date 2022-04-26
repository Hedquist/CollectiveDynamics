import matplotlib.pyplot as plt
import numpy as np
import time
import simple_torque as st

num_times_run = 1
seed = [n for n in range(num_times_run)]


start_fish_turn = 0
end_fish_turn = 0.1
start_shark_turn = 0
end_shark_turn = 0.1

start = time.time()
fish_turns = np.linspace(start_fish_turn, end_fish_turn, 4)  # Ger start till end
shark_turns = np.linspace(start_shark_turn, end_shark_turn, 4)  # Ger start till end
num_of_points = len(fish_turns)
print(fish_turns, 'fish turn speed')
print(shark_turns, 'shark turn speed')


fish_eaten_matrix = np.zeros((len(fish_turns), len(shark_turns)))
print(fish_eaten_matrix, 'initial fish eaten matrix eaten')

i = 0
new_simulation = True
if new_simulation:
    print('Simulation initiated')
    for fts in fish_turns:
        j = 0
        for sts in shark_turns:
            res = 0.0
            for k in range(num_times_run):
                temp = st.main(fts, sts, False, seed[k])
                print('FTS:', fts, ' STS:', sts, ' Fiskar ätna:', temp)
                res += temp  # Anropa simulationen med olika turning speed
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

print(fish_eaten_matrix, 'final fish eaten matrix')
heatmap = plt.imshow(fish_eaten_matrix, interpolation='spline16', origin='lower')
plt.xlabel('Fish turn speed')
plt.ylabel('Shark turn speed')
cbar = plt.colorbar(heatmap)
cbar.set_label('Average fish eaten', rotation=270, labelpad=15)
plt.show()

fish_turns = np.linspace(start_fish_turn - 1, end_fish_turn, 5)  # Ger start till end
shark_turns = np.linspace(start_shark_turn - 1, end_shark_turn, 5)  # Ger start till end
x_, y_ = np.meshgrid(fish_turns, shark_turns)
fig = plt.figure()
ax1 = plt.pcolormesh(x_, y_, fish_eaten_matrix)
plt.xlabel('Fish turn speed')
plt.ylabel('Shark turn speed')
plt.xticks(np.arange(start_obst_count, end_obst_count+1, step=1))  # Set label locations.
plt.yticks(np.arange(start_obst_radius, end_obst_radius+1, step=3))  # Set label locations.
cbar = plt.colorbar(ax1)
cbar.set_label('Average fish eaten', rotation=270, labelpad=15)
plt.show()
