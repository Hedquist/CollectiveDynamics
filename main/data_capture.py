import matplotlib.pyplot as plt
import numpy as np
import time
import simple_torque as st

num_times_run = 100
seed = [n for n in range(num_times_run)]


start_fish_turn = 0.01
end_fish_turn = 0.1
start_shark_turn = 0.01
end_shark_turn = 0.1

start = time.time()
fish_turns = np.linspace(start_fish_turn, end_fish_turn, 10)  # Ger start till end
shark_turns = np.linspace(start_shark_turn, end_shark_turn, 10)  # Ger start till end
num_of_points = len(fish_turns)
print(fish_turns, 'fish turn speed')
print(shark_turns, 'shark turn speed')


#fish_eaten_matrix = np.zeros((len(fish_turns), len(shark_turns)))
fish_eaten_matrix = np.load('fish_eaten_matrix.npy')
#standard_dev = np.zeros((len(fish_turns), len(shark_turns)))
standard_dev = np.load('standard_dev.npy')
print(fish_eaten_matrix, 'initial fish eaten matrix eaten')

i = 0
new_simulation = False
if new_simulation:
    print('Simulation initiated')
    for fts in fish_turns:
        j = 0
        for sts in shark_turns:
            data = np.zeros(num_times_run)
            for k in range(num_times_run):
                data[k] = st.main(fts, sts, False, seed[k]) # Anropa simulationen med olika turning speed
                print('FTS:', fts, ' STS:', sts, ' Fiskar ätna:', data[k])
                #print(fish_eaten_matrix)
            fish_eaten_matrix[i, j] = np.mean(data)  # Medelvärde av antal ätna fiskar
            standard_dev[i, j] = np.std(data)
            #print(fish_eaten_matrix)
            j = j + 1
            np.save('fish_eaten_matrix.npy', fish_eaten_matrix)
            np.save('standard_dev.npy', standard_dev)
            # Print how far through the simulation we are currently
            print('Progress =', round((i / len(fish_turns) * 100) + ((j / len(shark_turns) * 100) / len(fish_turns)), 2), '%')
        i = i + 1
        # Print how much time has passed since simulation start
        duration = divmod(time.time() - start, 60)
        print('Time passed:', int(duration[0]), 'm,', round(duration[1], 1), 's')
        np.save('fish_eaten_matrix.npy', fish_eaten_matrix)
        np.save('standard_dev.npy', standard_dev)

    np.save('fish_eaten_matrix.npy', fish_eaten_matrix)
    np.save('standard_dev.npy', standard_dev)
    # Print total time taken
    duration = divmod(time.time() - start, 60)
    print('Total time:', int(duration[0]), 'm,', round(duration[1], 1), 's')
else:
    fish_eaten_matrix = np.load('result_100_1.npy')
    standard_dev = np.load('result_std_100_1.npy')

fish_eaten_matrix = np.rot90(fish_eaten_matrix, 1)
print(fish_eaten_matrix, 'final fish eaten matrix')
fish_eaten_matrix = fish_eaten_matrix/200*100
plt.figure(figsize=(7.5, 6.5))
heatmap = plt.imshow(fish_eaten_matrix, interpolation='none', origin='lower', extent=[0.005, 0.105, 0.005, 0.105])
plt.xlabel('Vinkelhastighet byte [\u03C0 rad/\u0394t]', fontsize=16)
plt.ylabel('Vinkelhastighet rovdjur [\u03C0 rad/\u0394t]', fontsize=16)
plt.tick_params(axis='both', labelsize=16, right=True, top=True)
cbar = plt.colorbar(heatmap)
cbar.set_label('Medelvärde av andel fångade bytesdjur [%]', rotation=270, labelpad=15, fontsize=16)
plt.show()

fish_turns = np.linspace(start_fish_turn - 1, end_fish_turn, 5)  # Ger start till end
shark_turns = np.linspace(start_shark_turn - 1, end_shark_turn, 5)  # Ger start till end
x_, y_ = np.meshgrid(fish_turns, shark_turns)
fig = plt.figure()
# ax1 = plt.pcolormesh(x_, y_, fish_eaten_matrix)
# plt.xlabel('Fish turn speed')
# plt.ylabel('Shark turn speed')
# plt.xticks(np.arange(start_fish_turn, end_fish_turn+1, step=1))  # Set label locations.
# plt.yticks(np.arange(start_shark_turn, end_shark_turn+1, step=3))  # Set label locations.
# cbar = plt.colorbar(ax1)
# cbar.set_label('Average fish eaten', rotation=270, labelpad=15)
# plt.show()
