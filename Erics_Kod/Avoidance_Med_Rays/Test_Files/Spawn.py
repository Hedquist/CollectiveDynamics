import numpy as np

fish_count = 100
shark_count = 1
canvas_length = 100

fish_graphic_radius = 2  # Radius of agent
shark_graphic_radius = 3
murder_radius = shark_graphic_radius+3  # Hajen äter fiskar inom denna radie

# Fiskens koordinater
x = np.array(np.random.rand(fish_count) * 2 * canvas_length - canvas_length)
y = np.array(np.random.rand(fish_count) * 2 * canvas_length - canvas_length)
fish_coords = np.column_stack((x, y))
fish_orientations = np.random.rand(fish_count) * 2 * np.pi  # orientations

# Startkoordinater hajar
shark_coords = np.column_stack((0.0, 0.0))  # Array med alla hajars x- och y-koord
shark_orientations = np.random.rand(shark_count) * 2 * np.pi  # Array med alla hajars riktning

r = np.linalg.norm(shark_coords - fish_coords, axis=1)
indices = np.where(r < fish_graphic_radius + murder_radius)[0]
print(fish_coords)
print(r)
print(indices)
print(fish_coords[indices])
for i in indices:
    while np.linalg.norm(shark_coords- fish_coords[i], axis=1) < fish_graphic_radius + murder_radius :
        fish_coords[i] = [canvas_length*(np.random.rand() * 2 - 1), canvas_length * (np.random.rand() * 2 - 1)]
print(fish_coords[indices])

