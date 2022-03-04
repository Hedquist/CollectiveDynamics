import numpy as np
from tkinter import *
from scipy.spatial import *
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull
import time
from shapely.geometry import Polygon

res = 500  # Resolution of the animation
tk = Tk()
tk.geometry(str(int(res * 1.1)) + 'x' + str(int(res * 1.3)))
tk.configure(background='white')

canvas = Canvas(tk, bd=2, bg='white')  # Generate animation window
tk.attributes('-topmost', 0)
canvas.place(x=res / 20, y=res / 20, height=res, width=res)
ccolor = ['#17888E', '#C1D02B', '#9E00C9', '#D80000', '#E87B00', '#9F68D3', '#4B934F', '#FFFFFF']

# Variabler
fish_count = 50  # Antal fiskar
canvas_length = 100  # Storlek på ruta, från mitten till kant. En sida är alltså 2*l
fish_graphic_radius = 4  # Radie av ritad cirkel
fish_interaction_radius = 10  # Interraktionsradie för fisk
fish_speed = 2  # Hastighet fiskar
time_step = 0.2  # Storlek tidssteg
simulation_iterations = 4000  # Antalet iterationer simulationen kör
fish_noise = 0.1  # Brus i vinkel
murder_radius = 5  # Hajen äter fiskar inom denna radie

shark_count = 1  # Antal hajar (kan bara vara 1 just nu...)
shark_speed = 3  # Hajens fart
fish_eaten = []  # Array med antal fiskar ätna som 0e element och när det blev äten som 1a element
fish_eaten_count = 0  # Antal fiskar ätna

# Start koordinater fiskar
fish_coords_file = 'fish_coords_initial.npy'
fish_orientations_file = 'fish_orientations_initial.npy'
if True:
    x = np.random.rand(fish_count) * 2 * canvas_length - canvas_length  # x coordinates
    y = np.random.rand(fish_count) * 2 * canvas_length - canvas_length  # y coordinates
    fish_orientations = np.random.rand(fish_count) * 2 * np.pi  # orientations
    fish_coords = np.column_stack((x, y))
    np.save(fish_coords_file, fish_coords)
    np.save(fish_orientations_file, fish_orientations)
else:
    fish_coords = np.load(fish_coords_file)  # Array med alla fiskars x- och y-koord
    fish_orientations = np.load(fish_orientations_file)  # Array med alla fiskars riktning

# Startkoordinater hajar
shark_coords = np.column_stack((0.0, 0.0))  # Array med alla hajars x- och y-koord
shark_orientations = np.random.rand(shark_count) * 2 * np.pi  # Array med alla hajars riktning

# Raycasting
step_angle = 2 * np.arctan(fish_graphic_radius / fish_interaction_radius)
casted_rays = 6
FOV_angle = step_angle * (casted_rays - 1)  # Field of view angle
half_FOV = FOV_angle / 2

fish_rays_coords = [[] for i in range(fish_count)]
fish_rays_angle_relative_velocity = [[] for i in range(fish_count)]
shark_rays_coords = []
shark_rays_angle_relative_velocity = []

fish_canvas_graphics = []  # De synliga cirklarna som är fiskar sparas här
shark_canvas_graphics = []  # De synliga cirklarna som är hajar sparas här


def update_position(coords, speed, orientations):  # Uppdaterar en partikels position
    coords[:, 0] = (coords[:, 0] + speed * np.cos(orientations) * time_step + canvas_length) % (
            2 * canvas_length) - canvas_length
    coords[:, 1] = (coords[:, 1] + speed * np.sin(orientations) * time_step + canvas_length) % (
            2 * canvas_length) - canvas_length
    return coords


def calculate_distance(coords, coord):  # Räknar ut avstånd mellan punkterna coords och punkten coord
    return np.sqrt((coords[:, 0] - coord[0]) ** 2 + (coords[:, 1] - coord[1]) ** 2)


def get_direction(coord1, coord2):  # Ger riktningen från coord1 till coord2 i radianer
    return np.arctan2(coord2[1] - coord1[1], coord2[0] - coord1[0])


def calculate_cluster_coeff(coords, interaction_radius, count):  # Beräknar Cluster Coefficient
    v = Voronoi(coords)
    coeff = 0
    for i, reg_num in enumerate(v.point_region):
        # clock = time.time()
        indices = v.regions[reg_num]

        if -1 not in indices:  # some regions can be opened
            area = Polygon(v.vertices[indices]).area
            if area < interaction_radius ** 2 * np.pi:
                coeff = coeff + 1

    return coeff / count


def murder_fish_coords(dead_fish_index):  # Tar bort fisk som blivit uppäten
    new_fish_coords = np.delete(fish_coords, dead_fish_index, 0)
    return new_fish_coords


def murder_fish_orientations(dead_fish_index):
    new_fish_orientations = np.delete(fish_orientations, dead_fish_index)
    return new_fish_orientations


def detect_wall(ray_coords):
    rays = canvas_length - np.absolute(np.array(ray_coords)) < 0
    rays_outside_wall = np.array([False, False, False, False, False, False])
    for l in range(casted_rays):
        rays_outside_wall[l] = True in rays[l]
    if not rays_outside_wall[int(len(rays_outside_wall) / 2 - 1)] and not rays_outside_wall[
        int(len(rays_outside_wall) / 2)]:
        sign = 0
    else:
        if all(rays_outside_wall):
            sign = 1
        else:
            i = 1
            first_free_index = int(len(rays_outside_wall) / 2) - 1
            while rays_outside_wall[first_free_index]:
                first_free_index += i * (-1) ** (i - 1)
                i += 1
            sign = -1 if (first_free_index <= 2) else 1

    angle_weight = np.pi / 4 * sign  # Vikta med avstånd
    return angle_weight


def cast_rays():
    for j in range(fish_count):
        start_angle = fish_orientations[j] - half_FOV  # Startvinkel
        for ray in range(casted_rays):
            fish_rays_coords[j].append([fish_coords[j][0] + fish_interaction_radius * np.cos(start_angle),
                                        fish_coords[j][1] + fish_interaction_radius * np.sin(start_angle)])
            fish_rays_angle_relative_velocity[j].append(start_angle)
            start_angle += step_angle  # Uppdaterar vinkel för ray

    start_angle = shark_orientations - half_FOV  # Startvinkel
    for ray in range(casted_rays):
        shark_rays_coords.append([shark_coords[0,0] + fish_interaction_radius * np.cos(start_angle),
                                     shark_coords[0,1] + fish_interaction_radius * np.sin(start_angle)])
        shark_rays_angle_relative_velocity.append(start_angle)
        start_angle += step_angle  # Uppdaterar vinkel för ray


for j in range(shark_count):  # Skapar cirklar för hajar
    shark_canvas_graphics.append(
        canvas.create_oval((shark_coords[j, 0] - fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                           (shark_coords[j, 1] - fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                           (shark_coords[j, 0] + fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                           (shark_coords[j, 1] + fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                           outline=ccolor[1], fill=ccolor[1]))
for j in range(fish_count):  # Skapar cirklar för fiskar
    fish_canvas_graphics.append(
        canvas.create_oval((fish_coords[j, 0] - fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                           (fish_coords[j, 1] - fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                           (fish_coords[j, 0] + fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                           (fish_coords[j, 1] + fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                           outline=ccolor[0], fill=ccolor[0]))

cast_rays()

# Skapar ett canvas textobjekt för global alignemnt coefficent
global_alignment_canvas_text = canvas.create_text(100, 20, text=1 / fish_count * np.linalg.norm(
    [np.sum(np.cos(fish_orientations)),
     np.sum(np.sin(fish_orientations))]))

# Skapar ett canvas textobjekt för clustering coefficent
clustering_coeff_canvas_text = canvas.create_text(100, 40,
                                                  text=calculate_cluster_coeff(fish_coords, fish_interaction_radius,
                                                                               fish_count))

# Loop för allt som ska ske varje tidssteg i simulationen
for t in range(simulation_iterations):
    fish_coords = update_position(fish_coords, fish_speed, fish_orientations)  # Uppdatera fiskposition
    shark_coords = update_position(shark_coords, shark_speed, shark_orientations)  # Uppdatera hajposition
    shark_fish_distances = calculate_distance(fish_coords, shark_coords[
        0])  # Räknar ut det kortaste avståndet mellan haj och varje fisk

    fish_distance_to_wall = (canvas_length - np.absolute(np.array(fish_coords)))
    fish_near_wall = fish_distance_to_wall < fish_interaction_radius
    shark_distance_to_wall = (canvas_length - np.absolute(np.array(shark_coords)))
    shark_near_wall = shark_distance_to_wall < fish_interaction_radius
    avoid_angle = 0

    closest_fish = np.argmin(shark_fish_distances)  # Index av fisk närmst haj

    for j in range(shark_count):
        # Updating animation coordinates haj
        canvas.coords(shark_canvas_graphics[j],
                      (shark_coords[j, 0] - fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                      (shark_coords[j, 1] - fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                      (shark_coords[j, 0] + fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                      (shark_coords[
                           j, 1] + fish_graphic_radius + canvas_length) * res / canvas_length / 2, )

    start_angle = shark_orientations - half_FOV  # Startvinkel
    start_angle_arc = start_angle  # Memorerar för j:te partikeln
    for ray in range(casted_rays):
        shark_rays_coords[ray] = [shark_coords[0, 0] + fish_interaction_radius * np.cos(start_angle),
                                     shark_coords[0, 1] + fish_interaction_radius * np.sin(start_angle)]
        shark_rays_angle_relative_velocity[ray] = start_angle
        start_angle += step_angle  # Uppdaterar vinkel för ray

    for j in range(len(fish_coords)):
        # Updating animation coordinates fisk
        canvas.coords(fish_canvas_graphics[j],
                      (fish_coords[j, 0] - fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                      (fish_coords[j, 1] - fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                      (fish_coords[j, 0] + fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                      (fish_coords[
                           j, 1] + fish_graphic_radius + canvas_length) * res / canvas_length / 2, )

        start_angle = fish_orientations[j] - half_FOV  # Startvinkel
        start_angle_arc = start_angle  # Memorerar för j:te partikeln
        for ray in range(casted_rays):
            fish_rays_coords[j][ray] = [fish_coords[j][0] + fish_interaction_radius * np.cos(start_angle),
                                        fish_coords[j][1] + fish_interaction_radius * np.sin(start_angle)]
            fish_rays_angle_relative_velocity[j][ray] = start_angle
            start_angle += step_angle  # Uppdaterar vinkel för ray

        if j == closest_fish:
            canvas.itemconfig(fish_canvas_graphics[j], fill=ccolor[2])  # Byt färg på fisk närmst haj
        else:
            canvas.itemconfig(fish_canvas_graphics[j], fill=ccolor[0])

        inter_fish_distances = calculate_distance(fish_coords, fish_coords[
            j])  # Räknar ut avstånd mellan fisk j och alla andra fiskar

        if fish_near_wall[j, 0] or fish_near_wall[j, 1]:
            avoid_angle = detect_wall(fish_rays_coords[j]) / (
                        np.minimum(fish_distance_to_wall[j, 0], fish_distance_to_wall[j, 1]) - fish_graphic_radius)
            print(np.rad2deg(avoid_angle))

        fish_in_interaction_radius = inter_fish_distances < fish_interaction_radius  # Vilka fiskar är inom en fisks interraktionsradie

        if shark_fish_distances[j] < fish_interaction_radius:  # Om hajen är nära fisken, undvik hajen
            fish_orientations[j] = get_direction(shark_coords[0], fish_coords[j])
        else:  # Annars Vicsek-modellen
            fish_orientations[j] = np.angle(
                np.sum(np.exp(fish_orientations[fish_in_interaction_radius] * 1j))) + fish_noise * np.random.uniform(
                -1 / 2, 1 / 2) + avoid_angle

        #   Shark direction härifrån
        if shark_near_wall[0, 0] or shark_near_wall[0, 1]:
            avoid_angle = detect_wall(shark_rays_coords) / (
                        np.minimum(shark_distance_to_wall[0, 0], shark_distance_to_wall[0, 1]) - fish_graphic_radius)
            # print(np.rad2deg(avoid_angle))
            if avoid_angle == 0:
                shark_orientations = get_direction(shark_coords[0], fish_coords[closest_fish])
            else:
                shark_orientations += avoid_angle
        else:
            shark_orientations = get_direction(shark_coords[0], fish_coords[closest_fish])

    # Beräknar Global Alignment
    global_alignment_coeff = 1 / fish_count * np.linalg.norm(
        [np.sum(np.cos(fish_orientations)), np.sum(np.sin(fish_orientations))])

    # Beräknar clustering coefficent
    clustering_coeff = calculate_cluster_coeff(fish_coords, fish_interaction_radius, fish_count)

    # Kollar om närmaste fisk är inom murder radien
    if len(fish_coords) > 4:  # <- den if-satsen är för att stoppa crash vid få fiskar
        if calculate_distance(shark_coords, fish_coords[closest_fish])[
            0] < murder_radius:
            last_index = len(fish_coords) - 1  # Sista index som kommer försvinna efter den mördade fisken tas bort

            canvas.delete(fish_canvas_graphics[last_index])
            fish_coords = murder_fish_coords(closest_fish)  # Tar bort index i koordinaterna
            fish_orientations = murder_fish_orientations(closest_fish)  # Tar bort index i orientations
            fish_eaten_count += 1  # Lägg till en äten fisk
            fish_eaten.append((fish_eaten_count, t * time_step))  # Spara hur många fiskar som ätits och när
    else:
        break

    # Skriver Global Alignment och Cluster Coefficient längst upp till vänster i rutan
    canvas.itemconfig(global_alignment_canvas_text, text='Global Alignment: {:.3f}'.format(global_alignment_coeff))
    canvas.itemconfig(clustering_coeff_canvas_text, text='Global Clustering: {:.3f}'.format(clustering_coeff))

    tk.title('Iteration =' + str(t))
    tk.update()  # Update animation frame
    time.sleep(0.01)  # Wait between loops
fish_eaten = np.array(fish_eaten)  # Gör om till array för att kunna plotta
plt.plot(fish_eaten[:, 1], fish_eaten[:, 0])  # Plotta
plt.xlabel('Tid')
plt.ylabel('Antal fiskar ätna')
plt.show()
tk.mainloop()
