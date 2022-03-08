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
fish_speed = 1  # Hastighet fiskar
time_step = 1  # Storlek tidssteg
simulation_iterations = 4000  # Antalet iterationer simulationen kör
fish_noise = 0.2  # Brus i vinkel
murder_radius = 5  # Hajen äter fiskar inom denna radie

shark_count = 1  # Antal hajar (kan bara vara 1 just nu...)
shark_speed = 0.85  # Hajens fart
fish_eaten = []  # Array med antal fiskar ätna som 0e element och när det blev äten som 1a element
fish_eaten_count = 0    # Antal fiskar ätna

fish_turn_speed = 0.015
shark_turn_speed = 0.05

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

fish_desired_orientations = fish_orientations.copy()  # Array med alla fiskars önskade riktning

# Startkoordinater hajar
shark_coords = np.column_stack((0.0, 0.0))  # Array med alla hajars x- och y-koord
shark_orientations = np.random.rand(shark_count) * 2 * np.pi  # Array med alla hajars riktning
shark_desired_orientations = shark_orientations.copy()  # Array med alla hajars önskade riktning

fish_canvas_graphics = []  # De synliga cirklarna som är fiskar sparas här
shark_canvas_graphics = []  # De synliga cirklarna som är hajar sparas här


def update_position(coords, speed, orientations):  # Uppdaterar en partikels position
    coords[:, 0] = (coords[:, 0] + speed * np.cos(orientations) * time_step + canvas_length) % (
            2 * canvas_length) - canvas_length
    coords[:, 1] = (coords[:, 1] + speed * np.sin(orientations) * time_step + canvas_length) % (
            2 * canvas_length) - canvas_length
    return coords


def calculate_distance(coords, coord):  # Räknar ut avstånd mellan punkterna coords och punkten coord
    return np.minimum(
        np.sqrt(((coords[:, 0]) % (2 * canvas_length) - (coord[0]) % (2 * canvas_length)) ** 2 + (
                (coords[:, 1]) % (2 * canvas_length) - (coord[1]) % (2 * canvas_length)) ** 2),
        np.sqrt((coords[:, 0] - coord[0]) ** 2 + (coords[:, 1] - coord[1]) ** 2))


def get_direction(coord1, coord2):  # Ger riktningen från coord1 till coord2 i radianer
    if np.sqrt((coord2[0] - coord1[0]) ** 2 + (coord2[1] - coord1[1]) ** 2) < np.sqrt(
            ((coord2[0]) % (2 * canvas_length) - (coord1[0]) % (2 * canvas_length)) ** 2 + (
                    (coord2[1]) % (2 * canvas_length) - (coord1[1]) % (2 * canvas_length)) ** 2):
        return np.arctan2(coord2[1] - coord1[1], coord2[0] - coord1[0])
    else:
        return np.arctan2((coord2[1]) % (2 * canvas_length) - (coord1[1]) % (2 * canvas_length),
                          (coord2[0]) % (2 * canvas_length) - (coord1[0]) % (2 * canvas_length))


def torque_turn(desired_orientation, current_orientation, turn_speed):
    relative_orientation = current_orientation - desired_orientation  # makes calculation easier
    if relative_orientation == 0:
        return desired_orientation  # if desired angle is equal to current angle, do nothing
    elif relative_orientation > 0:
        calc = current_orientation - (np.pi * turn_speed)  # turn speed of 1 means you can turn pi radians per tick,
        new_orientation = np.maximum(calc, desired_orientation)  # turn speed of 0 means no turning
    else:
        calc = current_orientation + (np.pi * turn_speed)
        new_orientation = np.minimum(calc, desired_orientation)  # This line prevents "overturning"
    return new_orientation


# uses torque_turn to update orientation array according to desired_orientation
def update_orientation(current_orientations, desired_orientations, turn_speed):
    for n in range(len(current_orientations)):
        current_orientations[n] = torque_turn(desired_orientations[n], current_orientations[n], turn_speed)
    return current_orientations


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
    # Uppdatera fiskarnas orientationer innan uppdatering av positioner
    fish_orientations = update_orientation(fish_orientations, fish_desired_orientations, fish_turn_speed)
    fish_coords = update_position(fish_coords, fish_speed, fish_orientations)  # Uppdatera fiskposition
    # Uppdatera hajarnas orientationer innan uppdatering av positioner
    shark_orientations = update_orientation(shark_orientations, shark_desired_orientations, shark_turn_speed)
    shark_coords = update_position(shark_coords, shark_speed, shark_orientations)  # Uppdatera hajposition
    shark_fish_distances = calculate_distance(fish_coords, shark_coords[
        0])  # Räknar ut det kortaste avståndet mellan haj och varje fisk

    closest_fish = np.argmin(shark_fish_distances)  # Index av fisk närmst haj

    # print(closest_fish)
    # print(shark_coords)

    for j in range(shark_count):
        # Updating animation coordinates haj
        canvas.coords(shark_canvas_graphics[j],
                      (shark_coords[j, 0] - fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                      (shark_coords[j, 1] - fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                      (shark_coords[j, 0] + fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                      (shark_coords[
                           j, 1] + fish_graphic_radius + canvas_length) * res / canvas_length / 2, )

    for j in range(len(fish_coords)):
        # Updating animation coordinates fisk
        canvas.coords(fish_canvas_graphics[j],
                      (fish_coords[j, 0] - fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                      (fish_coords[j, 1] - fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                      (fish_coords[j, 0] + fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                      (fish_coords[
                           j, 1] + fish_graphic_radius + canvas_length) * res / canvas_length / 2, )

        if j == closest_fish:
            canvas.itemconfig(fish_canvas_graphics[j], fill=ccolor[2])  # Byt färg på fisk närmst haj
        else:
            canvas.itemconfig(fish_canvas_graphics[j], fill=ccolor[0])

        inter_fish_distances = calculate_distance(fish_coords, fish_coords[
            j])  # Räknar ut avstånd mellan fisk j och alla andra fiskar

        fish_in_interaction_radius = inter_fish_distances < fish_interaction_radius  # Vilka fiskar är inom en fisks interraktionsradie

        if shark_fish_distances[j] < fish_interaction_radius:  # Om hajen är nära fisken, undvik hajen
            fish_desired_orientations[j] = get_direction(shark_coords[0], fish_coords[j])
        else:  # Annars Vicsek-modellen
            fish_desired_orientations[j] = np.angle(
                np.sum(np.exp(fish_orientations[fish_in_interaction_radius] * 1j))) + fish_noise * np.random.uniform(
                -1 / 2, 1 / 2)

        #   Shark direction härifrån (change 0 to variable when implementing more sharks!)
        shark_desired_orientations[0] = get_direction(shark_coords[0], fish_coords[closest_fish])

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
fish_eaten = np.array(fish_eaten) # Gör om till array för att kunna plotta
plt.plot(fish_eaten[:, 1], fish_eaten[:, 0]) # Plotta
plt.xlabel('Tid')
plt.ylabel('Antal fiskar ätna')
plt.show()
tk.mainloop()
