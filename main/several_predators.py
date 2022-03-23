import numpy as np
from tkinter import *
from scipy.spatial import *
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull
import scipy
import time
from shapely.geometry import Polygon
from timeit import default_timer as timer

start = timer()  # Timer startas
visuals_on = True  # Välj om simulationen ska visas eller ej.
if visuals_on:
    res = 500  # Resolution of the animation
    tk = Tk()
    tk.geometry(str(int(res * 1.1)) + 'x' + str(int(res * 1.3)))
    tk.configure(background='white')
    canvas = Canvas(tk, bd=2, bg='white')  # Generate animation window
    tk.attributes('-topmost', 0)
    canvas.place(x=res / 20, y=res / 20, height=res, width=res)
    ccolor = ['#1E1BB1', '#F0092C', '#F5F805', '#D80000', '#E87B00', '#9F68D3', '#4B934F', '#FFFFFF']
# Variabler
canvas_length = 200  # Storlek på ruta, från mitten till kant. En sida är alltså 2*l
time_step = 1  # Storlek tidssteg
simulation_iterations = 4000  # Antalet iterationer simulationen kör
wait_time = 0.01  # Väntetiden mellan varje iteration

# Fisk
fish_count = 1  # Antal fiskar
fish_graphic_radius = 2  # Radie av ritad cirkel
fish_interaction_radius = 10  # Interraktionsradie för fisk
fish_speed = 2  # Hastighet fiskar
fish_noise = 0.1  # Brus i vinkel

shark_fish_relative_speed = 0.9  # Relativ hastighet mellan haj och fisk

# Haj
shark_count = 2  # Antal hajar
shark_graphic_radius = 4  # Radie av ritad cirkel för hajar
shark_speed = fish_speed * shark_fish_relative_speed  # Hajens fart
murder_radius = 2  # Hajen äter fiskar inom denna radie
fish_eaten = []  # Array med antal fiskar ätna som 0e element och när det blev äten som 1a element
fish_eaten_count = 0  # Antal fiskar ätna

# Start koordinater fiskar
fish_coords_file = 'fish_coords_initial.npy'
fish_orientations_file = 'fish_orientations_initial.npy'
if True:
    shark_closest_fish_distances = np.random.rand(fish_count) * 2 * canvas_length - canvas_length  # x coordinates
    y = np.random.rand(fish_count) * 2 * canvas_length - canvas_length  # y coordinates
    fish_orientations = np.random.rand(fish_count) * 2 * np.pi  # orientations
    fish_coords = np.column_stack((shark_closest_fish_distances, y))
    np.save(fish_coords_file, fish_coords)
    np.save(fish_orientations_file, fish_orientations)
else:
    fish_coords = np.load(fish_coords_file)  # Array med alla fiskars x- och y-koord
    fish_orientations = np.load(fish_orientations_file)  # Array med alla fiskars riktning

# Startkoordinater hajar
shark_x = np.random.rand(shark_count) * 2 * canvas_length - canvas_length  # shark x coordinates
shark_y = np.random.rand(shark_count) * 2 * canvas_length - canvas_length  # shark y coordinates
shark_coords = np.column_stack((shark_x, shark_y))  # Array med alla hajars x- och y-koord
shark_orientations = np.random.rand(shark_count) * 2 * np.pi  # Array med alla hajars riktning
fish_canvas_graphics = []  # De synliga cirklarna som är fiskar sparas här
shark_canvas_graphics = []  # De synliga cirklarna som är hajar sparas här
tmp = scipy.spatial.distance.cdist(fish_coords, shark_coords)
print(tmp, tmp.shape)


def update_position(coords, speed, orientations, time_step):  # Uppdaterar en partikels position
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


'''
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
'''


def murder_fish_coords(dead_fish_index):  # Tar bort fisk som blivit uppäten
    new_fish_coords = np.delete(fish_coords, dead_fish_index, 0)
    return new_fish_coords


def murder_fish_orientations(dead_fish_index):
    new_fish_orientations = np.delete(fish_orientations, dead_fish_index)
    return new_fish_orientations


def predict_position(fish_coord, fish_orientation, distance_to_fish):
    predicted_fish_coords = update_position(np.array([fish_coord]), fish_speed, fish_orientation,
                                            distance_to_fish / shark_speed * shark_fish_relative_speed * 0.9)
    return predicted_fish_coords[0]


if visuals_on:
    for j in range(shark_count):  # Skapar cirklar för hajar
        shark_canvas_graphics.append(
            canvas.create_oval((shark_coords[j, 0] - shark_graphic_radius + canvas_length) * res / canvas_length / 2,
                               (shark_coords[j, 1] - shark_graphic_radius + canvas_length) * res / canvas_length / 2,
                               (shark_coords[j, 0] + shark_graphic_radius + canvas_length) * res / canvas_length / 2,
                               (shark_coords[j, 1] + shark_graphic_radius + canvas_length) * res / canvas_length / 2,
                               outline=ccolor[1], fill=ccolor[1]))
    for j in range(fish_count):  # Skapar cirklar för fiskar
        fish_canvas_graphics.append(
            canvas.create_oval((fish_coords[j, 0] - fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                               (fish_coords[j, 1] - fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                               (fish_coords[j, 0] + fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                               (fish_coords[j, 1] + fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                               outline=ccolor[0], fill=ccolor[0]))

    # Skapar ett canvas textobjekt för antalet fiskar
    fish_count_canvas_text = canvas.create_text(100, 20,
                                                text=len(fish_coords))
# Loop för allt som ska ske varje tidssteg i simulationen
for t in range(simulation_iterations):
    fish_coords = update_position(fish_coords, fish_speed, fish_orientations, time_step)  # Uppdatera fiskposition
    shark_coords = update_position(shark_coords, shark_speed, shark_orientations, time_step)  # Uppdatera hajposition
    fish_orientations_old = np.copy(fish_orientations)  # Spara gamla orientations för Vicsek
    # Bestäm närmsta fisk
    shark_fish_distances = np.zeros((shark_count, len(fish_coords)))
    closest_fish = np.zeros(shark_count, dtype=int)
    for j in range(shark_count):
        shark_fish_distances[j] = calculate_distance(fish_coords, shark_coords[
            j])  # Räknar ut det kortaste avståndet mellan haj och varje fisk
        closest_fish[j] = np.argmin(shark_fish_distances[j, :])  # Index av fisk närmst haj
    if visuals_on:
        for j in range(shark_count):
            # Updating animation coordinates haj
            canvas.coords(shark_canvas_graphics[j],
                          (shark_coords[j, 0] - shark_graphic_radius + canvas_length) * res / canvas_length / 2,
                          (shark_coords[j, 1] - shark_graphic_radius + canvas_length) * res / canvas_length / 2,
                          (shark_coords[j, 0] + shark_graphic_radius + canvas_length) * res / canvas_length / 2,
                          (shark_coords[
                               j, 1] + shark_graphic_radius + canvas_length) * res / canvas_length / 2, )
    for j in range(len(fish_coords)):
        if visuals_on:
            # Updating animation coordinates fisk
            canvas.coords(fish_canvas_graphics[j],
                          (fish_coords[j, 0] - fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                          (fish_coords[j, 1] - fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                          (fish_coords[j, 0] + fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                          (fish_coords[
                               j, 1] + fish_graphic_radius + canvas_length) * res / canvas_length / 2, )
            if j in closest_fish:
                canvas.itemconfig(fish_canvas_graphics[j], fill=ccolor[2])  # Byt färg på fisk närmst haj
            else:
                canvas.itemconfig(fish_canvas_graphics[j], fill=ccolor[0])
        inter_fish_distances = calculate_distance(fish_coords, fish_coords[
            j])  # Räknar ut avstånd mellan fisk j och alla andra fiskar
        # Vilka fiskar är inom en fisks interraktionsradie
        fish_in_interaction_radius = inter_fish_distances < fish_interaction_radius
        shark_near_fish = shark_fish_distances[:, j] < fish_interaction_radius
        if any(shark_near_fish):  # Om hajen är nära fisken, undvik hajen
            avoidance = 0
            for i in range(shark_count):
                if shark_near_fish[i]:
                    avoidance = get_direction(shark_coords[i], fish_coords[j]) + avoidance
            fish_orientations[j] = avoidance / sum(shark_near_fish)
        else:  # Annars Vicsek-modellen
            fish_orientations[j] = np.angle(
                np.sum(np.exp(
                    fish_orientations_old[fish_in_interaction_radius] * 1j))) + fish_noise * np.random.uniform(
                -1 / 2, 1 / 2)
    #   Shark direction härifrån
    for i in range(shark_count):
        predicted_fish_coord = predict_position(fish_coords[closest_fish[i]], fish_orientations[closest_fish[i]],
                                                shark_fish_distances[i, closest_fish[i]])
        shark_orientations[i] = get_direction(shark_coords[i], predicted_fish_coord)

    # Kollar om närmaste fisk är inom murder radien
    shark_closest_fish_distances = np.zeros(shark_count)  # Avstånd från varje haj till dess närmsta fisk
    # Haj äter fisk
    if len(fish_coords) > 0:  # <- den if-satsen är för att stoppa crash vid få fiskar
        for j in range(shark_count):
            # Räkna om vilken fisk som är närmst efter att fiskar ätits
            shark_fish_distances = np.zeros((shark_count, len(fish_coords)))
            closest_fish = np.zeros(shark_count, dtype=int)
            shark_fish_distances[j] = calculate_distance(fish_coords, shark_coords[
                j])  # Räknar ut det kortaste avståndet mellan haj och varje fisk
            closest_fish[j] = np.argmin(shark_fish_distances[j, :])  # Index av fisk närmst haj
            shark_closest_fish_distances[j] = \
                calculate_distance(np.array([shark_coords[j]]), fish_coords[closest_fish[j]])[
                    0]  # Avstånd från haj till närmsta fisk
            if shark_closest_fish_distances[j] < murder_radius:  # Allt som händer då en fisk blir äten
                last_index = len(fish_coords) - 1  # Sista index som kommer försvinna efter den mördade fisken tas bort
                if visuals_on:
                    canvas.delete(fish_canvas_graphics[last_index])  # Ta bort sista fisk-cirkeln i array
                fish_coords = murder_fish_coords(closest_fish[j])  # Tar bort index i koordinaterna
                fish_orientations = murder_fish_orientations(closest_fish[j])  # Tar bort index i orientations
                fish_eaten_count += 1 / fish_count * 100  # Lägg till en äten fisk
                fish_eaten.append((fish_eaten_count, t * time_step))  # Spara hur många fiskar som ätits och när
    else:
        break
    # Skriver Global Alignment och Cluster Coefficient längst upp till vänster i rutan
    # canvas.itemconfig(global_alignment_canvas_text, text='Global Alignment: {:.3f}'.format(global_alignment_coeff))
    # canvas.itemconfig(clustering_coeff_canvas_text, text='Global Clustering: {:.3f}'.format(clustering_coeff))
    if visuals_on:
        canvas.itemconfig(fish_count_canvas_text, text='Antal Fiskar: {:.3f}'.format(len(fish_coords)))
        tk.title('Iteration =' + str(t))
        tk.update()  # Update animation frame
        time.sleep(wait_time)  # Wait between loops
fish_eaten = np.array(fish_eaten)  # Gör om till array för att kunna plotta
plt.plot(fish_eaten[:, 1], fish_eaten[:, 0])  # Plotta
plt.xlabel('Tid')
plt.ylabel('% av fiskar ätna')
print("Time:", timer() - start)  # Skriver hur lång tid simulationen tog
plt.show()
if visuals_on:
    tk.mainloop()
