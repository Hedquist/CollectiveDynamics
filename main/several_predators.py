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
canvas_length = 500  # Storlek på ruta, från mitten till kant. En sida är alltså 2*l
time_step = 1  # Storlek tidssteg
simulation_iterations = 4000  # Antalet iterationer simulationen kör
wait_time = 0.001  # Väntetiden mellan varje iteration

# Fisk
fish_count = 5  # Antal fiskar
fish_graphic_radius = 8  # Radie av ritad cirkel
fish_interaction_radius = 30  # Interraktionsradie för fisk
fish_speed = 2  # Hastighet fiskar
fish_noise = 0.1  # Brus i vinkel

shark_fish_relative_speed = 0.9  # Relativ hastighet mellan haj och fisk

# Haj
shark_count = 4  # Antal hajar
shark_graphic_radius = 10  # Radie av ritad cirkel för hajar
shark_speed = fish_speed * shark_fish_relative_speed  # Hajens fart
shark_interaction_radius = 100  # Radien som Hajen "ser"
murder_radius = 10  # Hajen äter fiskar inom denna radie
fish_eaten = []  # Array med antal fiskar ätna som 0e element och när det blev äten som 1a element
fish_eaten_count = 0  # Antal fiskar ätna

# Start koordinater fiskar
fish_coords_file = 'fish_coords_initial.npy'
fish_orientations_file = 'fish_orientations_initial.npy'
if True:
    x = (np.random.rand(fish_count) * 2 * canvas_length - canvas_length) / 4  # x coordinates
    y = (np.random.rand(fish_count) * 2 * canvas_length - canvas_length) / 4  # y coordinates
    fish_orientations = np.random.rand(fish_count) * 2 * np.pi  # orientations
    fish_coords = np.column_stack((x, y))
    np.save(fish_coords_file, fish_coords)
    np.save(fish_orientations_file, fish_orientations)
else:
    fish_coords = np.load(fish_coords_file)  # Array med alla fiskars x- och y-koord
    fish_orientations = np.load(fish_orientations_file)  # Array med alla fiskars riktning

# Startkoordinater hajar
shark_x = (np.random.rand(shark_count) * 2 * canvas_length - canvas_length) / 2  # shark x coordinates
shark_y = (np.random.rand(shark_count) * 2 * canvas_length - canvas_length) / 2  # shark y coordinates
shark_coords = np.column_stack((shark_x, shark_y))  # Array med alla hajars x- och y-koord
shark_orientations = np.random.rand(shark_count) * 2 * np.pi  # Array med alla hajars riktning

# shark_coords = np.array([[200.0, 0.0], [-200.0, 0.0]])

fish_canvas_graphics = []  # De synliga cirklarna som är fiskar sparas här
shark_canvas_graphics = []  # De synliga cirklarna som är hajar sparas här

'''def update_position(coords, speed, orientations, time_step):  # Uppdaterar en partikels position
    coords[:, 0] = (coords[:, 0] + speed * np.cos(orientations) * time_step + canvas_length) % (
            2 * canvas_length) - canvas_length
    coords[:, 1] = (coords[:, 1] + speed * np.sin(orientations) * time_step + canvas_length) % (
            2 * canvas_length) - canvas_length
    return coords'''


def update_position(coords, speed, orientations, time_step):  # Uppdaterar position
    coords[:, 0] = coords[:, 0] + speed * np.cos(orientations) * time_step
    coords[:, 1] = coords[:, 1] + speed * np.sin(orientations) * time_step
    return coords


def bounce_angle(coords, orientations):  # Ändra vinkel om partikeln åker ur rutan
    for i in range(len(orientations)):
        if coords[i, 0] < -canvas_length:
            orientations[i] = np.pi - orientations[i]
        elif coords[i, 0] > canvas_length:
            orientations[i] = np.pi - orientations[i]
        if coords[i, 1] < -canvas_length:
            orientations[i] = -orientations[i]
        elif coords[i, 1] > canvas_length:
            orientations[i] = -orientations[i]
    return orientations


def bounce_pos(coords):  # Flytta in partikel i rutan om den åker utanför
    for coord in coords:
        if coord[0] < -canvas_length:
            coord[0] = -canvas_length - (coord[0] + canvas_length)
        elif coord[0] > canvas_length:
            coord[0] = canvas_length - (coord[0] - canvas_length)
        if coord[1] < -canvas_length:
            coord[1] = -canvas_length - (coord[1] + canvas_length)
        elif coord[1] > canvas_length:
            coord[1] = canvas_length - (coord[1] - canvas_length)
    return coords


def calculate_distance(coords, coord):  # Räknar ut avstånd mellan punkterna coords och punkten coord
    return np.minimum(
        np.sqrt(((coords[:, 0]) % (2 * canvas_length) - (coord[0]) % (2 * canvas_length)) ** 2 + (
                (coords[:, 1]) % (2 * canvas_length) - (coord[1]) % (2 * canvas_length)) ** 2),
        np.sqrt((coords[:, 0] - coord[0]) ** 2 + (coords[:, 1] - coord[1]) ** 2))


def get_distance(a, b):  # Räknar ut avstånd mellan två punkter
    return np.minimum(
        np.sqrt(((a[0]) % (2 * canvas_length) - (b[0]) % (2 * canvas_length)) ** 2 + (
                (a[1]) % (2 * canvas_length) - (b[1]) % (2 * canvas_length)) ** 2),
        np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


def get_direction(coord1, coord2):  # Ger riktningen från coord1 till coord2 i radianer

    return np.arctan2(coord2[1] - coord1[1], coord2[0] - coord1[0])


def get_shark_direction(sharks, target_fish, index):  # Ger riktningen för hajar
    shark = sharks[index]
    avoidance = get_shark_avoidance(sharks, target_fish, index) #Ta bort 2:an och understrecket så kör den gamla

    return np.arctan2(target_fish[1] - shark[1], target_fish[0] - shark[0]) + avoidance


def get_shark_avoidance(sharks_coords, target_fish, index):
    distance_to_fish = get_distance(sharks_coords[index], target_fish)  # Avstånd till target fish
    orientation = get_direction(sharks_coords[index], target_fish)  # Vilkeln till target fish
    avoidance = 0
    for k in range(shark_count):  # Tanken att det blir något average här mot alla hajar
        if k != index:
            distance_to_shark = get_distance(sharks_coords[index], sharks_coords[k])  # Avstånd från hajen till haj K
            angle_to_shark = get_direction(sharks_coords[index], sharks_coords[k])  # Vinkeln från hajen till haj K
            # print(index)
            # print(angle_to_shark)
            if orientation - angle_to_shark < 0:  # Vill att den ska svänga bort från den andra hajen
                # Osäker på hur mycket de olika faktornerna ska påverka
                avoidance = np.maximum(-1 / (np.maximum(distance_to_shark, 1)) * \
                                       np.minimum(distance_to_fish, 100), -1) + avoidance
            elif orientation - angle_to_shark > 0:
                avoidance = np.minimum(1 / (np.maximum(distance_to_shark, 1)) *
                                       np.minimum(distance_to_fish, 100), 1) + avoidance
    # if distance_to_shark < 6 * fish_interaction_radius:
    # avoidance = (orientation + angle_to_shark) / 2

    # cos(orientation - angle_to_shark) * distance_to_fish(normerad för att gå från 0 till 1) / distance_to_shark(normerad)

    return avoidance / (len(shark_coords) - 1)


def get_shark_avoidance_2(sharks_coords, target_fish, index):
    distance_to_fish = get_distance(sharks_coords[index], target_fish)  # Avstånd till target fish
    orientation = get_direction(sharks_coords[index], target_fish)  # Vilkeln till target fish
    avoidance = 0
    for k in range(shark_count):  # Tanken att det blir något average här mot alla hajar
        if k != index:
            distance_to_shark = get_distance(sharks_coords[index], sharks_coords[k])  # Avstånd från hajen till haj K
            angle_to_shark = get_direction(sharks_coords[index], sharks_coords[k])  # Vinkeln från hajen till haj K
            # print(index)
            # print(angle_to_shark)
            avoidance = np.cos(orientation - angle_to_shark) * (distance_to_fish / canvas_length) / \
            (distance_to_shark / canvas_length) + avoidance

    # cos(orientation - angle_to_shark) * distance_to_fish(normerad för att gå från 0 till 1) / distance_to_shark(normerad)
    print(avoidance / (len(shark_coords) - 1))
    return avoidance / (len(shark_coords) - 1)


def murder_fish_coords(dead_fish_index):  # Tar bort fisk-coord som blivit uppäten
    new_fish_coords = np.delete(fish_coords, dead_fish_index, 0)
    return new_fish_coords


def murder_fish_orientations(dead_fish_index):  # Tar bort fisk-vinkel som blivit uppäten
    new_fish_orientations = np.delete(fish_orientations, dead_fish_index)
    return new_fish_orientations


def predict_position(fish_coord, fish_orientation,
                     distance_to_fish):  # Förutspår positionen av en fisk beroende på avstånd till fisk
    predicted_fish_position = update_position(np.array([fish_coord]), fish_speed, fish_orientation,
                                              distance_to_fish / shark_speed * shark_fish_relative_speed * 0.9)
    predicted_fish_position = bounce_pos((predicted_fish_position))
    return predicted_fish_position[0]


def get_fish_avoidance(fish_index, fish_near_shark, shark_near_fish):
    avoidance = 0
    count = 0

    for i in range(len(fish_near_shark)):
        if fish_near_shark[i] == fish_index:
            avoidance = get_direction(shark_coords[shark_near_fish[i]], fish_coords[fish_index]) + avoidance
            count += 1
    return avoidance / count


def get_target_fish(sharks_fish_distance, index):
    target_fish = get_target_fish_2(sharks_fish_distance, index)

    return target_fish


def get_target_fish_2(sharks_fish_distance, index):
    current_fish = 0
    min_distance = 100000000000000000000

    for i in range(len(sharks_fish_distance[0])):

        next = sum(sharks_fish_distance[:, i])

        if next < min_distance:
            min_distance = next
            current_fish = i

    return current_fish


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
if visuals_on:
    fish_count_canvas_text = canvas.create_text(100, 20,
                                                text=len(fish_coords))

# Loop för allt som ska ske varje tidssteg i simulationen
for t in range(simulation_iterations):
    fish_coords = update_position(fish_coords, fish_speed, fish_orientations, time_step)  # Uppdatera fiskposition
    shark_coords = update_position(shark_coords, shark_speed, shark_orientations, time_step)  # Uppdatera hajposition

    # Fixar så att fiskar inte simmar ur bild
    fish_orientations = bounce_angle(fish_coords, fish_orientations)
    fish_coords = bounce_pos(fish_coords)

    shark_orientations = bounce_angle(shark_coords, shark_orientations)
    shark_coords = bounce_pos(shark_coords)

    fish_orientations_old = np.copy(fish_orientations)  # Spara gamla orientations för Vicsek
    if len(fish_coords) == 0:
        break

    fish_fish_distance_matrix = scipy.spatial.distance.cdist(fish_coords,
                                                             fish_coords)  # Skapa matris med fisk-till-fisk-avstånd
    shark_fish_distance_matrix = scipy.spatial.distance.cdist(shark_coords,
                                                              fish_coords)  # Skapa matris med haj-till-fisk-avstånd

    # print(shark_fish_distance_matrix)
    shark_near_fish_index = np.where(shark_fish_distance_matrix < fish_interaction_radius)[0]
    # print(np.where(shark_fish_distance_matrix < fish_interaction_radius)[0])

    fish_near_shark_index = np.where(shark_fish_distance_matrix < fish_interaction_radius)[1]
    # print(np.where(shark_fish_distance_matrix < fish_interaction_radius)[1])
    # Bestäm närmsta fisk

    closest_fish = np.zeros(shark_count, dtype=int)
    for j in range(shark_count):
        closest_fish[j] = np.argmin(shark_fish_distance_matrix[j, :])  # Index av fisk närmst haj

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


            canvas.itemconfig(fish_canvas_graphics[j], fill=ccolor[0])

        # inter_fish_distances = calculate_distance(fish_coords, fish_coords[j])  # Räknar ut avstånd mellan fisk j
        # och alla andra fiskar

        # Vilka fiskar är inom en fisks interraktionsradie
        fish_in_interaction_radius = fish_fish_distance_matrix[:, j] < fish_interaction_radius

        # closest_shark = np.argmin(shark_fish_distance_matrix[:, j])  # Hittar index för närmaste hajen
        if j in fish_near_shark_index:  # Om hajen är nära fisken, undvik hajen
            fish_orientations[j] = get_fish_avoidance(j, fish_near_shark_index, shark_near_fish_index)
        else:  # Annars Vicsek-modellen
            fish_orientations[j] = np.angle(
                np.sum(np.exp(
                    fish_orientations_old[fish_in_interaction_radius] * 1j))) + fish_noise * np.random.uniform(
                -1 / 2, 1 / 2)

    #   Shark direction härifrån
    for i in range(shark_count):
        target_fish = get_target_fish(shark_fish_distance_matrix, i)
        predicted_fish_coord = predict_position(fish_coords[target_fish], fish_orientations[target_fish],
                                                shark_fish_distance_matrix[i, target_fish])
        # shark_orientations[i] = get_direction(shark_coords[i], predicted_fish_coord)
        shark_orientations[i] = get_shark_direction(shark_coords, predicted_fish_coord, i)

        canvas.itemconfig(fish_canvas_graphics[target_fish], fill=ccolor[2])
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
            if len(fish_coords) == 0:
                break
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
# plt.show()

# if visuals_on:
# tk.mainloop()
