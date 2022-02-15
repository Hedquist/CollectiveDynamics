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

canvas = Canvas(tk, bd=2)  # Generate animation window
tk.attributes('-topmost', 0)
canvas.place(x=res / 20, y=res / 20, height=res, width=res)
ccolor = ['#17888E', '#C1D02B', '#9E00C9', '#D80000', '#E87B00', '#9F68D3', '#4B934F']

# Variabler
fish_count = 50  # Antal fiskar
canvas_length = 100  # Storlek på ruta, från mitten till kant. En sida är alltså 2*l
fish_graphic_radius = 4  # Radie av ritad cirkel
fish_interaction_radius = 10  # Interraktionsradie för fisk
fish_speed = 2  # Hastighet fiskar
time_step = 1  # Storlek tidssteg
simulation_iterations = 100  # Antalet iterationer simulationen kör
fish_noise = 0.1  # Brus i vinkel

shark_count = 1  # Antal hajar (kan bara vara 1 just nu...)
shark_speed = 1.8  # Hajens fart

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

fish_canvas_graphics = []  # De synliga cirklarna som är fiskar sparas här
shark_canvas_graphics = []  # De synliga cirklarna som är hajar sparas här


def update_position(coords, speed, orientations, time_step):  # Uppdaterar en partikels position
    coords[:, 0] = (coords[:, 0] + speed * np.cos(orientations) * time_step + canvas_length) % (
            2 * canvas_length) - canvas_length
    coords[:, 1] = (coords[:, 1] + speed * np.sin(orientations) * time_step + canvas_length) % (
            2 * canvas_length) - canvas_length
    return coords


def calculate_distance(r1, r2, l):  # Doesnt work, change argument names
    return np.minimum(
        np.sqrt(((r1[0]) % (2 * l) - (r2[0]) % (2 * l)) ** 2 + ((r1[1]) % (2 * l) - (r2[1]) % (2 * l)) ** 2),
        np.sqrt((r1[0] - r2[0]) ** 2 + (r1[1] - r2[1]) ** 2))


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
    fish_coords = update_position(fish_coords, fish_speed, fish_orientations, time_step)  # Uppdatera fiskposition
    shark_coords = update_position(shark_coords, shark_speed, shark_orientations, time_step)  # Uppdatera hajposition
    shark_fish_distances = np.minimum(np.sqrt(
        ((fish_coords[:, 0]) % (2 * canvas_length) - shark_coords[0, 0] % (2 * canvas_length)) ** 2 + (
                (fish_coords[:, 1]) % (2 * canvas_length) - shark_coords[0, 1] % (2 * canvas_length)) ** 2),
        np.sqrt((fish_coords[:, 0] - shark_coords[0, 0]) ** 2 + (
                fish_coords[:, 1] - shark_coords[
            0, 1]) ** 2))  # Räknar ut det kortaste avståndet mellan haj och varje fisk

    closest_fish = np.where(shark_fish_distances == np.amin(shark_fish_distances))[0]  # Index av fisk närmst haj

    for j in range(shark_count):
        # Updating animation coordinates haj
        canvas.coords(shark_canvas_graphics[j],
                      (shark_coords[j, 0] - fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                      (shark_coords[j, 1] - fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                      (shark_coords[j, 0] + fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                      (shark_coords[
                           j, 1] + fish_graphic_radius + canvas_length) * res / canvas_length / 2, )

    for j in range(fish_count):
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

        inter_fish_distances = np.minimum(np.sqrt(
            ((fish_coords[:, 0]) % (2 * canvas_length) - (fish_coords[j, 0]) % (2 * canvas_length)) ** 2 + (
                    (fish_coords[:, 1]) % (2 * canvas_length) - (fish_coords[j, 1]) % (2 * canvas_length)) ** 2),
            np.sqrt((fish_coords[:, 0] - fish_coords[j, 0]) ** 2 + (fish_coords[:, 1] - fish_coords[j, 1]) ** 2))
        fish_in_interaction_radius = inter_fish_distances < fish_interaction_radius  # Vilka fiskar är inom en fisks interraktionsradie

        if shark_fish_distances[j] < fish_interaction_radius:  # Om hajen är nära fisken, undvik hajen
            if np.sqrt(((fish_coords[j, 0]) % (2 * canvas_length) - shark_coords[0, 0] % (2 * canvas_length)) ** 2 + (
                    (fish_coords[j, 1]) % (2 * canvas_length) - shark_coords[0, 1] % (
                    2 * canvas_length)) ** 2) < np.sqrt(
                (fish_coords[j, 0] - shark_coords[0, 0]) ** 2 + (fish_coords[j, 1] - shark_coords[
                    0, 1]) ** 2):  # Om hajen är närmst genom väggen: undvik "genom väggen"
                fish_orientations[j] = np.arctan2(
                    (fish_coords[j, 1]) % (2 * canvas_length) - shark_coords[0, 1] % (2 * canvas_length),
                    (fish_coords[j, 0]) % (2 * canvas_length) - shark_coords[0, 0] % (2 * canvas_length))
            else:  # Om hajen är närmst inom rutan: undvik inom rutan
                fish_orientations[j] = np.arctan2((fish_coords[j, 1] - shark_coords[0, 1]),
                                                  (fish_coords[j, 0] - shark_coords[0, 0]))
        else:  # Annars Vicsek-modellen
            fish_orientations[j] = np.angle(
                np.sum(np.exp(fish_orientations[fish_in_interaction_radius] * 1j))) + fish_noise * np.random.uniform(
                -1 / 2, 1 / 2)

        #   Shark direction härifrån
        if np.sqrt(((fish_coords[closest_fish, 0]) % (2 * canvas_length) - shark_coords[0, 0] % (
                2 * canvas_length)) ** 2 + (
                           (fish_coords[closest_fish, 1]) % (2 * canvas_length) - shark_coords[0, 1] % (
                           2 * canvas_length)) ** 2) < np.sqrt(
            (fish_coords[closest_fish, 0] - shark_coords[0, 0]) ** 2 + (fish_coords[closest_fish, 1] - shark_coords[
                0, 1]) ** 2):  # Om kortaste vägen till fisken (för hajen) är genom väggen: sikta genom väggen
            shark_orientations = np.arctan2(
                (fish_coords[closest_fish, 1]) % (2 * canvas_length) - shark_coords[0, 1] % (2 * canvas_length),
                (fish_coords[closest_fish, 0]) % (2 * canvas_length) - shark_coords[0, 0] % (2 * canvas_length))
        else:  # Om kortaste vägen till fisken (för hajen) är inom rutan: sikta inom rutan
            shark_orientations = np.arctan2((fish_coords[closest_fish, 1] - shark_coords[0, 1]),
                                            (fish_coords[closest_fish, 0] - shark_coords[0, 0]))

    # Beräknar Global Alignment
    global_alignment_coeff = 1 / fish_count * np.linalg.norm(
        [np.sum(np.cos(fish_orientations)), np.sum(np.sin(fish_orientations))])

    # Beräknar clustering coefficent
    clustering_coeff = calculate_cluster_coeff(fish_coords, fish_interaction_radius, fish_count)

    # Skriver Global Alignment och Cluster Coefficient längst upp till vänster i rutan
    canvas.itemconfig(global_alignment_canvas_text, text='Global Alignment: {:.3f}'.format(global_alignment_coeff))
    canvas.itemconfig(clustering_coeff_canvas_text, text='Global Clustering: {:.3f}'.format(clustering_coeff))

    tk.title('Iteration =' + str(t))
    tk.update()  # Update animation frame
    time.sleep(0.01)  # Wait between loops
tk.mainloop()

