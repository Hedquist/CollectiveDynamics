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
ccolor = ['#1E1BB1', '#F0092C', '#F5F805', '#D80000', '#E87B00', '#9F68D3', '#4B934F', '#FFFFFF']

# Variabler
canvas_length = 100  # Storlek på ruta, från mitten till kant. En sida är alltså 2*l
time_step = 1  # Storlek tidssteg
simulation_iterations = 4000  # Antalet iterationer simulationen kör
wait_time = 0.05  # Väntetiden mellan varje iteration

# Fisk
fish_count = 1  # Antal fiskar
fish_graphic_radius = 3  # Radie av ritad cirkel
fish_interaction_radius = 30  # Interraktionsradie för fisk
fish_speed = 2  # Hastighet fiskar
fish_noise = 0.1  # Brus i vinkel

# Haj
shark_count = 2  # Antal hajar
shark_graphic_radius = 4  # Radie av ritad cirkel för hajar
shark_speed = 1.8  # Hajens fart
murder_radius = 2  # Hajen äter fiskar inom denna radie
fish_eaten = []  # Array med antal fiskar ätna som 0e element och när det blev äten som 1a element
fish_eaten_count = 0  # Antal fiskar ätna

visuals_on = True

# Start koordinater fiskar
fish_coords_file = 'fish_coords_initial.npy'
fish_orientations_file = 'fish_orientations_initial.npy'
if False:
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

shark_coords = np.array([[50,50],[50,40]])

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


def get_distance(a, b):  # Räknar ut avstånd mellan två punkter
    return np.minimum(
        np.sqrt(((a[0]) % (2 * canvas_length) - (b[0]) % (2 * canvas_length)) ** 2 + (
                (a[1]) % (2 * canvas_length) - (b[1]) % (2 * canvas_length)) ** 2),
        np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


def get_direction(coord1, coord2):  # Ger riktningen från coord1 till coord2 i radianer
    if np.sqrt((coord2[0] - coord1[0]) ** 2 + (coord2[1] - coord1[1]) ** 2) < np.sqrt(
            ((coord2[0]) % (2 * canvas_length) - (coord1[0]) % (2 * canvas_length)) ** 2 + (
                    (coord2[1]) % (2 * canvas_length) - (coord1[1]) % (2 * canvas_length)) ** 2):
        return np.arctan2(coord2[1] - coord1[1], coord2[0] - coord1[0])
    else:
        return np.arctan2((coord2[1]) % (2 * canvas_length) - (coord1[1]) % (2 * canvas_length),
                          (coord2[0]) % (2 * canvas_length) - (coord1[0]) % (2 * canvas_length))


def get_shark_direction(sharks, target_fish, index):  # Ger riktningen för hajar
    shark = sharks[index]
    avoidance = get_shark_avoidance(sharks, target_fish, index)
    if np.sqrt((target_fish[0] - shark[0]) ** 2 + (target_fish[1] - shark[1]) ** 2) < np.sqrt(
            ((target_fish[0]) % (2 * canvas_length) - (shark[0]) % (2 * canvas_length)) ** 2 + (
                    (target_fish[1]) % (2 * canvas_length) - (shark[1]) % (2 * canvas_length)) ** 2):

        return np.arctan2(target_fish[1] - shark[1], target_fish[0] - shark[0]) + avoidance
    else:

        return np.arctan2((target_fish[1]) % (2 * canvas_length) - (shark[1]) % (2 * canvas_length),
                          (target_fish[0]) % (2 * canvas_length) - (shark[0]) % (2 * canvas_length)) + avoidance


def get_shark_avoidance(sharks_coords, target_fish, index):
    distance_to_fish = get_distance(sharks_coords[index], target_fish) # Avstånd till target fish
    orientation = get_direction(sharks_coords[index], target_fish) # Vilkeln till target fish
    avoidance = 0
    print(index)
    for k in range(shark_count): # Tanken att det blir något average här mot alla hajar
        if k != index:
            distance_to_shark = get_distance(sharks_coords[index], sharks_coords[k]) # Avstånd från hajen till haj K
            angle_to_shark = get_direction(sharks_coords[index], sharks_coords[k]) # Vinkeln från hajen till haj K
            print(orientation)
            print(angle_to_shark)
            if orientation - angle_to_shark <= 0: # Vill att den ska svänga bort från den andra hajen
                # Osäker på hur mycket de olika faktornerna ska påverka
                avoidance = -1 / distance_to_shark * distance_to_fish + avoidance
            else:
                avoidance = 1 / distance_to_shark * distance_to_fish + avoidance

    print(avoidance)
    return avoidance


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
'''
# Skapar ett canvas textobjekt för global alignemnt coefficent
global_alignment_canvas_text = canvas.create_text(100, 20, text=1 / fish_count * np.linalg.norm(
    [np.sum(np.cos(fish_orientations)),
     np.sum(np.sin(fish_orientations))]))

# Skapar ett canvas textobjekt för clustering coefficent
clustering_coeff_canvas_text = canvas.create_text(100, 40,
                                                  text=calculate_cluster_coeff(fish_coords, fish_interaction_radius,
                                                                               fish_count))
'''
# Skapar ett canvas textobjekt för antalet fiskar
fish_count_canvas_text = canvas.create_text(100, 20,
                                            text=len(fish_coords))

# Loop för allt som ska ske varje tidssteg i simulationen
for t in range(simulation_iterations):
    fish_coords = update_position(fish_coords, fish_speed, fish_orientations)  # Uppdatera fiskposition
    shark_coords = update_position(shark_coords, shark_speed, shark_orientations)  # Uppdatera hajposition

    fish_orientations_old = np.copy(fish_orientations)  # Spara gamla orientations för Vicsek
    if len(fish_coords) == 0:
        break
    # Bestäm närmsta fisk
    shark_fish_distances = np.zeros((shark_count, len(fish_coords)))
    closest_fish = np.zeros(shark_count, dtype=int)
    for j in range(shark_count):
        if len(fish_coords) == 0:
            break
        shark_fish_distances[j] = calculate_distance(fish_coords, shark_coords[
            j])  # Räknar ut det kortaste avståndet mellan haj och varje fisk
        closest_fish[j] = np.argmin(shark_fish_distances[j, :])  # Index av fisk närmst haj

    for j in range(shark_count):
        # Updating animation coordinates haj
        canvas.coords(shark_canvas_graphics[j],
                      (shark_coords[j, 0] - shark_graphic_radius + canvas_length) * res / canvas_length / 2,
                      (shark_coords[j, 1] - shark_graphic_radius + canvas_length) * res / canvas_length / 2,
                      (shark_coords[j, 0] + shark_graphic_radius + canvas_length) * res / canvas_length / 2,
                      (shark_coords[
                           j, 1] + shark_graphic_radius + canvas_length) * res / canvas_length / 2, )

    for j in range(len(fish_coords)):
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

        closest_shark = np.argmin(shark_fish_distances[:, j])  # Hittar index för närmaste hajen
        if shark_fish_distances[closest_shark, j] < fish_interaction_radius:  # Om hajen är nära fisken, undvik hajen
            fish_orientations[j] = get_direction(shark_coords[closest_shark], fish_coords[j])
        else:  # Annars Vicsek-modellen
            fish_orientations[j] = np.angle(
                np.sum(np.exp(
                    fish_orientations_old[fish_in_interaction_radius] * 1j))) + fish_noise * np.random.uniform(
                -1 / 2, 1 / 2)

        #   Shark direction härifrån
        for i in range(shark_count):
            shark_orientations[i] = get_shark_direction(shark_coords, fish_coords[closest_fish[i]], i)
    '''
    # Beräknar Global Alignment
    global_alignment_coeff = 1 / fish_count * np.linalg.norm(
        [np.sum(np.cos(fish_orientations)), np.sum(np.sin(fish_orientations))])

    # Beräknar clustering coefficent
    clustering_coeff = calculate_cluster_coeff(fish_coords, fish_interaction_radius, fish_count)
'''
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
    canvas.itemconfig(fish_count_canvas_text, text='Antal Fiskar: {:.3f}'.format(len(fish_coords)))

    tk.title('Iteration =' + str(t))
    tk.update()  # Update animation frame
    time.sleep(wait_time)  # Wait between loops
fish_eaten = np.array(fish_eaten)  # Gör om till array för att kunna plotta
plt.plot(fish_eaten[:, 1], fish_eaten[:, 0])  # Plotta
plt.xlabel('Tid')
plt.ylabel('% av fiskar ätna')
plt.show()
tk.mainloop()
