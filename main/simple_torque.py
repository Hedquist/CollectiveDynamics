import math
import numpy as np
from tkinter import *
import scipy
from scipy.spatial import *
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull
import time
from shapely.geometry import Polygon
from timeit import default_timer as timer

# Systemets parametrar
canvas_length = 200
time_step = 1
simulation_iterations = 1000


def main(fish_turn_speed, shark_turn_speed, visuals_on, seed):
    rng = np.random.default_rng(seed)  # Random Number Generator with fixed seed
    start = timer()  # Timer startas
    if visuals_on:
        res = 500  # Resolution of the animation
        tk = Tk()
        tk.geometry(str(int(res * 1.1)) + 'x' + str(int(res * 1.3)))
        tk.configure(background='white')
        canvas = Canvas(tk, bd=2, bg='white')  # Generate animation window
        tk.attributes('-topmost', 0)
        canvas.place(x=res / 20, y=res / 20, height=res, width=res)
        ccolor = ['#1E1BB1', '#F0092C', '#F5F805', '#D80000', '#E87B00', '#9F68D3', '#4B934F', '#FFFFFF']

    # Mutual parameters
    BL = 3  # Mutual unit
    arrow_length = BL  # Pillängd

    # Fiskars parameter
    fish_graphic_radius = BL  # Radius of agent
    fish_interaction_radius = 25  # Interaction radius
    fish_noise = 0.1  # Diffusional noise constant
    fish_count = 200  # Antal fiskar
    fish_speed = 2  # Fiskens fart

    # Haj parametrar
    shark_graphic_radius = BL
    shark_interaction_radius = 4 * fish_interaction_radius
    shark_count = 1  # Antal hajar (kan bara vara 1 just nu...)
    shark_speed = 0.9 * fish_speed  # Hajens fart
    murder_radius = 2 * shark_graphic_radius  # Hajen äter fiskar inom denna radie
    fish_eaten = []  # Array med antal fiskar ätna som 0e element och när det blev äten som 1a element
    fish_eaten_count = 0  # Antal fiskar ätna

    # Fiskens koordinater
    x = np.array(rng.random(fish_count) * 2 * canvas_length - canvas_length)
    y = np.array(rng.random(fish_count) * 2 * canvas_length - canvas_length)
    fish_coords = np.column_stack((x, y))
    fish_orientations = rng.random(fish_count) * 2 * np.pi  # orientations
    fish_desired_orientations = fish_orientations.copy()  # Array med alla fiskars önskade riktning

    # Startkoordinater hajar
    shark_coords = np.column_stack((0.0, 0.0))  # Array med alla hajars x- och y-koord
    shark_orientations = rng.random(shark_count) * 2 * np.pi  # Array med alla hajars riktning
    shark_desired_orientations = shark_orientations.copy()  # Array med alla hajars önskade riktning

    # Spawn fishes outside sharks murder radius
    spawn_dist = np.linalg.norm(shark_coords - fish_coords, axis=1)
    indices = np.where(spawn_dist < fish_interaction_radius + murder_radius)[0]
    for i in indices:
        while np.linalg.norm(shark_coords - fish_coords[i], axis=1) < fish_interaction_radius + murder_radius:
            fish_coords[i] = [canvas_length * (rng.random() * 2 - 1), canvas_length * (rng.random() * 2 - 1)]

    # # Fisk
    # fish_count = 200  # Antal fiskar
    # fish_graphic_radius = 3  # Radie av ritad cirkel
    # fish_interaction_radius = 25  # Interraktionsradie för fisk
    # fish_speed = 2  # Hastighet fiskar
    # fish_noise = 0.1  # Brus i vinkel
    #
    # shark_fish_relative_speed = 0.9  # Relativ hastighet mellan haj och fisk
    #
    # # Haj
    # shark_count = 1  # Antal hajar
    # shark_graphic_radius = fish_graphic_radius  # Radie av ritad cirkel för hajar
    # shark_interaction_radius = 4 * fish_interaction_radius
    # shark_speed = fish_speed * shark_fish_relative_speed  # Hajens fart
    # murder_radius = 2*shark_graphic_radius  # Hajen äter fiskar inom denna radie
    # fish_eaten = []  # Array med antal fiskar ätna som 0e element och när det blev äten som 1a element
    # fish_eaten_count = 0  # Antal fiskar ätna
    #
    # fish_turn_speed = fish_turn_speed
    # shark_turn_speed = shark_turn_speed
    #
    # # Start koordinater fiskar
    # fish_coords_file = 'fish_coords_initial.npy'
    # fish_orientations_file = 'fish_orientations_initial.npy'
    # if True:
    #     x = rng.uniform(-canvas_length, canvas_length, fish_count) # x coordinates
    #     y = rng.uniform(-canvas_length, canvas_length, fish_count)  # y coordinates
    #     fish_orientations = rng.uniform(0, 2*np.pi, fish_count)  # orientations
    #     fish_coords = np.column_stack((x, y))
    #     np.save(fish_coords_file, fish_coords)
    #     np.save(fish_orientations_file, fish_orientations)
    # else:
    #     fish_coords = np.load(fish_coords_file)  # Array med alla fiskars x- och y-koord
    #     fish_orientations = np.load(fish_orientations_file)  # Array med alla fiskars riktning
    #
    # fish_desired_orientations = fish_orientations.copy()  # Array med alla fiskars önskade riktning
    #
    # # Startkoordinater hajar
    # shark_coords = np.column_stack((0.0, 0.0))  # Array med alla hajars x- och y-koord
    # shark_orientations = rng.uniform(0, 2*np.pi, shark_count) # Array med alla hajars riktning
    # shark_desired_orientations = shark_orientations.copy()  # Array med alla hajars önskade riktning
    fish_canvas_graphics = []  # De synliga cirklarna som är fiskar sparas här
    shark_canvas_graphics = []  # De synliga cirklarna som är hajar sparas här

    # fish_eaten = []  # Array med antal fiskar ätna som 0e element och när det blev äten som 1a element
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
        v = np.array([np.cos(desired_orientation), np.sin(desired_orientation)])
        w = np.array([np.cos(current_orientation), np.sin(current_orientation)])
        A = np.cross(v, w)
        relative_orientation = math.asin(np.linalg.norm(A)) * np.sign(A)
        if math.fabs(A) <= 1e-3 and math.fabs(np.dot(v, w) + 1) <= 1e-3:
            relative_orientation = np.pi
        if math.fabs(relative_orientation) <= turn_speed * np.pi and np.dot(v, w) >= (turn_speed * (-2) + 1):
            return desired_orientation  # if desired angle is equal to current angle, do nothing

        # turn speed of 1 means you can turn pi radians per tick,
        calc = current_orientation - (np.pi * turn_speed) * np.sign(relative_orientation)
        if calc > np.pi:
            calc -= 2 * np.pi
        elif calc < -np.pi:
            calc += 2 * np.pi
        return calc

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

    def volume_extraction(coords, radius):
        distances = scipy.spatial.distance.cdist(coords, coords)
        overlap = distances < 2 * radius
        for i in range(len(distances)):
            overlap[i, i] = False
        overlap[np.tril_indices(len(distances), 1)] = False
        index1 = np.where(overlap == True)[0]
        index2 = np.where(overlap == True)[1]
        for j in range(len(index1)):
            coords[i][0] = coords[i][0] + (distances[index1[j], index2[j]] - 2 * radius) * np.cos(
                get_direction(fish_coords[index2[j]], fish_coords[index1[j]])) / 2
            coords[i][1] = coords[i][1] + (distances[index1[j], index2[j]] - 2 * radius) * np.sin(
                get_direction(fish_coords[index2[j]], fish_coords[index1[j]])) / 2
        return coords

    if visuals_on:
        for j in range(shark_count):  # Skapar cirklar för hajar
            shark_canvas_graphics.append(
                canvas.create_oval((shark_coords[j, 0] - fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                                   (shark_coords[j, 1] - fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                                   (shark_coords[j, 0] + fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                                   (shark_coords[j, 1] + fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                                   outline='#f0650c', fill='#f0650c'))
        for j in range(fish_count):  # Skapar cirklar för fiskar
            fish_canvas_graphics.append(
                canvas.create_oval((fish_coords[j, 0] - fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                                   (fish_coords[j, 1] - fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                                   (fish_coords[j, 0] + fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                                   (fish_coords[j, 1] + fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                                   outline='#0994da', fill='#0994da'))

    if visuals_on:
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

        for j in range(len(fish_coords)):
            # # Overlapp fishes
            fish_distances = calculate_distance(fish_coords, fish_coords[j])
            angle = np.arctan2(fish_coords[:, 1] - fish_coords[j, 1],
                               fish_coords[:, 0] - fish_coords[j, 0])  # Directions of others array from the particle
            overlap = fish_distances < (2 * fish_graphic_radius)  # Applying
            overlap[j] = False  # area extraction
            for ind in np.where(overlap)[0]:
                fish_coords[j, 0] = fish_coords[j, 0] + (fish_distances[ind] - 2 * fish_graphic_radius) * np.cos(
                    angle[ind]) / 2
                fish_coords[j, 1] = fish_coords[j, 1] + (fish_distances[ind] - 2 * fish_graphic_radius) * np.sin(
                    angle[ind]) / 2
                fish_coords[ind] = fish_coords[ind] - (fish_distances[ind] - 2 * fish_graphic_radius) * np.cos(
                    angle[ind]) / 2
                fish_coords[ind] = fish_coords[ind] - (fish_distances[ind] - 2 * fish_graphic_radius) * np.sin(
                    angle[ind]) / 2

        closest_fish = np.argmin(shark_fish_distances)  # Index av fisk närmst haj

        # print(closest_fish)
        # print(shark_coords)

        if visuals_on:
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
                    canvas.itemconfig(fish_canvas_graphics[j], fill='#0994da')

        for j in range(len(fish_coords)):
            inter_fish_distances = calculate_distance(fish_coords, fish_coords[
                j])  # Räknar ut avstånd mellan fisk j och alla andra fiskar

            fish_in_interaction_radius = inter_fish_distances < fish_interaction_radius  # Vilka fiskar är inom en fisks interraktionsradie

            if shark_fish_distances[j] < fish_interaction_radius:  # Om hajen är nära fisken, undvik hajen
                fish_desired_orientations[j] = get_direction(shark_coords[0], fish_coords[j])
            else:  # Annars Vicsek-modellen
                fish_desired_orientations[j] = np.angle(
                    np.sum(np.exp(fish_orientations[fish_in_interaction_radius] * 1j))) + fish_noise * rng.uniform(
                    -1 / 2, 1 / 2)

            if shark_fish_distances[closest_fish] <= shark_interaction_radius:
                #   Shark direction härifrån (change 0 to variable when implementing more sharks!)
                shark_desired_orientations[0] = get_direction(shark_coords[0], fish_coords[closest_fish])

        if visuals_on:
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

                if visuals_on:
                    canvas.delete(fish_canvas_graphics[last_index])
                fish_coords = murder_fish_coords(closest_fish)  # Tar bort index i koordinaterna
                fish_orientations = murder_fish_orientations(closest_fish)  # Tar bort index i orientations
                fish_eaten_count += 1  # Lägg till en äten fisk
                fish_eaten.append((fish_eaten_count, t * time_step))  # Spara hur många fiskar som ätits och när
        else:
            break

        if visuals_on:
            # Skriver Global Alignment och Cluster Coefficient längst upp till vänster i rutan
            canvas.itemconfig(global_alignment_canvas_text, text='Global Alignment: {:.3f}'.format(global_alignment_coeff))
            canvas.itemconfig(clustering_coeff_canvas_text, text='Global Clustering: {:.3f}'.format(clustering_coeff))

            tk.title('Iteration =' + str(t))
            tk.update()  # Update animation frame
            time.sleep(0.01)  # Wait between loops

    if __name__ == "__main__":
        fish_eaten = np.array(fish_eaten)  # Gör om till array för att kunna plotta
        plt.plot(fish_eaten[:, 1], fish_eaten[:, 0])  # Plotta
        plt.xlabel('Tid')
        plt.ylabel('Antal fiskar ätna')
        plt.show()
    if visuals_on:
        tk.mainloop()

    return fish_eaten_count
    # end main()


if __name__ == "__main__":
    main(0.035, 0.05, True, 0)
