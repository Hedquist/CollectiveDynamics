import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk
from scipy.spatial.distance import *
import time
from timeit import default_timer as timer

# Systemets parameter
simulation_iterations = 500 # Simulation time
time_step = 1  # Time step
canvas_length = 200  # Size of box


def main(obst_type_main, row_main, col_main, obst_size_main, displacement_main, seed):
    visuals_on = True
    rng = np.random.default_rng(seed)

    if visuals_on:
        res = 600  # Resolution of the animation
        tk = Tk()
        tk.geometry(str(int(res * 1.1)) + 'x' + str(int(res * 1.3)))  # Set height x width window
        tk.configure(background='white')

        canvas = Canvas(tk, bd=2)  # Generate animation window
        tk.attributes('-topmost', 0)
        canvas.place(x=res / 20, y=res / 20, height=res, width=res)  # Place canvas with origin in x och y

        # Färger
        fish_color = '#2E86C1'
        shark_color = '#f0650c'
        closest_fish_color = '#9E00C9'
        black = '#000000'



    # Mutual parameters
    BL = 3 # Mutual unit
    arrow_length = BL  # Pillängd

    # Fiskars parameter
    fish_graphic_radius = BL  # Radius of agent
    fish_interaction_radius = 10  # Interaction radius
    fish_ray_radius = fish_interaction_radius/2 # Strållängd
    fish_noise = 0.1  # Diffusional noise constant
    fish_count = 200  # Antal fiskar
    fish_speed = 2  # Fiskens fart

    # Haj parametrar
    shark_graphic_radius = BL
    shark_ray_radius = fish_ray_radius + (shark_graphic_radius - fish_graphic_radius)
    shark_count = 1  # Antal hajar (kan bara vara 1 just nu...)
    shark_speed = 0.9 * fish_speed  # Hajens fart
    murder_radius = 2 * shark_graphic_radius  # Hajen äter fiskar inom denna radie
    fish_eaten_count = 0  # Antal fiskar ätna

    # Fiskens koordinater
    x = np.array(rng.random(fish_count) * 2 * canvas_length - canvas_length)
    y = np.array(rng.random(fish_count) * 2 * canvas_length - canvas_length)
    fish_coords = np.column_stack((x, y))
    fish_orientations = rng.random(fish_count) * 2 * np.pi  # orientations

    # Startkoordinater hajar
    shark_coords = np.column_stack((0.0, 0.0))  # Array med alla hajars x- och y-koord
    shark_orientations = rng.random(shark_count) * 2 * np.pi  # Array med alla hajars riktning

    # Spawn fishes outside sharks murder radius
    spawn_dist = np.linalg.norm(shark_coords - fish_coords, axis=1)
    indices = np.where(spawn_dist < fish_interaction_radius + murder_radius)[0]
    for i in indices:
        while np.linalg.norm(shark_coords- fish_coords[i], axis=1) < fish_interaction_radius + murder_radius :
            fish_coords[i] = [canvas_length*(rng.random() * 2 - 1), canvas_length * (rng.random() * 2 - 1)]

    # Rayscasting
    casted_rays = 6

    # Raycasting fishes
    fish_step_angle = 2 * np.arctan(fish_graphic_radius / fish_ray_radius)
    fish_FOV_angle = fish_step_angle * (casted_rays - 1)  # Field of view angle
    fish_half_FOV = fish_FOV_angle / 2

    # Rayscasting sharks
    shark_step_angle = 2 * np.arctan(shark_graphic_radius / shark_ray_radius)
    shark_FOV_angle = shark_step_angle * (casted_rays - 1)  # Field of view angle
    shark_half_FOV = shark_FOV_angle / 2

    fish_rays_coords = [[] for i in range(fish_count)]
    shark_rays_coords = []

    # Hinder info
    circ_obst_coords = []
    rect_obst_coords = []

    circ_obst_radius = []
    rect_obst_width = []
    rect_obst_height = []

    def load_obstacles(obstacle_type, num_row, num_col, obstacle_size, displacement):
        if num_row != 0:
            horisontal_space = 2 * canvas_length / (num_col)  # Mellanrum i horisentell led
        if num_col != 0:
            vertical_space = 2 * canvas_length / (num_row)  # Mellanrum i vertikalled
            start_vertical = - canvas_length + vertical_space/2  # Start i vertikalled, högst upp till vänster
        for i in range(num_row):  # För varje rad
            start_horisontal = - canvas_length + horisontal_space if displacement and i % 2 != 0 \
                else - canvas_length + horisontal_space / 2  # Förskjuts om True annars vanlig start vi horisontell led
            for j in range(
                    num_col - 1 if displacement and i % 2 != 0 else num_col):  # För varje kolonn, minska antalet om displacement
                if obstacle_type == 'circles':
                    circ_obst_coords.append([start_horisontal, start_vertical])
                    circ_obst_radius.append(obstacle_size)
                elif obstacle_type == 'rectangles':
                    rect_obst_coords.append([start_horisontal, start_vertical])
                    rect_obst_width.append(obstacle_size)
                    rect_obst_height.append(obstacle_size)
                start_horisontal += horisontal_space  # Lägg till avståndet
            start_vertical += vertical_space  # Gå till nästa rad

    load_obstacles(obst_type_main, row_main, col_main, obst_size_main, displacement_main) # Genererar hinder
    circ_obst_coords, rect_obst_coords = np.array(circ_obst_coords), np.array(rect_obst_coords)

    circ_obst_radius = np.array(circ_obst_radius)
    rect_obst_width = np.array(rect_obst_width)
    rect_obst_height = np.array(rect_obst_height)

    obst_type = ['rectangle', 'circle']  #
    rect_obst_corner_coords = []

    # Canvas grafik fisk
    fish_canvas_graphics = []
    fish_direction_arrow_graphics = []

    # Canvas grafik haj
    shark_canvas_graphics = []  # De synliga cirklarna som är hajar sparas här
    shark_direction_arrow_graphics = []

    # Canvas grafik hinder
    circ_obst_canvas_graphics = []
    rect_obst_canvas_graphics = []
    fish_eaten_this_sim = []



# Ta fram hörnen till rektangulära hinder
    def calculate_rectangle_corner_coordinates(position, base, height):
        x_c, y_c = position[0], position[1]
        b, h = float(base), float(height)

        A = [x_c - b, y_c - h]
        B = [x_c + b, y_c - h]
        C = [x_c - b, y_c + h]
        D = [x_c + b, y_c + h]

        return [A, B, C, D]


    # Ta fram hörnen till rektangulära hinder och lägg det i en lista
    for i in range(len(rect_obst_coords)):
        rect_obst_corner_coords.append(
            calculate_rectangle_corner_coordinates(rect_obst_coords[i], rect_obst_width[i], rect_obst_height[i]))
    rect_obst_corner_coords = np.array(rect_obst_corner_coords)


    # Ritar ut fiskar och dess interaktionsradie
    def draw_fishes():
        for j in range(fish_count):  # Generate animated particles in Canvas
            # Convert to canvas coordinates
            fish_canvas_graphics.append(
                canvas.create_oval(
                    (fish_coords[j, 0] - fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                    (fish_coords[j, 1] - fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                    (fish_coords[j, 0] + fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                    (fish_coords[j, 1] + fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                    outline=fish_color, fill=fish_color))
            fish_direction_arrow_graphics.append(canvas.create_line((fish_coords[j][0] + fish_graphic_radius * np.cos(
                fish_orientations[j]) + canvas_length) * res / canvas_length / 2,
                                                                    (fish_coords[j, 1] + fish_graphic_radius * np.sin(
                                                                        fish_orientations[
                                                                            j]) + canvas_length) * res / canvas_length / 2,
                                                                    (fish_coords[j, 0] + (
                                                                            fish_graphic_radius + arrow_length) * np.cos(
                                                                        fish_orientations[
                                                                            j]) + canvas_length) * res / canvas_length / 2,
                                                                    (fish_coords[j, 1] + (
                                                                            fish_graphic_radius + arrow_length) * np.sin(
                                                                        fish_orientations[
                                                                            j]) + canvas_length) * res / canvas_length / 2,
                                                                    arrow=LAST))


    def draw_shark():
        for j in range(shark_count):  # Skapar cirklar för hajar
            shark_canvas_graphics.append(
                canvas.create_oval((shark_coords[j, 0] - shark_graphic_radius + canvas_length) * res / canvas_length / 2,
                                   (shark_coords[j, 1] - shark_graphic_radius + canvas_length) * res / canvas_length / 2,
                                   (shark_coords[j, 0] + shark_graphic_radius + canvas_length) * res / canvas_length / 2,
                                   (shark_coords[j, 1] + shark_graphic_radius + canvas_length) * res / canvas_length / 2,
                                   outline=shark_color, fill=shark_color))
            shark_direction_arrow_graphics.append(canvas.create_line((shark_coords[j, 0] + shark_graphic_radius * np.cos(
                shark_orientations[j]) + canvas_length) * res / canvas_length / 2,
                                                                     (shark_coords[j, 1] + shark_graphic_radius * np.sin(
                                                                         shark_orientations[
                                                                             j]) + canvas_length) * res / canvas_length / 2,
                                                                     (shark_coords[j, 0] + (
                                                                             shark_graphic_radius + arrow_length) * np.cos(
                                                                         shark_orientations[
                                                                             j]) + canvas_length) * res / canvas_length / 2,
                                                                     (shark_coords[j, 1] + (
                                                                             shark_graphic_radius + arrow_length) * np.sin(
                                                                         shark_orientations[
                                                                             j]) + canvas_length) * res / canvas_length / 2,
                                                                     arrow=LAST))


    # Ritar ut rays och lägger dess vinkel och spetsens koordinater i en lista
    def cast_rays():
        for j in range(fish_count):
            start_angle = fish_orientations[j] - fish_half_FOV  # Startvinkel
            for ray in range(casted_rays):
                fish_rays_coords[j].append([fish_coords[j, 0] + fish_ray_radius * np.cos(start_angle),
                                            fish_coords[j, 1] + fish_ray_radius * np.sin(start_angle)])
                start_angle += fish_step_angle  # Uppdaterar vinkel för ray

        start_angle = shark_orientations - shark_half_FOV  # Startvinkel
        for ray in range(casted_rays):
            shark_rays_coords.append([shark_coords[0, 0] + shark_ray_radius * np.cos(start_angle),
                                      shark_coords[0, 1] + shark_ray_radius * np.sin(start_angle)])
            start_angle += shark_step_angle  # Uppdaterar vinkel för ray


    # Ritar cirkulära hinder
    def draw_circular_obstacles():
        for j in range(circ_obst_coords.shape[0]):
            circ_obst_canvas_graphics.append(
                canvas.create_oval((circ_obst_coords[j, 0] - circ_obst_radius[j] + canvas_length) * res / canvas_length / 2,
                                   (circ_obst_coords[j, 1] - circ_obst_radius[j] + canvas_length) * res / canvas_length / 2,
                                   (circ_obst_coords[j, 0] + circ_obst_radius[j] + canvas_length) * res / canvas_length / 2,
                                   (circ_obst_coords[j, 1] + circ_obst_radius[j] + canvas_length) * res / canvas_length / 2,
                                   outline= black, fill=black))


    # Ritar rektangulära hinder
    def draw_rectangular_obstacles():
        for j in range(rect_obst_coords.shape[0]):
            rect_obst_canvas_graphics.append(canvas.create_rectangle(
                (rect_obst_coords[j, 0] + rect_obst_width[j] + canvas_length) * res / canvas_length / 2,
                (rect_obst_coords[j, 1] + rect_obst_height[j] + canvas_length) * res / canvas_length / 2,
                (rect_obst_coords[j, 0] - rect_obst_width[j] + canvas_length) * res / canvas_length / 2,
                (rect_obst_coords[j, 1] - rect_obst_height[j] + canvas_length) * res / canvas_length / 2,
                outline='#000000', fill='#000000'))


    def is_point_inside_circle(circ_obst_coords, point, radius):
        return np.linalg.norm(circ_obst_coords - point) < radius


    # Beräknar arean av en triangel givet tre punkter
    def triangle_area(x1, y1, x2, y2, x3, y3):
        return abs((x1 * (y2 - y3) +
                    x2 * (y3 - y1) +
                    x3 * (y1 - y2)) / 2.0)


    # Kollar om en punkt är innanför eller utanför en given rektangel med fyra hörn
    def is_point_outside_rectangle(rectangle_corner_coords, point, outside):
        x = point[0]
        y = point[1]

        x1, y1 = rectangle_corner_coords[0, 0], rectangle_corner_coords[0, 1]
        x2, y2 = rectangle_corner_coords[1, 0], rectangle_corner_coords[1, 1]
        x3, y3 = rectangle_corner_coords[2, 0], rectangle_corner_coords[2, 1]
        x4, y4 = rectangle_corner_coords[3, 0], rectangle_corner_coords[3, 1]

        A = (triangle_area(x1, y1, x2, y2, x3, y3) +  # Calculate area of rektangel
             triangle_area(x1, y1, x4, y4, x3, y3))
        A1 = triangle_area(x, y, x1, y1, x2, y2)  # Calculate area of triangle PAB
        A2 = triangle_area(x, y, x2, y2, x3, y3)  # Calculate area of triangle PBC
        A3 = triangle_area(x, y, x3, y3, x4, y4)  # Calculate area of triangle PCD
        A4 = triangle_area(x, y, x1, y1, x4, y4)  # Calculate area of triangle PAD

        # Check if sum of A1, A2, A3 and A4 is same as A
        # 0.01 Felmarginal då datorn inte räknar exakt

        return A1 + A2 + A3 + A4 - 0.1 > A if outside else A1 + A2 + A3 + A4 < A


    def calculate_distance_circ_to_rect(circ_coord, circ_radius, rect_corners_coords, many_rectangles, calculate_distance):
        R = circ_radius
        Xc, Yc = circ_coord[0], circ_coord[1]  # Fiskens koordinater
        if many_rectangles:  # Om flera rektanglar
            X1, Y1 = rect_corners_coords[:, 0, 0], rect_corners_coords[:, 0, 1]  # Ena hörnet
            X2, Y2 = rect_corners_coords[:, 3, 0], rect_corners_coords[:, 3, 1]  # Andra hörnet
        else:
            X1, Y1 = rect_corners_coords[0, 0], rect_corners_coords[0, 1]  # Ena hörnet
            X2, Y2 = rect_corners_coords[3, 0], rect_corners_coords[3, 1]  # Andra hörnet

        NearestX = np.maximum(X1, np.minimum(Xc, X2))  # Tar fram de närmsta punkten
        NearestY = np.maximum(Y1, np.minimum(Yc, Y2))

        Dx = Xc - NearestX  # Avståndet från närmsta punkten på rektangeln till cirkelns centrum
        Dy = Yc - NearestY
        # Avstånd eller boolean om cirkeln är innanför rektangeln
        return np.absolute(np.sqrt(Dx ** 2 + Dy ** 2) - R) if calculate_distance else [Dx * Dx + Dy * Dy <= R * R, Dx, Dy]


    def detect_obst_in_radius(agent_coord, radius):
        obst_type_in_radius = [[], []]  # Lista med index för de hinder som detekterats innanför interaktionsradien
        detected = []
        if np.size(rect_obst_corner_coords) != 0:
            rect_obst_in_radius = calculate_distance_circ_to_rect(agent_coord, radius, rect_obst_corner_coords, True, False)[0]
            if True in rect_obst_in_radius:
                obst_type_in_radius[0].extend([index for index, element in enumerate(rect_obst_in_radius) if element])
                detected.append(True)
        if np.size(circ_obst_coords) != 0:
            circ_obst_in_radius = calculate_distance(circ_obst_coords, agent_coord) - circ_obst_radius < radius
            if True in circ_obst_in_radius:
                obst_type_in_radius[1].extend([index for index, element in enumerate(circ_obst_in_radius) if element])
                detected.append(True)
        return [True in detected, obst_type_in_radius]


    def detect_closest_obst(ray_coords, agent_coord, obst_type_in_radius, agent_graphic_radius):
        n_rays = len(ray_coords)
        obst_type_detect = [[], []]  # Lista med vilken typ av hinder den känner av Korta listan
        obst_detect = [[], []]  # Lista med vilken typ plus alla ray Långa listan
        closest_obst_all = [[], []]  # Lista med (ett enda) hinder index, strålavstånd och boolean
        list_obst_index_raydist_raybool = [[], []]  # Lista med (flera) hinder index, avstånd och boolean
        common_ray_boolean = [False for i in range(n_rays)]  # Boolean relativ partikeln
        for type in range(len(obst_type)):
            for k in range(len(obst_type_in_radius[type])):
                obst_index = obst_type_in_radius[type][k]  # Hinder index
                if obst_type[type] == 'rectangle':
                    booleans = [is_point_outside_rectangle(rect_obst_corner_coords[obst_index], ray_coords[i], False)
                                for i in range(n_rays)]
                    detect = True in booleans
                    if detect:
                        obst_detect[type].append(detect)
                        closest_dist = calculate_distance_circ_to_rect(agent_coord, agent_graphic_radius,
                                                                       rect_obst_corner_coords[obst_index], False, True) if \
                            booleans[2] or booleans[3] else np.inf  # Kollar på de mittersta strålarna
                        list_obst_index_raydist_raybool[type].append(
                            [obst_index, closest_dist, booleans])  # Tar fram närmast avstånden om rays är träffad
                        a, b = common_ray_boolean, booleans  # Merga ihop boolean till boolean
                        common_ray_boolean = [a or b for a, b in zip(a, b)]
                elif obst_type[type] == 'circle':
                    booleans = [
                        is_point_inside_circle(circ_obst_coords[obst_index], ray_coords[i], circ_obst_radius[obst_index])
                        for
                        i in range(n_rays)]
                    detect = True in booleans
                    if detect:
                        obst_detect[type].append(detect)
                        closest_dist = np.absolute(np.linalg.norm(np.array(circ_obst_coords[obst_index]) -
                                                                  agent_coord) - agent_graphic_radius - circ_obst_radius[
                                                       obst_index]) if booleans[2] or booleans[3] else np.inf
                        list_obst_index_raydist_raybool[type].append([obst_index, closest_dist, booleans])
                        a, b = common_ray_boolean, booleans  # Merga ihop boolean till boolean
                        common_ray_boolean = [a or b for a, b in zip(a, b)]
            obst_type_detect[type] = True in obst_detect[type]
            closest_obst_all[type] = min(list_obst_index_raydist_raybool[type], key=lambda x: x[1]) if obst_type_detect[
                type] else [-1, np.inf, False]

        min_dist_type = min(closest_obst_all, key=lambda x: x[1])  # Ger den array med minsta avståndet i [[],[],[]]
        min_dist = min_dist_type[1]
        # ray_boolean = min_dist_type[2] if all(
        #   common_ray_boolean) else common_ray_boolean  # Om alla är träffade, tar den närmsta
        ray_boolean = common_ray_boolean
        result = [min_dist, ray_boolean]
        return result


    def avoid_obstacle(closest_obst_distance, ray_boolean):
        if not ray_boolean[int(len(ray_boolean) / 2 - 1)] and not ray_boolean[int(len(ray_boolean) / 2)]:
            sign = 0
        else:
            if all(ray_boolean):
                return np.pi / 2
            else:
                i = 1
                first_free_index = int(len(ray_boolean) / 2) - 1
                while ray_boolean[first_free_index]:
                    first_free_index += i * (-1) ** (i - 1)
                    i += 1
                sign = -1 if (first_free_index <= 2) else 1

        angle_weight = np.pi / 4 / (closest_obst_distance + 1) * sign  # August får ändra
        if np.abs(angle_weight) > np.pi / 2:
            angle_weight = np.pi / 4 * sign

        return angle_weight


    def calculate_distance(coords, coord):  # Räknar ut avstånd mellan punkterna coords och punkten coord
        return np.minimum(
            np.sqrt(((coords[:, 0]) % (2 * canvas_length) - (coord[0]) % (2 * canvas_length)) ** 2 + (
                    (coords[:, 1]) % (2 * canvas_length) - (coord[1]) % (2 * canvas_length)) ** 2),
            np.sqrt((coords[:, 0] - coord[0]) ** 2 + (coords[:, 1] - coord[1]) ** 2))


    def update_position(coords, speed, orientations):  # Uppdaterar en partikels position
        coords[:, 0] = (coords[:, 0] + speed * np.cos(orientations) * time_step + canvas_length) % (
                2 * canvas_length) - canvas_length
        coords[:, 1] = (coords[:, 1] + speed * np.sin(orientations) * time_step + canvas_length) % (
                2 * canvas_length) - canvas_length
        return coords


    def get_direction(coord1, coord2):  # Ger riktningen från coord1 till coord2 i radianer
        if np.sqrt((coord2[0] - coord1[0]) ** 2 + (coord2[1] - coord1[1]) ** 2) < np.sqrt(
                ((coord2[0]) % (2 * canvas_length) - (coord1[0]) % (2 * canvas_length)) ** 2 + (
                        (coord2[1]) % (2 * canvas_length) - (coord1[1]) % (2 * canvas_length)) ** 2):
            return np.arctan2(coord2[1] - coord1[1], coord2[0] - coord1[0])
        else:
            return np.arctan2((coord2[1]) % (2 * canvas_length) - (coord1[1]) % (2 * canvas_length),
                              (coord2[0]) % (2 * canvas_length) - (coord1[0]) % (2 * canvas_length))

    def murder_fish_coords(dead_fish_index):  # Tar bort fisk som blivit uppäten
        new_fish_coords = np.delete(fish_coords, dead_fish_index, 0)
        return new_fish_coords


    def murder_fish_orientations(dead_fish_index):
        new_fish_orientations = np.delete(fish_orientations, dead_fish_index)
        return new_fish_orientations

    # Kallar på de grafiska funktionerna

    if visuals_on:
        draw_fishes()
        draw_shark()
        draw_circular_obstacles()
        draw_rectangular_obstacles()
        fish_count_canvas_text = canvas.create_text(100, 20,
                                                    text=len(fish_coords))
    cast_rays()

    for t in range(simulation_iterations):
        fish_coords = update_position(fish_coords, fish_speed, fish_orientations)  # Uppdatera fiskposition
        shark_coords = update_position(shark_coords, shark_speed, shark_orientations)  # Uppdatera hajposition
        shark_fish_distances = calculate_distance(fish_coords, shark_coords[0])  # Räknar ut det kortaste avståndet mellan haj och varje fisk
        closest_fish = np.argmin(shark_fish_distances)  # Index av fisk närmst haj

        # Haj loop
        for j in range(shark_count):
            # Updating animation coordinates haj
            if visuals_on:
                canvas.coords(shark_canvas_graphics[j],
                              (shark_coords[j, 0] - shark_graphic_radius + canvas_length) * res / canvas_length / 2,
                              (shark_coords[j, 1] - shark_graphic_radius + canvas_length) * res / canvas_length / 2,
                              (shark_coords[j, 0] + shark_graphic_radius + canvas_length) * res / canvas_length / 2,
                              (shark_coords[j, 1] + shark_graphic_radius + canvas_length) * res / canvas_length / 2, )

                canvas.coords(shark_direction_arrow_graphics[j],
                              (shark_coords[j, 0] + shark_graphic_radius * np.cos(
                                  shark_orientations[j]) + canvas_length) * res / canvas_length / 2,
                              (shark_coords[j, 1] + shark_graphic_radius * np.sin(
                                  shark_orientations[j]) + canvas_length) * res / canvas_length / 2,
                              (shark_coords[j, 0] + (shark_graphic_radius + arrow_length) * np.cos(
                                  shark_orientations[j]) + canvas_length) * res / canvas_length / 2,
                              (shark_coords[j, 1] + (shark_graphic_radius + arrow_length) * np.sin(
                                  shark_orientations[j]) + canvas_length) * res / canvas_length / 2)
            start_angle = shark_orientations[j] - shark_half_FOV  # Startvinkel
            for ray in range(casted_rays):
                shark_rays_coords[ray] = [shark_coords[j, 0] + shark_ray_radius * np.cos(start_angle),
                                          shark_coords[j, 1] + shark_ray_radius * np.sin(start_angle)]
                start_angle += shark_step_angle  # Uppdaterar vinkel för ray

            # Obstacle avoidance shark
            shark_avoid_angle = 0
            detect_obst_in_radius_info = detect_obst_in_radius(shark_coords[j],
                                                               shark_ray_radius)  # [[[index vägg],[index rekt],[index circ]]]
            obst_type_in_radius = detect_obst_in_radius_info[1]  # De hindren i interaktionsradien
            if detect_obst_in_radius_info[0]:
                detect_info = detect_closest_obst(shark_rays_coords, shark_coords[j],
                                                  obst_type_in_radius,
                                                  shark_graphic_radius)  # [[hindertyp],[hinderindex],[ray_boolean]]
                closest_obst_distance, ray_boolean = detect_info[0], detect_info[1]  # Tilldelar namn
                shark_avoid_angle = avoid_obstacle(closest_obst_distance,
                                                   ray_boolean) if closest_obst_distance != np.inf else 0
                # print(np.rad2deg(shark_avoid_angle))

            rect_obst_detect_ind = detect_obst_in_radius_info[1][0]  # De hindren som detekterats
            circ_obst_detect_ind = detect_obst_in_radius_info[1][1]

            # Overlapp circular obstacles shark
            if np.size(circ_obst_coords) != 0:
                circ_obst_distances = calculate_distance(circ_obst_coords[circ_obst_detect_ind], shark_coords[j])
                angle = np.arctan2(circ_obst_coords[circ_obst_detect_ind, 1] - shark_coords[j, 1],
                                   circ_obst_coords[circ_obst_detect_ind, 0] - shark_coords[
                                       j, 0])  # Directions of others array from the particle
                overlap = circ_obst_distances < (shark_graphic_radius + circ_obst_radius[circ_obst_detect_ind])  # Applying

                for ind in np.where(overlap)[0]:
                    shark_coords[j, 0] = shark_coords[j, 0] + (circ_obst_distances[ind] - (
                            shark_graphic_radius + circ_obst_radius[circ_obst_detect_ind[ind]])) * np.cos(angle[ind])
                    shark_coords[j, 1] = shark_coords[j, 1] + (circ_obst_distances[ind] - (
                            shark_graphic_radius + circ_obst_radius[circ_obst_detect_ind[ind]])) * np.sin(angle[ind])

            # Overlap rectangular obstacles shark
            if np.size(rect_obst_coords) != 0:
                rect_obst_overlap_info = calculate_distance_circ_to_rect(shark_coords[j], shark_graphic_radius,
                                                                         rect_obst_corner_coords[rect_obst_detect_ind], True,
                                                                         False)
                shark_inside_rect_obst = rect_obst_overlap_info[0]
                Dx, Dy = rect_obst_overlap_info[1], rect_obst_overlap_info[
                    2]  # Avstånd mellan närmaste punkt på rektangen till cirklens centrum
                for ind in np.where(shark_inside_rect_obst)[0]:
                    angle = np.arctan2(Dy[ind], Dx[ind])  # Directions of others array from the particle
                    normal_distance = np.sqrt(Dx[ind] ** 2 + Dy[ind] ** 2) - shark_graphic_radius
                    shark_coords[j, 0] = shark_coords[j, 0] + np.absolute(normal_distance) * np.cos(angle)
                    shark_coords[j, 1] = shark_coords[j, 1] + np.absolute(normal_distance) * np.sin(angle)

        # Fisk loop
        for j in range(len(fish_coords)):
            if visuals_on:
                # Updating animation coordinates fisk
                canvas.coords(fish_canvas_graphics[j],
                              (fish_coords[j, 0] - fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                              (fish_coords[j, 1] - fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                              (fish_coords[j, 0] + fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                              (fish_coords[j, 1] + fish_graphic_radius + canvas_length) * res / canvas_length / 2)
                canvas.coords(fish_direction_arrow_graphics[j],
                              (fish_coords[j, 0] + fish_graphic_radius * np.cos(
                                  fish_orientations[j]) + canvas_length) * res / canvas_length / 2,
                              (fish_coords[j, 1] + fish_graphic_radius * np.sin(
                                  fish_orientations[j]) + canvas_length) * res / canvas_length / 2,
                              (fish_coords[j, 0] + (fish_graphic_radius + arrow_length) * np.cos(
                                  fish_orientations[j]) + canvas_length) * res / canvas_length / 2,
                              (fish_coords[j, 1] + (fish_graphic_radius + arrow_length) * np.sin(
                                  fish_orientations[j]) + canvas_length) * res / canvas_length / 2)

            # Rays casting fish
            start_angle = fish_orientations[j] - fish_half_FOV  # Startvinkel
            for ray in range(casted_rays):
                fish_rays_coords[j][ray] = [fish_coords[j, 0] + fish_ray_radius * np.cos(start_angle),
                                            fish_coords[j, 1] + fish_ray_radius * np.sin(start_angle)]
                start_angle += fish_step_angle  # Uppdaterar vinkel för ray

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

            # Obstacle avoidance fisk
            fish_avoid_angle = 0
            weight_boolean_avoid = False
            weight_boolean_vicsek = True

            detect_obst_in_radius_info = detect_obst_in_radius(fish_coords[j],
                                                               fish_ray_radius)  # [[[index vägg],[index rekt],[index circ]]]
            obst_type_in_radius = detect_obst_in_radius_info[1]  # De hindren i interaktionsradien
            if detect_obst_in_radius_info[0]:
                detect_info = detect_closest_obst(fish_rays_coords[j], fish_coords[j],
                                                  obst_type_in_radius,
                                                  fish_graphic_radius)  # [[hindertyp],[hinderindex],[ray_boolean]]
                closest_obst_distance, ray_boolean = detect_info[0], detect_info[1]  # Tilldelar namn
                fish_avoid_angle = avoid_obstacle(closest_obst_distance,
                                                  ray_boolean) if closest_obst_distance != np.inf else 0
                if np.abs(fish_avoid_angle) > 0:  # Prioriterar avoidance
                    weight_boolean_vicsek = False  # Stänger av vicsek
                    weight_boolean_avoid = True

                rect_obst_detect_ind = detect_obst_in_radius_info[1][0]  # De hindren som detekterats
                circ_obst_detect_ind = detect_obst_in_radius_info[1][1]

                # Overlapp circular obstacles
                if np.size(circ_obst_coords) != 0:
                    circ_obst_distances = calculate_distance(circ_obst_coords[circ_obst_detect_ind], fish_coords[j])
                    angle = np.arctan2(circ_obst_coords[circ_obst_detect_ind, 1] - fish_coords[j, 1],
                                       circ_obst_coords[circ_obst_detect_ind, 0] - fish_coords[
                                           j, 0])  # Directions of others array from the particle
                    overlap = circ_obst_distances < (fish_graphic_radius + circ_obst_radius[circ_obst_detect_ind])  # Applying

                    for ind in np.where(overlap)[0]:
                        fish_coords[j, 0] = fish_coords[j, 0] + (circ_obst_distances[ind] - (
                                fish_graphic_radius + circ_obst_radius[circ_obst_detect_ind[ind]])) * np.cos(angle[ind])
                        fish_coords[j, 1] = fish_coords[j, 1] + (circ_obst_distances[ind] - (
                                fish_graphic_radius + circ_obst_radius[circ_obst_detect_ind[ind]])) * np.sin(angle[ind])

                # Overlap rectangular obstacles
                if np.size(rect_obst_coords) != 0:
                    rect_obst_overlap_info = calculate_distance_circ_to_rect(fish_coords[j], fish_graphic_radius,
                                                                             rect_obst_corner_coords[rect_obst_detect_ind],
                                                                             True, False)
                    fish_inside_rect_obst = rect_obst_overlap_info[0]
                    Dx, Dy = rect_obst_overlap_info[1], rect_obst_overlap_info[
                        2]  # Avstånd mellan närmaste punkt på rektangen till cirklens centrum
                    for ind in np.where(fish_inside_rect_obst)[0]:
                        angle = np.arctan2(Dy[ind], Dx[ind])  # Directions of others array from the particle
                        normal_distance = np.sqrt(Dx[ind] ** 2 + Dy[ind] ** 2) - fish_graphic_radius
                        fish_coords[j, 0] = fish_coords[j, 0] + np.absolute(normal_distance) * np.cos(angle)
                        fish_coords[j, 1] = fish_coords[j, 1] + np.absolute(normal_distance) * np.sin(angle)
            if visuals_on:
                if j == closest_fish:
                    canvas.itemconfig(fish_canvas_graphics[j], fill=closest_fish_color)  # Byt färg på fisk närmst haj
                else:
                    canvas.itemconfig(fish_canvas_graphics[j], fill=fish_color)
            inter_fish_distances = calculate_distance(fish_coords, fish_coords[
                j])  # Räknar ut avstånd mellan fisk j och alla andra fiskar
            fish_in_interaction_radius = inter_fish_distances < fish_interaction_radius  # Vilka fiskar är inom en fisks interraktionsradie

            if shark_fish_distances[j] < fish_interaction_radius:  # Om hajen är nära fisken, undvik hajen
                fish_orientations[j] = get_direction(shark_coords[0], fish_coords[j]) + fish_avoid_angle + fish_noise * rng.uniform(-1 / 2,1 / 2)
            else:
                fish_orientations[j] = weight_boolean_vicsek * np.angle(
                    np.sum(np.exp(fish_orientations[fish_in_interaction_radius] * 1j))) + fish_noise * rng.uniform(-1 / 2, 1 / 2) + \
                                       weight_boolean_avoid * (fish_orientations[j] + fish_avoid_angle)

        #Haj undvik hinder, annars jaga fisk
        shark_orientations[0] = shark_orientations[0] + shark_avoid_angle + fish_noise * rng.uniform(-1 / 2, 1 / 2) if np.absolute(shark_avoid_angle) > 0 \
           else get_direction(shark_coords[0], fish_coords[closest_fish]) + fish_noise * rng.uniform(-1 / 2, 1 / 2)
        #shark_orientations[0] = get_direction(shark_coords[0], fish_coords[closest_fish])

        # Kollar om närmaste fisk är inom murder radien
        if len(fish_coords) > 4:  # <- den if-satsen är för att stoppa crash vid få fiskar
            if calculate_distance(shark_coords, fish_coords[closest_fish])[0] < murder_radius:
                last_index = len(fish_coords) - 1  # Sista index som kommer försvinna efter den mördade fisken tas bort
                if visuals_on:
                    canvas.delete(fish_canvas_graphics[last_index])  # Tar bort fisken
                    canvas.delete(fish_direction_arrow_graphics[last_index])  # Tar bort fiskens pil
                fish_coords = murder_fish_coords(closest_fish)  # Tar bort index i koordinaterna
                fish_orientations = murder_fish_orientations(closest_fish)  # Tar bort index i orientations
                fish_eaten_count += 1  # Lägg till en äten fiskx
        else:
            break
        fish_eaten_this_sim.append(fish_eaten_count)
        if visuals_on:
            canvas.itemconfig(fish_count_canvas_text, text='Antal Fiskar: {:.3f}'.format(len(fish_coords)))
            tk.title('Iteration =' + str(t))
            tk.update()  # Update animation frame
    np.save('fish_eaten_this_sim.npy', fish_eaten_this_sim)

    if visuals_on:
        Tk.destroy(tk) # Destroy window
    return fish_eaten_count

main('circles', 8,8,8, True, seed=1)
