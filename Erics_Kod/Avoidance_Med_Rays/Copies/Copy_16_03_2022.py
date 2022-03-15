import cdist as cdist
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk
from scipy.spatial.distance import *
import time
from itertools import chain

res = 700  # Resolution of the animation
tk = Tk()
tk.geometry(str(int(res * 1.1)) + 'x' + str(int(res * 1.3)))  # Set height x width window
tk.configure(background='white')

canvas = Canvas(tk, bd=2)  # Generate animation window
tk.attributes('-topmost', 0)
canvas.place(x=res / 20, y=res / 20, height=res, width=res)  # Place canvas with origin in x och y
ccolor = ['#17888E', '#C1D02B', '#9E00C9', '#D80000', '#E87B00', '#9F68D3', '#4B934F']

# Parameters of the fishes
fish_interaction_radius = 10  # Interaction radius
fish_graphic_radius = 2  # Radius of agent
fish_noise = 0.1  # Diffusional noise constant
fish_arrow_length = fish_graphic_radius
fish_ray_radius = fish_interaction_radius

# Raycasting
step_angle = 2 * np.arctan(fish_graphic_radius / fish_interaction_radius)
casted_rays = 6
FOV_angle = step_angle * (casted_rays - 1)  # Field of view angle
half_FOV = FOV_angle / 2

# Parameters for shark

# Mutual parameters
fish_shark_graphic_radius = 2
fish_shark_ray_radius = 10

# Physical parameters of the system
simulation_iterations = 100000  # Simulation time
time_step = 0.03  # Time step
canvas_length = 100  # Size of box
fish_count = 100 # Number of particles
fish_speed = 20
visual_debug = False

if not visual_debug:
    x = np.array(np.random.rand(fish_count) * 2 * canvas_length - canvas_length)
    y = np.array(np.random.rand(fish_count) * 2 * canvas_length - canvas_length)
    fish_coords = np.column_stack((x, y))
    fish_orientations = np.random.rand(fish_count) * 2 * np.pi  # orientations
else:
    x = np.array(np.ones(fish_count) * -50)
    y = np.array(np.ones(fish_count) * 50)
    fish_coords = np.column_stack((x, y))
    fish_orientations = -np.ones(fish_count) * np.pi/4  # orientations

circ_obst_coords = []
rect_obst_coords = []

circ_obst_radius = []
rect_obst_width = []
rect_obst_height = []

with open('../Environments/Environment4', 'r') as filestream:
    next(filestream)  # Skip first row
    for line in filestream:  # Read every row
        if line != "\n":
            currentline = line.split(',')
            if ('None' not in currentline[:3]):
                circ_obst_coords.append([float(currentline[0]), float(currentline[1])])
                circ_obst_radius.append(float(currentline[2]))
            if ('None' not in currentline[3:]):
                rect_obst_coords.append([float(currentline[3]), float(currentline[4])])
                rect_obst_width.append(float(currentline[5]))
                rect_obst_height.append(float(currentline[6]))
    circ_obst_coords, rect_obst_coords = np.array(circ_obst_coords), np.array(rect_obst_coords)

circ_obst_radius = np.array(circ_obst_radius)
rect_obst_width = np.array(rect_obst_width)
rect_obst_height = np.array(rect_obst_height)


obst_type = ['rect', 'circ'] #
obst_coords = [rect_obst_coords, circ_obst_coords] #
rect_obst_corner_coords = []

# Diverse tomma listor, mest grafik
circ_and_rect_obst_coords = np.concatenate((circ_obst_coords, rect_obst_coords))
fish_canvas_graphics = []
fish_interaction_radius_canvas_graphics = []
fish_ray_radius_canvas_graphics = []
fish_canvas_rays_graphics = [[] for i in range(fish_count)]
rays_coords = [[] for i in range(fish_count)]
rays_angle_relative_velocity = [[] for i in range(fish_count)]
circ_obst_canvas_graphics = []
rect_obst_canvas_graphics = []
fish_direction_arrow_graphics = []

# Ta fram hörnen till rektangulära hinder
def calculate_rectangle_corner_coordinates(position, base, height):
    x_c, y_c = position[0], position[1]
    b,h = float(base), float(height)

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

# Funktion för att fisken startposition inte hamnar i ett hinder
def generate_fish_not_inside_obstacle_coordinates():
    for j in range(fish_count):
        while (True):  # Check if the obstacle is within agent
            distance_to_obstacles = []
            for k in range(circ_and_rect_obst_coords.shape[0]):
                distance_to_obstacles.append(np.linalg.norm(fish_coords[j] - circ_and_rect_obst_coords[k]))
            if (np.min(distance_to_obstacles) < fish_interaction_radius):
                fish_coords[j, 0] = np.random.rand() * 2 * canvas_length - canvas_length
                fish_coords[j, 1] = np.random.rand() * 2 * canvas_length - canvas_length
            else:
                break


# Ritar ut fiskar och dess interaktionsradie
def draw_fishes():
    for j in range(fish_count):  # Generate animated particles in Canvas
        # Convert to canvas coordinates
        fish_canvas_graphics.append(
            canvas.create_oval((fish_coords[j, 0] - fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                               (fish_coords[j, 1] - fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                               (fish_coords[j, 0] + fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                               (fish_coords[j, 1] + fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                               outline=ccolor[0], fill=ccolor[0]))
        fish_direction_arrow_graphics.append(canvas.create_line((fish_coords[j][0] + fish_graphic_radius * np.cos(
            fish_orientations[j]) + canvas_length) * res / canvas_length / 2,
                                                                (fish_coords[j, 1] + fish_graphic_radius * np.sin(fish_orientations[j]) + canvas_length) * res / canvas_length / 2,
                                                                (fish_coords[j, 0] + (fish_graphic_radius + fish_arrow_length) * np.cos(fish_orientations[j]) + canvas_length) * res / canvas_length / 2,
                                                                (fish_coords[j, 1] + (fish_graphic_radius + fish_arrow_length) * np.sin(fish_orientations[j]) + canvas_length) * res / canvas_length / 2, arrow=LAST))
        if visual_debug:
            fish_interaction_radius_canvas_graphics.append(
                canvas.create_oval((fish_coords[j, 0] - fish_interaction_radius + canvas_length) * res / canvas_length / 2,
                                   (fish_coords[j, 1] - fish_interaction_radius + canvas_length) * res / canvas_length / 2,
                                   (fish_coords[j, 0] + fish_interaction_radius + canvas_length) * res / canvas_length / 2,
                                   (fish_coords[j, 1] + fish_interaction_radius + canvas_length) * res / canvas_length / 2,
                                   outline=ccolor[2], width=1))
            fish_ray_radius_canvas_graphics.append(
                canvas.create_oval((fish_coords[j, 0] - fish_ray_radius + canvas_length) * res / canvas_length / 2,
                                   (fish_coords[j, 1] - fish_ray_radius + canvas_length) * res / canvas_length / 2,
                                   (fish_coords[j, 0] + fish_ray_radius + canvas_length) * res / canvas_length / 2,
                                   (fish_coords[j, 1] + fish_ray_radius + canvas_length) * res / canvas_length / 2,
                                   outline=ccolor[2], width=1))

# Ritar ut rays och lägger dess vinkel och spetsens koordinater i en lista
def cast_rays():
    for j in range(fish_count):
        start_angle = fish_orientations[j] - half_FOV  # Startvinkel
        for ray in range(casted_rays):
            if visual_debug:
                fish_canvas_rays_graphics[j].append(
                    canvas.create_line((fish_coords[j, 0] + canvas_length) * res / canvas_length / 2,
                                       (fish_coords[j, 1] + canvas_length) * res / canvas_length / 2,
                                       (fish_coords[j, 0] + fish_ray_radius * np.cos(
                                           start_angle) + canvas_length) * res / canvas_length / 2,
                                       (fish_coords[j, 1] + fish_ray_radius * np.sin(
                                           start_angle) + canvas_length) * res / canvas_length / 2, fill=ccolor[3]))
            rays_coords[j].append([fish_coords[j, 0] + fish_ray_radius * np.cos(start_angle),
                                   fish_coords[j, 1] + fish_ray_radius * np.sin(start_angle)])
            rays_angle_relative_velocity[j].append(start_angle)
            start_angle += step_angle  # Uppdaterar vinkel för ray


# Ritar cirkulära hinder
def draw_circular_obstacles():
    for j in range(circ_obst_coords.shape[0]):
        circ_obst_canvas_graphics.append(
            canvas.create_oval((circ_obst_coords[j, 0] - circ_obst_radius[j] + canvas_length) * res / canvas_length / 2,
                               (circ_obst_coords[j, 1] - circ_obst_radius[j] + canvas_length) * res / canvas_length / 2,
                               (circ_obst_coords[j, 0] + circ_obst_radius[j] + canvas_length) * res / canvas_length / 2,
                               (circ_obst_coords[j, 1] + circ_obst_radius[j] + canvas_length) * res / canvas_length / 2,
                               outline=ccolor[5], fill=ccolor[3]))


# Ritar rektangulära hinder
def draw_rectangular_obstacles():
    for j in range(rect_obst_coords.shape[0]):
        rect_obst_canvas_graphics.append(canvas.create_rectangle(
            (rect_obst_coords[j, 0] + rect_obst_width[j] + canvas_length) * res / canvas_length / 2,
            (rect_obst_coords[j, 1] + rect_obst_height[j] + canvas_length) * res / canvas_length / 2,
            (rect_obst_coords[j, 0] - rect_obst_width[j] + canvas_length) * res / canvas_length / 2,
            (rect_obst_coords[j, 1] - rect_obst_height[j] + canvas_length) * res / canvas_length / 2,
            outline=ccolor[5], fill=ccolor[4]))


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
    if many_rectangles:
        X1, Y1 = rect_corners_coords[:,0,0], rect_corners_coords[:,0,1] # Ena hörnet
        X2, Y2 = rect_corners_coords[:,3,0],rect_corners_coords[:,3,1] # Andra hörnet
    else:
        X1, Y1 = rect_corners_coords[0,0], rect_corners_coords[0,1] # Ena hörnet
        X2, Y2 = rect_corners_coords[3,0],rect_corners_coords[3,1] # Andra hörnet

    NearestX = np.maximum(X1, np.minimum(Xc, X2)) # Tar fram de närmsta punkten
    NearestY = np.maximum(Y1, np.minimum(Yc, Y2))

    Dx = NearestX - Xc # Avståndet från närmsta punkten på rektangeln till cirkelns centrum
    Dy = NearestY - Yc
    # Avstånd eller boolean om cirkeln är innanför rektangeln
    return np.absolute(np.sqrt(Dx ** 2 + Dy ** 2)-R) if calculate_distance else Dx * Dx + Dy * Dy <= R *R


def detect_obst_in_radius(agent_coord):
    obst_type_in_radius = [[], []]  # Lista med index för de hinder som detekterats innanför interaktionsradien
    detected = []
    rect_obst_in_radius = calculate_distance_circ_to_rect(agent_coord, fish_shark_ray_radius, rect_obst_corner_coords, True, False)
    circ_obst_in_radius = calculate_distance(circ_obst_coords, agent_coord) - circ_obst_radius < fish_shark_ray_radius
    if True in rect_obst_in_radius:
        obst_type_in_radius[0].extend([index for index, element in enumerate(rect_obst_in_radius) if element])
        detected.append(True)
    elif True in circ_obst_in_radius:
        obst_type_in_radius[1].extend([index for index, element in enumerate(circ_obst_in_radius) if element])
        detected.append(True)

    return [True in detected, obst_type_in_radius]


def detect_closest_obst(ray_coords, agent_coord, obst_type_in_radius):
    n_rays = len(ray_coords)
    obst_type_detect = [[], []]  # Lista med vilken typ av hinder den känner av Korta listan
    obst_detect = [[], []]  # Lista med vilken typ plus alla ray Långa listan
    closest_obst_all = [[], []]  # Lista med (ett enda) hinder index, strålavstånd och boolean
    list_obst_index_raydist_raybool = [[], []]  # Lista med (flera) hinder index, avstånd och boolean
    common_ray_boolean = [False for i in range(n_rays)]  # Boolean relativ partikeln
    for type in range(len(obst_type)):
        for k in range(len(obst_type_in_radius[type])):
            obst_index = obst_type_in_radius[type][k]  # Hinder index
            if obst_type[type] == 'rect':
                booleans = [is_point_outside_rectangle(rect_obst_corner_coords[obst_index], ray_coords[i], False)
                            for i in range(n_rays)]
                detect = True in booleans
                if detect:
                    obst_detect[type].append(detect)
                    closest_dist = calculate_distance_circ_to_rect(agent_coord, fish_shark_graphic_radius,
                                                                   rect_obst_corner_coords[obst_index], False, True) if booleans[2] or booleans[3] else np.inf # Kollar på de mittersta strålarna
                    list_obst_index_raydist_raybool[type].append([obst_index, closest_dist, booleans])  # Tar fram närmast avstånden om rays är träffad
                    a, b = common_ray_boolean, booleans  # Merga ihop boolean till boolean
                    common_ray_boolean = [a or b for a, b in zip(a, b)]
            elif obst_type[type] == 'circ':
                booleans = [is_point_inside_circle(circ_obst_coords[obst_index], ray_coords[i], circ_obst_radius[obst_index]) for
                            i in range(n_rays)]
                detect = True in booleans
                if detect:
                    obst_detect[type].append(detect)
                    closest_dist = np.absolute(np.linalg.norm(np.array(circ_obst_coords[obst_index]) -
                                                              agent_coord) - fish_graphic_radius - circ_obst_radius[obst_index]) if booleans[2] or booleans[3] else np.inf
                    list_obst_index_raydist_raybool[type].append([obst_index, closest_dist, booleans])
                    a, b = common_ray_boolean, booleans  # Merga ihop boolean till boolean
                    common_ray_boolean = [a or b for a, b in zip(a, b)]
        obst_type_detect[type] = True in obst_detect[type]
        closest_obst_all[type] = min(list_obst_index_raydist_raybool[type], key=lambda x: x[1]) if obst_type_detect[type] else [-1, np.inf, False]

    min_dist_type = min(closest_obst_all, key=lambda x: x[1])  # Ger den array med minsta avståndet i [[],[],[]]
    min_dist = min_dist_type[1]
    #ray_boolean = min_dist_type[2] if all(
    #   common_ray_boolean) else common_ray_boolean  # Om alla är träffade, tar den närmsta
    ray_boolean = common_ray_boolean
    result = [min_dist, ray_boolean]
    # if True in obst_type_detect:
    #     time.sleep(0.2)
    #     print(result)
    return result


def avoid_obstacle(closest_obst_distance, ray_boolean):
    if not ray_boolean[int(len(ray_boolean) / 2 - 1)] and not ray_boolean[int(len(ray_boolean) / 2)]:
        sign = 0
    else:
        if all(ray_boolean):
            return np.pi/2
        else:
            i = 1
            first_free_index = int(len(ray_boolean) / 2) - 1
            while ray_boolean[first_free_index]:
                first_free_index += i * (-1) ** (i - 1)
                i += 1
            sign = -1 if (first_free_index <= 2) else 1

    angle_weight = np.pi / 4 / (closest_obst_distance +1)* sign # August får ändra
    if np.abs(angle_weight) > np.pi/2:
        angle_weight = np.pi/4*sign

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


# Kallar på de grafiska funktionerna
cast_rays()
generate_fish_not_inside_obstacle_coordinates()
draw_fishes()
draw_circular_obstacles()
draw_rectangular_obstacles()

for t in range(simulation_iterations):
    fish_coords = update_position(fish_coords, fish_speed, fish_orientations)  # Uppdatera fiskposition
    for j in range(len(fish_coords)):  # Updating animation coordinates fisk
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
                      (fish_coords[j, 0] + (fish_graphic_radius + fish_arrow_length) * np.cos(
                          fish_orientations[j]) + canvas_length) * res / canvas_length / 2,
                      (fish_coords[j, 1] + (fish_graphic_radius + fish_arrow_length) * np.sin(
                          fish_orientations[j]) + canvas_length) * res / canvas_length / 2)
        if visual_debug:
            canvas.coords(fish_interaction_radius_canvas_graphics[j],
                          (fish_coords[j, 0] - fish_interaction_radius + canvas_length) * res / canvas_length / 2,
                          (fish_coords[j, 1] - fish_interaction_radius + canvas_length) * res / canvas_length / 2,
                          (fish_coords[j, 0] + fish_interaction_radius + canvas_length) * res / canvas_length / 2,
                          (fish_coords[j, 1] + fish_interaction_radius + canvas_length) * res / canvas_length / 2)
            canvas.coords(fish_ray_radius_canvas_graphics[j],
                          (fish_coords[j, 0] - fish_ray_radius + canvas_length) * res / canvas_length / 2,
                          (fish_coords[j, 1] - fish_ray_radius + canvas_length) * res / canvas_length / 2,
                          (fish_coords[j, 0] + fish_ray_radius + canvas_length) * res / canvas_length / 2,
                          (fish_coords[j, 1] + fish_ray_radius + canvas_length) * res / canvas_length / 2)

        # Rays casting
        start_angle = fish_orientations[j] - half_FOV  # Startvinkel
        start_angle_arc = start_angle  # Memorerar för j:te partikeln
        for ray in range(casted_rays):
            if visual_debug:
                canvas.coords(fish_canvas_rays_graphics[j][ray],
                              (fish_coords[j, 0] + canvas_length) * res / canvas_length / 2,
                              (fish_coords[j, 1] + canvas_length) * res / canvas_length / 2,
                              (fish_coords[j, 0] + fish_ray_radius * np.cos(
                                  start_angle) + canvas_length) * res / canvas_length / 2,
                              (fish_coords[j, 1] + fish_ray_radius * np.sin(
                                  start_angle) + canvas_length) * res / canvas_length / 2)
            rays_coords[j][ray] = [fish_coords[j, 0] + fish_ray_radius * np.cos(start_angle),
                                   fish_coords[j, 1] + fish_ray_radius * np.sin(start_angle)]
            rays_angle_relative_velocity[j][ray] = start_angle
            start_angle += step_angle  # Uppdaterar vinkel för ray

        # # Overlapp fishes
        fish_distances = calculate_distance(fish_coords, fish_coords[j])
        angle = np.arctan2(fish_coords[:, 1] - fish_coords[j, 1], fish_coords[:, 0] - fish_coords[j, 0])  # Directions of others array from the particle
        overlapp = fish_distances < (2 * fish_graphic_radius)  # Applying
        overlapp[j] = False  # area extraction
        for ind in np.where(overlapp)[0]:
            fish_coords[j,0] = fish_coords[j,0] + (fish_distances[ind] - 2 * fish_graphic_radius) * np.cos(angle[ind]) / 2
            fish_coords[j,1] = fish_coords[j,1] + (fish_distances[ind] - 2 * fish_graphic_radius) * np.sin(angle[ind]) / 2
            fish_coords[ind] = fish_coords[ind] - (fish_distances[ind] - 2 * fish_graphic_radius) * np.cos(angle[ind]) / 2
            fish_coords[ind] = fish_coords[ind] - (fish_distances[ind] - 2 * fish_graphic_radius) * np.sin(angle[ind]) / 2

        # Obstacle Avoidance
        avoid_angle = 0
        weight_boolean_avoid = False
        weight_boolean_vicsek = True

        detect_obst_in_radius_info = detect_obst_in_radius(fish_coords[j]) # [[[index vägg],[index rekt],[index circ]]]
        obst_type_in_radius = detect_obst_in_radius_info[1] # De hindren i interaktionsradien
        if detect_obst_in_radius_info[0]:
            detect_info = detect_closest_obst(rays_coords[j], fish_coords[j],
                                              obst_type_in_radius)  # [[hindertyp],[hinderindex],[ray_boolean]]
            closest_obst_distance, ray_boolean = detect_info[0], detect_info[1]  # Tilldelar namn
            avoid_angle = avoid_obstacle(closest_obst_distance,ray_boolean) if closest_obst_distance != np.inf else 0
            if np.abs(avoid_angle) > 0: # Prioriterar avoidance
                weight_boolean_vicsek = False # Stänger av vicsek
                weight_boolean_avoid = True

            # rect_obst_detect_ind = detect_obst_in_radius_info[1][0] # De hindren som detekterats
            # circ_obst_detect_ind = detect_obst_in_radius_info[1][1]
            #
            # # Overlapp circular obstacles
            # circ_obst_distances = calculate_distance(circ_obst_coords[circ_obst_detect_ind], fish_coords[j])
            # angle = np.arctan2(circ_obst_coords[circ_obst_detect_ind, 1] - fish_coords[j, 1], circ_obst_coords[circ_obst_detect_ind, 0] - fish_coords[j, 0])  # Directions of others array from the particle
            # overlap = circ_obst_distances < (fish_graphic_radius + circ_obst_radius[circ_obst_detect_ind])  # Applying
            #
            # for ind in np.where(overlap)[0]:
            #     fish_coords[j,0] = fish_coords[j,0] + (circ_obst_distances[ind] - (fish_graphic_radius + circ_obst_radius[circ_obst_detect_ind[ind]]) ) * np.cos(angle[ind])
            #     fish_coords[j,1] = fish_coords[j,1] + (circ_obst_distances[ind] - (fish_graphic_radius + circ_obst_radius[circ_obst_detect_ind[ind]])) * np.sin(angle[ind])

            # Overlap rectangular obstacles
            # R = fish_graphic_radius
            # Xc, Yc = fish_coords[j,0], fish_coords[j,1]  # Fiskens koordinater
            # X1, Y1 = rect_obst_corner_coords[rect_obst_detect_ind,0,0], rect_obst_corner_coords[rect_obst_detect_ind,0,1] # Ena hörnet
            # X2, Y2 = rect_obst_corner_coords[rect_obst_detect_ind,3,0],rect_obst_corner_coords[rect_obst_detect_ind,3,1] # Andra hörnet
            #
            # NearestX = np.maximum(X1, np.minimum(Xc, X2)) # Tar fram de närmsta punkten
            # NearestY = np.maximum(Y1, np.minimum(Yc, Y2))
            #
            # Dx =  Xc - NearestX # Avståndet från närmsta punkten på rektangeln till fiskens centrum
            # Dy = Yc - NearestY
            #
            # fish_inside_rect_obst = (Dx * Dx + Dy * Dy) <= R *R
            # for ind in np.where(fish_inside_rect_obst)[0]:
            #     angle = np.arctan2(Dy[ind], Dx[ind])  # Directions of others array from the particle
            #     normal_distance = np.sqrt(Dx[ind]**2 + Dy[ind]**2)
            #     fish_coords[j,0] = fish_coords[j,0] + np.absolute(normal_distance - R) * np.cos(angle)
            #     fish_coords[j,1] = fish_coords[j,1] + np.absolute(normal_distance  - R) * np.sin(angle)

         # Overlapp circular obstacles
        circ_obst_distances = calculate_distance(circ_obst_coords, fish_coords[j])
        angle = np.arctan2(circ_obst_coords[:, 1] - fish_coords[j, 1], circ_obst_coords[:, 0] - fish_coords[j, 0])  # Directions of others array from the particle
        overlapp = circ_obst_distances < (fish_graphic_radius + circ_obst_radius)  # Applying
        for ind in np.where(overlapp)[0]:
            fish_coords[j,0] = fish_coords[j,0] + (circ_obst_distances[ind] - (fish_graphic_radius + circ_obst_radius[ind]) ) * np.cos(angle[ind])
            fish_coords[j,1] = fish_coords[j,1] + (circ_obst_distances[ind] - (fish_graphic_radius + circ_obst_radius[ind])) * np.sin(angle[ind])

        # Overlap rectangular obstacles
        R = fish_graphic_radius
        Xc, Yc = fish_coords[j,0], fish_coords[j,1]  # Fiskens koordinater
        X1, Y1 = rect_obst_corner_coords[:,0,0], rect_obst_corner_coords[:,0,1] # Ena hörnet
        X2, Y2 = rect_obst_corner_coords[:,3,0],rect_obst_corner_coords[:,3,1] # Andra hörnet

        NearestX = np.maximum(X1, np.minimum(Xc, X2)) # Tar fram de närmsta punkten
        NearestY = np.maximum(Y1, np.minimum(Yc, Y2))

        Dx = NearestX - Xc # Avståndet från närmsta punkten på rektangeln till fiskens centrum
        Dy = NearestY - Yc

        fish_inside_rect_obst = (Dx * Dx + Dy * Dy) <= R *R
        for ind in np.where(fish_inside_rect_obst)[0]:
            normal_vec = fish_coords[j] - np.array([NearestX[ind],NearestY[ind]]) # Normalens vektor
            normal_distance = np.linalg.norm(normal_vec)
            angle = np.arctan2(normal_vec[1], normal_vec[0])  # Directions of others array from the particle
            fish_coords[j,0] = fish_coords[j,0] + np.absolute(normal_distance - R) * np.cos(angle)
            fish_coords[j,1] = fish_coords[j,1] + np.absolute(normal_distance  - R) * np.sin(angle)

        inter_fish_distances = calculate_distance(fish_coords, fish_coords[j])  # Räknar ut avstånd mellan fisk j och alla andra fiskar
        fish_in_interaction_radius = inter_fish_distances < fish_interaction_radius  # Vilka fiskar är inom en fisks interraktionsradie
        fish_orientations[j] =  weight_boolean_vicsek * (np.angle(
            np.sum(np.exp(fish_orientations[fish_in_interaction_radius] * 1j))) \
                                                         +  fish_noise * np.random.uniform(-1 / 2,1 / 2)) + weight_boolean_avoid * (fish_orientations[j] +avoid_angle)

    tk.title('Iteration =' + str(t))
    tk.update()  # Update animation frame
    time.sleep(0.01)  # Wait between loops
Tk.mainloop(canvas)  # Release animation handle (close window to finish)
