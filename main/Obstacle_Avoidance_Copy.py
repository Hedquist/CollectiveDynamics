import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk
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

# Raycasting
FOV_angle = np.pi / 3  # Field of view angle
half_FOV = FOV_angle / 2
casted_rays = 6
step_angle = FOV_angle / (casted_rays - 1)

# Parameters of the fishes
fish_interaction_radius = 10  # Interaction radius
fish_graphic_radius = 2  # Radius of agent
fish_noise = 0.1  # Diffusional noise constant

# Parameters for shark

# Parameters for the obstacles
obstacle_radius = 5
obstacle_rectangular_width = 4
obstacle_rectangular_height = 4

# Physical parameters of the system
simulation_iterations = 100000  # Simulation time
time_step = 0.03  # Time step
canvas_length = 100  # Size of box
fish_speed = 20  # Particle velocity
fish_count = 20  # Number of particles

x = np.array(np.random.rand(fish_count) * 2 * canvas_length - canvas_length)
y = np.array(np.random.rand(fish_count) * 2 * canvas_length - canvas_length)
fish_coords = np.column_stack((x, y))
fish_orientations = np.random.rand(fish_count) * 2 * np.pi  # orientations

# Koordinater för runda hinder
x_circ_obst = [(0.5 * 2 - 1) * canvas_length, (0.75 * 2 - 1) * canvas_length, (0.5 * 2 - 1) * canvas_length,
               (0.25 * 2 - 1) * canvas_length, (0.5 * 2 - 1) * canvas_length]
y_circ_obst = [(0.25 * 2 - 1) * canvas_length, (0.5 * 2 - 1) * canvas_length, (0.75 * 2 - 1) * canvas_length,
               (0.5 * 2 - 1) * canvas_length, (0.5 * 2 - 1) * canvas_length]
circ_obst_coords = np.array(list(zip(x_circ_obst, y_circ_obst)))

# Koordinater för rektangulära hinder, samt dess hörner
x_rect_obst = [(0.25 * 2 - 1) * canvas_length, (0.75 * 2 - 1) * canvas_length, (0.75 * 2 - 1) * canvas_length,
               (0.25 * 2 - 1) * canvas_length]
y_rect_obst = [(0.25 * 2 - 1) * canvas_length, (0.25 * 2 - 1) * canvas_length, (0.75 * 2 - 1) * canvas_length,
               (0.75 * 2 - 1) * canvas_length]
rect_obst_coords = np.array(list(zip(x_rect_obst, y_rect_obst)))
wall_corner_coords = np.array(
    [[canvas_length, canvas_length], [-canvas_length, canvas_length], [-canvas_length, -canvas_length],
     [canvas_length, -canvas_length]])
rect_obst_corner_coords = []

# Diverse tomma listor, mest grafik
circ_and_rect_obst_coords = np.concatenate((circ_obst_coords, rect_obst_coords))
fish_canvas_graphics = []
fish_interaction_radius_canvas_graphics = []
fish_canvas_rays_graphics = [[] for i in range(fish_count)]
rays_coords = [[] for i in range(fish_count)]
rays_angle_relative_velocity = [[] for i in range(fish_count)]
circ_obst_canvas_graphics = []
rect_obst_canvas_graphics = []


# Ortogonalprojektion
def proj_of_u_on_v(u, v):
    v_norm = np.sqrt(sum(v ** 2))
    return (np.dot(u, v) / v_norm ** 2) * v


# Vanlig avståndsfunktion mellan två punkter
def distance(r1, r2):
    return np.sqrt((r1[0] - r2[0]) ** 2 + (r1[1] - r2[1]) ** 2)


# Ta fram hörnen till rektangulära hinder
def calculate_rectangle_corner_coordinates(position, base, height):
    x_c = position[0]
    y_c = position[1]
    h = float(height)
    b = float(base)

    A = [x_c - b, y_c - h]
    B = [x_c + b, y_c - h]
    C = [x_c - b, y_c + h]
    D = [x_c + b, y_c + h]

    return [A, B, C, D]


# Ta fram hörnen till rektangulära hinder och lägg det i en lista
for i in range(len(rect_obst_coords)):
    rect_obst_corner_coords.append(
        calculate_rectangle_corner_coordinates(rect_obst_coords[i], obstacle_rectangular_width,
                                               obstacle_rectangular_height))


# Funktion för att fisken startposition inte hamnar i ett hinder
def generate_fish_not_inside_obstacle_coordinates():
    for j in range(fish_count):
        while (True):  # Check if the obstacle is within agent
            distance_to_obstacles = []
            for k in range(circ_and_rect_obst_coords.shape[0]):
                distance_to_obstacles.append(np.linalg.norm(fish_coords[j] - circ_and_rect_obst_coords[k]))
            if (np.min(distance_to_obstacles) < fish_interaction_radius):
                fish_coords[j][0] = np.random.rand() * 2 * canvas_length - canvas_length
                fish_coords[j][1] = np.random.rand() * 2 * canvas_length - canvas_length
            else:
                break


# Ritar ut fiskar och dess interaktionsradie
def draw_fishes():
    for j in range(fish_count):  # Generate animated particles in Canvas
        # Convert to canvas coordinates
        fish_canvas_graphics.append(
            canvas.create_oval((fish_coords[j][0] - fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                               (fish_coords[j][1] - fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                               (fish_coords[j][0] + fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                               (fish_coords[j][1] + fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                               outline=ccolor[0], fill=ccolor[0]))  # x0,y0 - x1,y1
        fish_interaction_radius_canvas_graphics.append(
            canvas.create_oval((fish_coords[j][0] - fish_interaction_radius + canvas_length) * res / canvas_length / 2,
                               (fish_coords[j][1] - fish_interaction_radius + canvas_length) * res / canvas_length / 2,
                               (fish_coords[j][0] + fish_interaction_radius + canvas_length) * res / canvas_length / 2,
                               (fish_coords[j][1] + fish_interaction_radius + canvas_length) * res / canvas_length / 2,
                               outline=ccolor[2], width=1))  # x0,y0 - x1,y1


# Ritar ut rays och lägger dess vinkel och spetsens koordinater i en lista
def cast_rays():
    for j in range(fish_count):
        start_angle = fish_orientations[j] - half_FOV  # Startvinkel
        start_angle_arc = start_angle  # Memorerar för j:te partikeln
        for ray in range(casted_rays):
            fish_canvas_rays_graphics[j].append(
                canvas.create_line((fish_coords[j][0] + canvas_length) * res / canvas_length / 2,
                                   (fish_coords[j][1] + canvas_length) * res / canvas_length / 2,
                                   (fish_coords[j][0] + fish_interaction_radius * np.cos(
                                       start_angle) + canvas_length) * res / canvas_length / 2,
                                   (fish_coords[j][1] + fish_interaction_radius * np.sin(
                                       start_angle) + canvas_length) * res / canvas_length / 2, fill=ccolor[3]))
            rays_coords[j].append([fish_coords[j][0] + fish_interaction_radius * np.cos(start_angle),
                                   fish_coords[j][1] + fish_interaction_radius * np.sin(start_angle)])
            rays_angle_relative_velocity[j].append(start_angle)
            start_angle += step_angle  # Uppdaterar vinkel för ray


# Ritar cirkulära hinder
def draw_circular_obstacles():
    for j in range(circ_obst_coords.shape[0]):
        circ_obst_canvas_graphics.append(
            canvas.create_oval((x_circ_obst[j] - obstacle_radius + canvas_length) * res / canvas_length / 2,
                               (y_circ_obst[j] - obstacle_radius + canvas_length) * res / canvas_length / 2,
                               (x_circ_obst[j] + obstacle_radius + canvas_length) * res / canvas_length / 2,
                               (y_circ_obst[j] + obstacle_radius + canvas_length) * res / canvas_length / 2,
                               outline=ccolor[5], fill=ccolor[3]))  # x0,y0 - x1,y1))


# Ritar rektangulära hinder
def draw_rectangular_obstacles():
    for j in range(rect_obst_coords.shape[0]):
        rect_obst_canvas_graphics.append(canvas.create_rectangle(
            (x_rect_obst[j] + obstacle_rectangular_width + canvas_length) * res / canvas_length / 2,
            (y_rect_obst[j] + obstacle_rectangular_height + canvas_length) * res / canvas_length / 2,
            (x_rect_obst[j] - obstacle_rectangular_width + canvas_length) * res / canvas_length / 2,
            (y_rect_obst[j] - obstacle_rectangular_height + canvas_length) * res / canvas_length / 2,
            outline=ccolor[5], fill=ccolor[4]))  # x0,y0 - x1,y1))))


# Ger index för den rayen som inte är upptagen
def ray_index(ray_booleans):
    if (isinstance(ray_booleans, np.ndarray)):
        ray_booleans = np.ndarray.tolist(ray_booleans)
    length = len(ray_booleans)
    left_boolean = ray_booleans[:length // 2]
    right_boolean = ray_booleans[length // 2:]
    left_count = left_boolean.count(False)
    right_count = right_boolean.count(False)
    arr = np.array(ray_booleans)
    indices = np.where(arr == False)
    if (all(element == True for element in ray_booleans)):
        index = 0
    else:
        index = np.min(indices) if (left_count >= right_count) else np.max(indices)
    return index


# Beräknar arean av en triangel givet tre punkter
def triangle_area(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) +
                x2 * (y3 - y1) +
                x3 * (y1 - y2)) / 2.0)


# Kollar om en punkt är innanför eller utanför en given rektangel med fyra hörn
def is_point_outside_rectangle(rectangle_corner_coords, point, outside):
    x = point[0]
    y = point[1]

    x1 = rectangle_corner_coords[0][0]
    y1 = rectangle_corner_coords[0][1]
    x2 = rectangle_corner_coords[1][0]
    y2 = rectangle_corner_coords[1][1]
    x3 = rectangle_corner_coords[2][0]
    y3 = rectangle_corner_coords[2][1]
    x4 = rectangle_corner_coords[3][0]
    y4 = rectangle_corner_coords[3][1]

    A = (triangle_area(x1, y1, x2, y2, x3, y3) +
         triangle_area(x1, y1, x4, y4, x3, y3))

    # Calculate area of triangle PAB
    A1 = triangle_area(x, y, x1, y1, x2, y2)

    # Calculate area of triangle PBC
    A2 = triangle_area(x, y, x2, y2, x3, y3)

    # Calculate area of triangle PCD
    A3 = triangle_area(x, y, x3, y3, x4, y4)

    # Calculate area of triangle PAD
    A4 = triangle_area(x, y, x1, y1, x4, y4);

    # Check if sum of A1, A2, A3
    # and A4 is same as A
    # 0.01 Felmarginal då datorn inte räknar exakt
    if (outside):
        return (A1 + A2 + A3 + A4 - 0.1 > A)
    else:
        return (A1 + A2 + A3 + A4 < A)


# Har tagit bort modulo beräkningen
def calculate_distance(coords, coord):  # Räknar ut avstånd mellan punkterna coords och punkten coord
    # return np.minimum(
    # np.sqrt(((coords[:, 0]) % (2 * canvas_length) - (coord[0]) % (2 * canvas_length)) ** 2 + (
    # (coords[:, 1]) % (2 * canvas_length) - (coord[1]) % (2 * canvas_length)) ** 2))
    return np.sqrt((coords[:, 0] - coord[0]) ** 2 + (coords[:, 1] - coord[1]) ** 2)


# Har tagit bort modulo beräkningen
def update_position(coords, speed, orientations):  # Uppdaterar en partikels position
    coords[:, 0] = (coords[:, 0] + speed * np.cos(orientations) * time_step + canvas_length) - canvas_length
    coords[:, 1] = (coords[:, 1] + speed * np.sin(orientations) * time_step + canvas_length) - canvas_length

    '''coords[:, 0] = (coords[:, 0] + speed * np.cos(orientations) * time_step + canvas_length)% (
            2 * canvas_length)  - canvas_length
    coords[:, 1] = (coords[:, 1] + speed * np.sin(orientations) * time_step + canvas_length)% (
        2 * canvas_length)  - canvas_length '''
    return coords


# Ger sant om någon ray är utanför
def wall_detect(rays_coords):
    rays_outside_wall = [False for number_of_rays in range(len(rays_coords))]
    # Kollar om någon ray är utanför boxen
    for ray in range(len(rays_coords)):
        rays_outside_wall[ray] = is_point_outside_rectangle(wall_corner_coords, rays_coords[ray], True)
    return (
        (
        True in rays_outside_wall, rays_outside_wall))  # Returnera sant eller falskt plus en lista vilka ray är utanför


# Returnera vinkel för att undvika vägg
def avoid_wall(fish_coord, ray_coords, ray_angle_relative_velocity, heading_direction, list_of_rays_outside_wall):
    distance_fish_to_wall = canvas_length - np.absolute(np.array(fish_coord))
    min_dist_fish = np.absolute(np.min(distance_fish_to_wall))

    distance_rays_to_wall = np.absolute(np.array(ray_coords)) - canvas_length
    min_dist_ray_xy = min(distance_rays_to_wall, key=lambda x: (x[0], x[1]))  # Hittar minsta x och y
    if (min_dist_ray_xy[0] < min_dist_ray_xy[1]):
        min_dist_ray = min_dist_ray_xy[0]
        min_index = np.where(distance_rays_to_wall == min_dist_ray_xy[0])
    else:
        min_dist_ray = min_dist_ray_xy[1]
        min_index = np.where(distance_rays_to_wall == min_dist_ray_xy[1])

    angle_weight = 0.5 * np.pi * (1 / (min_dist_fish + fish_interaction_radius))  # Storleken på vinkeländringen

    if (all(element == True for element in list_of_rays_outside_wall)):  # Om alla upptagna, välj den minst utstickande
        ray_index_not_occupied = min_index[0][0]
        angle_weight *= 3
    else:
        ray_index_not_occupied = ray_index(list_of_rays_outside_wall)

    sign = np.sign(ray_angle_relative_velocity[ray_index_not_occupied] - heading_direction)
    avoid_wall_angle = angle_weight * sign

    return avoid_wall_angle


# Kolla om en flerdimensionell array har true i sig
def has_true(arr):
    return any(chain(*arr))


# Index for den ray som är närmast hindret
def closest_ray_to_obstacle_index(ray_coords, obstacle_coord):
    distances = []
    for j in range(len(rays_coords)):
        distances.append(np.linalg.norm(ray_coords[j] - obstacle_coord))
    return distances.index(min(distances))


# Detekterar om ray ligger innanför cirkel, returna en boolean
def circ_obst_detect(rays_coords, circular_obstacle_coords):
    rays_inside_circular_obstacle = []
    for j in range(len(circular_obstacle_coords)):
        dist = calculate_distance(np.array(rays_coords), circular_obstacle_coords[j])
        rays_inside_circular_obstacle.append(dist < obstacle_radius)
    return has_true(np.array(rays_inside_circular_obstacle))


# Ger index för närmsta hinder från fisken
def circ_obst_closest_index(fish_coord, rays_coord, circular_obstacle_coords):
    distances = []
    for j in range(len(circular_obstacle_coords)):
        distances.append(np.linalg.norm(fish_coord - circular_obstacle_coords[j]))
    return distances.index(min(distances))


# Ger en lista med vilka ray som är innanför hindret
def circ_obst_boolean_rays(rays_coords, circular_obstacle_coords):
    rays_inside_circular_obstacle = []
    for j in range(len(circular_obstacle_coords)):
        dist = calculate_distance(np.array(rays_coords), circular_obstacle_coords[j])
        rays_inside_circular_obstacle.append(dist < obstacle_radius)
    return rays_inside_circular_obstacle


# Ger vinkel som ska undvikas
def circ_obst_avoid(fish_coord, ray_angles, heading_direction, index, rays_booleans, ray_coords):
    distance_fish_to_circular_obstacle = np.linalg.norm(np.array(fish_coord) - np.array(circ_obst_coords[index][:]))
    angle_weight = 0.5 * np.pi * (1 / (distance_fish_to_circular_obstacle))  # Storleken på vinkeländringen
    ray_index_not_occupied = ray_index(rays_booleans)
    sign = np.sign(ray_angles[ray_index_not_occupied] - heading_direction)
    avoid_wall_angle = 1 * angle_weight * sign
    return avoid_wall_angle


def rect_obst_detect(rays_coords, rectangular_obstacle_coords, rectangular_obstacle_corners):
    rays_inside_rectangular_obstacle = [[] for i in range(len(rectangular_obstacle_coords))]
    for rectangle in range(len(rectangular_obstacle_coords)):
        for ray in range(len(rays_coords)):
            rays_inside_rectangular_obstacle[rectangle].append(
                is_point_outside_rectangle(rectangular_obstacle_corners[rectangle], rays_coords[ray], False))
    # print(rays_inside_rectangular_obstacle)
    return has_true(np.array(rays_inside_rectangular_obstacle))


def rect_obst_closest_index(fish_coord, rectangular_obstacle_coords):
    distances = []
    for j in range(len(rectangular_obstacle_coords)):
        distances.append(np.linalg.norm(fish_coord - rectangular_obstacle_coords[j]))
    return distances.index(min(distances))


def rect_obst_boolean_rays(rays_coords, rectangular_obstacle_coords, rectangular_obstacle_corners):
    rays_inside_rectangular_obstacle = [[] for i in range(len(rectangular_obstacle_coords))]
    for rectangle in range(len(rectangular_obstacle_coords)):
        for ray in range(len(rays_coords)):
            rays_inside_rectangular_obstacle[rectangle].append(
                is_point_outside_rectangle(rectangular_obstacle_corners[rectangle], rays_coords[ray], False))
    return rays_inside_rectangular_obstacle


def rect_obst_avoid(fish_coord, ray_angles, heading_direction, index, rays_booleans):
    distance_fish_to_circular_obstacle = np.linalg.norm(np.array(fish_coord) - np.array(circ_obst_coords[index]))
    angle_weight = 0.5 * np.pi * (1 / (distance_fish_to_circular_obstacle))  # Storleken på vinkeländringen
    ray_index_not_occupied = ray_index(rays_booleans)
    sign = np.sign(ray_angles[ray_index_not_occupied] - heading_direction)
    avoid_wall_angle = 5 * angle_weight * sign
    return avoid_wall_angle


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
        canvas.coords(fish_interaction_radius_canvas_graphics[j],
                      (fish_coords[j][0] - fish_interaction_radius + canvas_length) * res / canvas_length / 2,
                      (fish_coords[j][1] - fish_interaction_radius + canvas_length) * res / canvas_length / 2,
                      (fish_coords[j][0] + fish_interaction_radius + canvas_length) * res / canvas_length / 2,
                      (fish_coords[j][
                           1] + fish_interaction_radius + canvas_length) * res / canvas_length / 2)  # x0,y0 - x1,y1
        # Rayscating
        start_angle = fish_orientations[j] - half_FOV  # Startvinkel
        start_angle_arc = start_angle  # Memorerar för j:te partikeln
        for ray in range(casted_rays):
            canvas.coords(fish_canvas_rays_graphics[j][ray],
                          (fish_coords[j][0] + canvas_length) * res / canvas_length / 2,
                          (fish_coords[j][1] + canvas_length) * res / canvas_length / 2,
                          (fish_coords[j][0] + fish_interaction_radius * np.cos(
                              start_angle) + canvas_length) * res / canvas_length / 2,
                          (fish_coords[j][1] + fish_interaction_radius * np.sin(
                              start_angle) + canvas_length) * res / canvas_length / 2)

            rays_coords[j][ray] = [fish_coords[j][0] + fish_interaction_radius * np.cos(start_angle),
                                   fish_coords[j][1] + fish_interaction_radius * np.sin(start_angle)]
            rays_angle_relative_velocity[j][ray] = start_angle
            start_angle += step_angle  # Uppdaterar vinkel för ray

        wall_avoid_angle = 0
        circ_avoid_angle = 0
        rect_avoid_angle = 0

        if (wall_detect(rays_coords[j])[0]):
            wall_avoid_angle = avoid_wall(fish_coords[j], rays_coords[j], rays_angle_relative_velocity[j],
                                          fish_orientations[j], wall_detect(rays_coords[j])[1])

        if (circ_obst_detect(rays_coords[j], circ_obst_coords)):
            circ_index = circ_obst_closest_index(fish_coords[j], rays_coords[j],
                                                 circ_obst_coords)  # Index av närmaste cirkulära hinder
            circ_obst_rays = circ_obst_boolean_rays(rays_coords[j], circ_obst_coords)[
                circ_index]  # Lista med vilka rays är upptagna
            circ_avoid_angle = circ_obst_avoid(fish_coords[j], rays_angle_relative_velocity[j], fish_orientations[j],
                                               circ_index, circ_obst_rays, rays_coords[j])  # Beräkna vinkeln

        if (rect_obst_detect(rays_coords[j], rect_obst_coords, rect_obst_corner_coords)):
            rect_index = rect_obst_closest_index(fish_coords[j], rect_obst_coords)
            rect_obst_rays = rect_obst_boolean_rays(rays_coords[j], rect_obst_coords, rect_obst_corner_coords)[
                rect_index]
            rect_avoid_angle = rect_obst_avoid(fish_coords[j], rays_angle_relative_velocity[j], fish_orientations[j],
                                               rect_index, rect_obst_rays)

        inter_fish_distances = calculate_distance(fish_coords, fish_coords[
            j])  # Räknar ut avstånd mellan fisk j och alla andra fiskar
        fish_in_interaction_radius = inter_fish_distances < fish_interaction_radius  # Vilka fiskar är inom en fisks interraktionsradie
        fish_orientations[j] = np.angle(
            np.sum(np.exp(fish_orientations[fish_in_interaction_radius] * 1j))) + fish_noise * np.random.uniform(-1 / 2,
                                                                                                                 1 / 2) + wall_avoid_angle + circ_avoid_angle + rect_avoid_angle
        # fish_orientations[j] += fish_noise * np.random.uniform(-1 / 2, 1 / 2) + wall_avoid_angle + circular_avoid_angle + rectangular_avoid_angle

    tk.title('Iteration =' + str(t))
    tk.update()  # Update animation frame
    time.sleep(0.01)  # Wait between loops
Tk.mainloop(canvas)  # Release animation handle (close window to finish)
