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

# Parameters of the fishes
fish_interaction_radius = 10  # Interaction radius
fish_graphic_radius = 2  # Radius of agent
fish_noise = 0.1  # Diffusional noise constant
fish_arrow_length = fish_interaction_radius/2


# Raycasting
step_angle = 2 * np.arctan(fish_graphic_radius / fish_interaction_radius)
casted_rays = 6
FOV_angle = step_angle * (casted_rays - 1)  # Field of view angle
half_FOV = FOV_angle / 2

# Parameters for shark

# Physical parameters of the system
simulation_iterations = 100000  # Simulation time
time_step = 0.03  # Time step
canvas_length = 100  # Size of box
fish_speed = 20  # Particle velocity
fish_count = 10 # Number of particles

x = np.array(np.random.rand(fish_count) * 2 * canvas_length - canvas_length)
y = np.array(np.random.rand(fish_count) * 2 * canvas_length - canvas_length)
fish_coords = np.column_stack((x, y))
fish_orientations = np.random.rand(fish_count) * 2 * np.pi  # orientations

circ_obst_coords = []
rect_obst_coords = []

circ_obst_radius = []
rect_obst_width = []
rect_obst_height = []

with open('Obstacles', 'r') as filestream:
    next(filestream) # Skip first row
    for line in filestream: # Read every row
        if line is not "\n":
            currentline = line.split(',')
            if('None' not in currentline[:3]):
                circ_obst_coords.append( [float(currentline[0]), float(currentline[1])] )
                circ_obst_radius.append(float(currentline[2]))
            if('None'not in currentline[3:]):
                rect_obst_coords.append( [float(currentline[3]), float(currentline[4])] )
                rect_obst_width.append(float(currentline[5]))
                rect_obst_height.append(float(currentline[6]))
    circ_obst_coords, rect_obst_coords =  np.array(circ_obst_coords), np.array(rect_obst_coords)


obst_type = ['wall', 'rect', 'circ']
obst_coords = [[[0, 0]], rect_obst_coords, circ_obst_coords]
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
fish_direction_arrow_graphics = []


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
        calculate_rectangle_corner_coordinates(rect_obst_coords[i], rect_obst_width[i], rect_obst_height[i]))


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
                               outline=ccolor[0], fill=ccolor[0]))
        fish_interaction_radius_canvas_graphics.append(
            canvas.create_oval((fish_coords[j][0] - fish_interaction_radius + canvas_length) * res / canvas_length / 2,
                               (fish_coords[j][1] - fish_interaction_radius + canvas_length) * res / canvas_length / 2,
                               (fish_coords[j][0] + fish_interaction_radius + canvas_length) * res / canvas_length / 2,
                               (fish_coords[j][1] + fish_interaction_radius + canvas_length) * res / canvas_length / 2,
                               outline=ccolor[2], width=1))
        fish_direction_arrow_graphics.append(canvas.create_line( (fish_coords[j][0]  + fish_graphic_radius *np.cos(fish_orientations[j]) + canvas_length) * res / canvas_length / 2,
                                                                 (fish_coords[j][1] + fish_graphic_radius *np.sin(fish_orientations[j]) + canvas_length) * res / canvas_length / 2,
                                                                 (fish_coords[j][0] +  (fish_graphic_radius + fish_arrow_length ) *np.cos(fish_orientations[j]) + canvas_length) * res / canvas_length / 2,
                                                                 (fish_coords[j][1] + (fish_graphic_radius + fish_arrow_length ) *np.sin(fish_orientations[j]) + canvas_length) * res / canvas_length / 2,arrow=LAST ))


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
        circ_obst_canvas_graphics.append(canvas.create_oval((circ_obst_coords[j][0] - circ_obst_radius[j] + canvas_length) * res / canvas_length / 2,
                                                            (circ_obst_coords[j][1] - circ_obst_radius[j] + canvas_length) * res / canvas_length / 2,
                                                            (circ_obst_coords[j][0] + circ_obst_radius[j] + canvas_length) * res / canvas_length / 2,
                                                            (circ_obst_coords[j][1] + circ_obst_radius[j] + canvas_length) * res / canvas_length / 2,
                                                            outline=ccolor[5], fill=ccolor[3]))
# Ritar rektangulära hinder
def draw_rectangular_obstacles():
    for j in range(rect_obst_coords.shape[0]):
        rect_obst_canvas_graphics.append(canvas.create_rectangle((rect_obst_coords[j][0] + rect_obst_width[j] + canvas_length) * res / canvas_length / 2,
                                                                 (rect_obst_coords[j][1] + rect_obst_height[j] + canvas_length) * res / canvas_length / 2,
                                                                 (rect_obst_coords[j][0] - rect_obst_width[j] + canvas_length) * res / canvas_length / 2,
                                                                 (rect_obst_coords[j][1] - rect_obst_height[j] + canvas_length) * res / canvas_length / 2,
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

    x1 = rectangle_corner_coords[0][0]
    y1 = rectangle_corner_coords[0][1]
    x2 = rectangle_corner_coords[1][0]
    y2 = rectangle_corner_coords[1][1]
    x3 = rectangle_corner_coords[2][0]
    y3 = rectangle_corner_coords[2][1]
    x4 = rectangle_corner_coords[3][0]
    y4 = rectangle_corner_coords[3][1]

    A = (triangle_area(x1, y1, x2, y2, x3, y3) +  # Calculate area of rectangel
         triangle_area(x1, y1, x4, y4, x3, y3))
    A1 = triangle_area(x, y, x1, y1, x2, y2)  # Calculate area of triangle PAB
    A2 = triangle_area(x, y, x2, y2, x3, y3)  # Calculate area of triangle PBC
    A3 = triangle_area(x, y, x3, y3, x4, y4)  # Calculate area of triangle PCD
    A4 = triangle_area(x, y, x1, y1, x4, y4)  # Calculate area of triangle PAD

    # Check if sum of A1, A2, A3 and A4 is same as A
    # 0.01 Felmarginal då datorn inte räknar exakt
    if (outside):
        return (A1 + A2 + A3 + A4 - 0.1 > A)
    else:
        return (A1 + A2 + A3 + A4 < A)


# Har tagit bort modulo beräkningen
def calculate_distance(coords, coord):  # Räknar ut avstånd mellan punkterna coords och punkten coord
    return np.sqrt((coords[:, 0] - coord[0]) ** 2 + (coords[:, 1] - coord[1]) ** 2)


# Har tagit bort modulo beräkningen
def update_position(coords, speed, orientations):  # Uppdaterar en partikels position
    # coords[:, 0] = (coords[:, 0] + speed * np.cos(orientations) * time_step + canvas_length)  - canvas_length
    # coords[:, 1] = (coords[:, 1] + speed * np.sin(orientations) * time_step + canvas_length)  - canvas_length
    coords[:, 0] = (coords[:, 0] + speed * np.cos(orientations) * time_step + canvas_length) % (
            2 * canvas_length) - canvas_length
    coords[:, 1] = (coords[:, 1] + speed * np.sin(orientations) * time_step + canvas_length) % (
            2 * canvas_length) - canvas_length
    return coords


def detect_obst_in_radius(fish_coord):
    obst_type_in_radius = [[], [], []] # Lista med index för de hinder som detekterats innanför interaktionsradien
    detected = []
    rect_obst_in_radius = calculate_distance(rect_obst_coords,fish_coord) - rect_obst_width < fish_interaction_radius
    circ_obst_in_radius = calculate_distance(circ_obst_coords,fish_coord) - circ_obst_radius < fish_interaction_radius
    is_outside_wall = (canvas_length - np.abs(fish_coord) -fish_interaction_radius) < 0
    if True in is_outside_wall:
        obst_type_in_radius[0].append(0)
        detected.append(True)
    elif True in rect_obst_in_radius:
        obst_type_in_radius[1].extend([index for index, element in enumerate(rect_obst_in_radius) if element])
        detected.append(True)
    elif True in circ_obst_in_radius:
        obst_type_in_radius[2].extend([index for index, element in enumerate(circ_obst_in_radius) if element])
        detected.append(True)
    return [True in detected,obst_type_in_radius]

def detect_closest_obst(ray_coords, fish_coord, obst_type_in_radius):
    obst_type = ['wall', 'rect', 'circ']
    n_rays = len(ray_coords)
    obst_type_detect = [[],[],[]] # Lista med vilken typ av hinder den känner av Korta listan
    obst_detect = [[],[],[]] # Lista med vilken typ plus alla ray Långa listan
    closest_obst_all = [[],[],[]] # Lista med (ett enda) hinder index, strålavstånd och boolean
    list_obst_index_raydist_raybool = [[],[],[]] # Lista med (flera) hinder index, strålavstånd och boolean
    common_ray_boolean = [False for i in range(n_rays)] # Boolean relativ partikeln

    for type in range(len(obst_type)):
        for k in range(len(obst_type_in_radius[type])):
            obst_index = obst_type_in_radius[type][k] # Hinder index
            if obst_type[type]=='wall':
                booleans = [ is_point_outside_rectangle(wall_corner_coords , ray_coords[i], True) for i in range(n_rays)] # Kollar om ray coordinaten är utanför
                detect =  True in booleans
                if detect:
                    obst_detect[type].append(detect) # Detekterat den här typen
                    closest_ray_dist = np.min(canvas_length- np.absolute(np.array(fish_coord)) - fish_graphic_radius) # Närmaste avstånd från upptagen stråle till väggen
                    list_obst_index_raydist_raybool[type].append([obst_index, closest_ray_dist, booleans]) # Lägger det i en lista
                    a , b = common_ray_boolean, booleans
                    common_ray_boolean = [a or b for a, b in zip(a, b)] # Merga ihop boolean till boolean
            elif obst_type[type]=='rect':
                booleans = [is_point_outside_rectangle(rect_obst_corner_coords[obst_index],ray_coords[i], False) for i in range(n_rays)]
                detect =  True in booleans
                if detect:
                    obst_detect[type].append(detect)
                    closest_ray_dist = np.min( [np.linalg.norm(np.array(rect_obst_coords[obst_index])
                                                               - np.array(ray_coords[index])) for index, element in enumerate(booleans) if element] )
                    list_obst_index_raydist_raybool[type].append([obst_index, closest_ray_dist , booleans])
                    # Tar fram endast de avstånden som en ray är träffad, och minsta avstånden av dessa
                    a , b = common_ray_boolean, booleans # Merga ihop boolean till boolean
                    common_ray_boolean = [a or b for a, b in zip(a, b)]
            elif obst_type[type]=='circ':
                booleans = [is_point_inside_circle(circ_obst_coords[obst_index], ray_coords[i], circ_obst_radius[k]) for i in range(n_rays)]
                detect =  True in booleans
                if detect :
                    obst_detect[type].append(detect)
                    closest_ray_dist = np.min( [np.linalg.norm(np.array(circ_obst_coords[obst_index])
                                                               -np.array(ray_coords[index])) for index, element in enumerate(booleans) if element] )
                    list_obst_index_raydist_raybool[type].append([obst_index, closest_ray_dist , booleans])
                    a , b = common_ray_boolean, booleans # Merga ihop boolean till boolean
                    common_ray_boolean = [a or b for a, b in zip(a, b)]
        obst_type_detect[type] = True in obst_detect[type]
        if obst_type_detect[type]:
            closest_obst_all[type] = min(list_obst_index_raydist_raybool[type], key = lambda x: x[1])
        else:
            closest_obst_all[type] = [-1,np.inf,False]
    min_dist_type =  min(closest_obst_all, key = lambda x: x[1]) # Ger den array med minsta avståndet i [[],[],[]]
    closest_obst_type = closest_obst_all.index(min_dist_type)
    closest_obst_index = min_dist_type[0]
    ray_boolean = min_dist_type[2] if all(common_ray_boolean) else common_ray_boolean # Om alla är träffade, tar den närmsta
    result = [obst_type[closest_obst_type], closest_obst_index, ray_boolean]
    return result

def avoid_obstacle(closest_type, closest_obst, ray_boolean):
    closest_obst_distance = 0
    if closest_type == 'circ':
        closest_obst_distance = np.linalg.norm(circ_obst_coords[closest_obst] -
                                               fish_coords[j]) - circ_obst_radius[closest_obst] - fish_graphic_radius
    elif closest_type == 'rect':
        closest_obst_distance = np.linalg.norm(rect_obst_coords[closest_obst] -
                                               fish_coords[j]) - rect_obst_width[closest_obst] - fish_graphic_radius
    elif closest_type == 'wall':
        closest_obst_distance = np.min(canvas_length - np.absolute(fish_coords[j]) - fish_graphic_radius)

    if not ray_boolean[int(len(ray_boolean) / 2 - 1)] and not ray_boolean[int(len(ray_boolean) / 2)] :
        sign = 0
    else:
        if all(ray_boolean):
            sign = 1
        else:
            i = 1
            first_free_index = int(len(ray_boolean) / 2) - 1
            while ray_boolean[first_free_index] :
                first_free_index += i * (-1) ** (i - 1)
                i += 1
            sign = -1  if(first_free_index <= 2) else 1

    angle_weight = np.pi/4/closest_obst_distance*sign
    return  angle_weight


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
                           1] + fish_interaction_radius + canvas_length) * res / canvas_length / 2)
        canvas.coords(fish_direction_arrow_graphics[j],
                      (fish_coords[j][0]  + fish_graphic_radius *np.cos(fish_orientations[j]) + canvas_length) * res / canvas_length / 2,
                      (fish_coords[j][1] + fish_graphic_radius *np.sin(fish_orientations[j]) + canvas_length) * res / canvas_length / 2,
                      (fish_coords[j][0] +  (fish_graphic_radius + fish_arrow_length ) *np.cos(fish_orientations[j]) + canvas_length) * res / canvas_length / 2,
                      (fish_coords[j][1] + (fish_graphic_radius + fish_arrow_length ) *np.sin(fish_orientations[j]) + canvas_length) * res / canvas_length / 2 )

        # Rays casting
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

        # Obstacle Avoidance
        avoid_angle = 0
        detect_obst_in_radius_info = detect_obst_in_radius(fish_coords[j])
        obst_type_in_radius = detect_obst_in_radius_info[1]
        if detect_obst_in_radius_info[0]:
            detect_info = detect_closest_obst(rays_coords[j],fish_coords[j],obst_type_in_radius) # Tar fram närmsta hindret
            closest_obst_type, closest_obst_index, ray_boolean = detect_info[0], detect_info[1], detect_info[2] # Tilldelar namn
            avoid_angle = 0 if detect_info[1] == -1 else avoid_obstacle(closest_obst_type,closest_obst_index,ray_boolean)

        inter_fish_distances = calculate_distance(fish_coords, fish_coords[
            j])  # Räknar ut avstånd mellan fisk j och alla andra fiskar
        fish_in_interaction_radius = inter_fish_distances < fish_interaction_radius  # Vilka fiskar är inom en fisks interraktionsradie
        fish_orientations[j] = np.angle(
            np.sum(np.exp(fish_orientations[fish_in_interaction_radius] * 1j))) + fish_noise * np.random.uniform(-1 / 2,
                                                                                                                 1 / 2) + avoid_angle
        # fish_orientations[j] += fish_noise * np.random.uniform(-1 / 2, 1 / 2) + wall_avoid_angle + circular_avoid_angle + rectangular_avoid_angle

    tk.title('Iteration =' + str(t))
    tk.update()  # Update animation frame
    time.sleep(0.01)  # Wait between loops
Tk.mainloop(canvas)  # Release animation handle (close window to finish)
